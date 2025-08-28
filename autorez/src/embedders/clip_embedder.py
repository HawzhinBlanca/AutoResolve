import torch
import av
import hashlib
import open_clip
from PIL import Image
from src.utils.memory import Budget, enforce_budget, set_seeds
from src.utils.cache import key, load, save

class CLIPEmbedder:
    def __init__(self, model_name="ViT-H-14", pretrained="laion2B-s32B-b79K", cache_dir="artifacts/cache", seed=1234):
        set_seeds(seed)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.tokenize = open_clip.get_tokenizer(model_name)
        self.cache_dir = cache_dir
        self.model_tag = f"openclip-{model_name}-{pretrained}"
        self.weights_hash = self._weights_fingerprint()

    def _weights_fingerprint(self):
        h = hashlib.sha256()
        for k, v in self.model.state_dict().items():
            h.update(k.encode()); h.update(str(tuple(v.shape)).encode()); h.update(str(v.dtype).encode())
            try: h.update(v.view(-1)[:1].detach().cpu().numpy().tobytes())
            except Exception as e:
                # Skip parameters that cannot be serialized for fingerprinting
                print(f"Warning: Skipping parameter {k} in fingerprint due to: {e}")
        return h.hexdigest()[:16]

    def _read_frames(self, path, fps=1.0):
        c = av.open(path); s = c.streams.video[0]
        frames, ts, step = [], [], int(round(s.average_rate / fps)) if s.average_rate else 30
        for i, fr in enumerate(c.decode(s)):
            if i % step == 0:
                frames.append(Image.fromarray(fr.to_ndarray(format="rgb24")))
                ts.append(float(fr.time) if fr.time else (i/float(s.average_rate or 30)))
        c.close(); return frames, ts

    @torch.inference_mode()
    def embed_segments(self, video_path, fps=1.0, window=16, strategy="temp_attn", max_segments=500):
        b = Budget(fps=fps, window=window, crop=224, max_segments=max_segments)
        k = key(video_path, b.fps, b.window, b.crop, strategy, self.model_tag, self.weights_hash)
        cached, meta = load(self.cache_dir, k)
        if cached is not None: return cached, meta

        frames, ts = self._read_frames(video_path, fps=b.fps)
        segs = []
        total_segments = min(len(frames)//b.window, b.max_segments)
        for si in range(total_segments):
            i = si * b.window
            chunk = frames[i:i+b.window]
            imgs = torch.stack([self.preprocess(im) for im in chunk]).to(self.device)
            feats = self.model.encode_image(imgs)              # [T,D]
            feats = torch.nn.functional.normalize(feats, dim=-1)
            q = feats @ feats.mean(0, keepdim=True).T          # temporal attention
            w = torch.softmax(q.squeeze(-1), dim=0)
            emb = torch.nn.functional.normalize((w[:,None]*feats).sum(0), dim=-1)
            segs.append({"t0": ts[i], "t1": ts[i+b.window-1], "emb": emb.cpu().numpy()})
            enforce_budget(b, self.device)
        meta = {"model": self.model_tag, "weights_hash": self.weights_hash,
                "fps": b.fps, "window": b.window, "crop": b.crop, "segments": len(segs)}
        save(self.cache_dir, k, segs, meta)
        return segs, meta

    @torch.inference_mode()
    def encode_text(self, texts):
        tok = self.tokenize(texts).to(self.device)
        z = self.model.encode_text(tok)
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()