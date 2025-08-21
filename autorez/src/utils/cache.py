import os, json, hashlib, numpy as np

def sha256_str(s: str):
    return hashlib.sha256(s.encode()).hexdigest()

def key(video_path, fps, window, crop, strategy, model_tag, weights_hash):
    base = f"{video_path}|{fps}|{window}|{crop}|{strategy}|{model_tag}|{weights_hash}"
    return hashlib.sha256(base.encode()).hexdigest()

def save(cache_dir, k, segments, meta):
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(os.path.join(cache_dir, f"{k}.npz"), data=segments)
    with open(os.path.join(cache_dir, f"{k}.json"), "w") as f: json.dump(meta, f)

def load(cache_dir, k):
    npz = os.path.join(cache_dir, f"{k}.npz")
    js  = os.path.join(cache_dir, f"{k}.json")
    if not (os.path.exists(npz) and os.path.exists(js)): return None, None
    arr = np.load(npz, allow_pickle=True)["data"]
    meta = json.load(open(js))
    return arr.tolist(), meta