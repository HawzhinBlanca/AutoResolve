import logging

logger = logging.getLogger(__name__)

"""
Blueprint3 Ops Module - Transcription (COMPLIANT)
Uses faster-whisper per blueprint, reads [transcribe] from conf/ops.ini,
and emits exact schema: {"language":"en","segments":[{"t0":...}],"meta":{"rtf":...}}
"""
import configparser
import time
import os
from faster_whisper import WhisperModel

CFG = configparser.ConfigParser()
CFG.read(os.getenv("OPS_INI", "conf/ops.ini"))


class Transcriber:
    def __init__(self, model: str | None = None, compute_type: str | None = None, device: str | None = None):
        # Resolve config
        model_name = model or CFG.get("transcribe", "model", fallback="medium")
        compute = compute_type or CFG.get("transcribe", "compute_type", fallback="int8")

        # Device auto
        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"

        self.device = device
        self.model_name = model_name
        self.compute_type = compute
        self.model = WhisperModel(model_name, device=("cpu" if device == "mps" else device), compute_type=compute)

    def transcribe_video(self, media_path: str, output_path: str | None = None):
        start_time = time.time()

        # Get duration using PyAV
        try:
            import av
            container = av.open(media_path)
            duration = float(container.duration) / 1_000_000.0 if container.duration else 0.0
            container.close()
        except Exception:
            duration = 0.0

        # Run transcription
        segments, info = self.model.transcribe(
            media_path,
            vad_filter=CFG.getboolean("transcribe", "vad", fallback=True),
            language=(None if CFG.get("transcribe", "lang", fallback="auto") == "auto" else CFG.get("transcribe", "lang")),
        )

        # Gather segments
        seg_list = []
        for s in segments:
            seg_list.append({
                "t0": float(getattr(s, "start", 0.0) or 0.0),
                "t1": float(getattr(s, "end", 0.0) or 0.0),
                "text": (getattr(s, "text", "") or "").strip()
            })

        elapsed = time.time() - start_time
        rtf = (elapsed / duration) if duration > 0 else float("inf")

        transcript = {
            "language": getattr(info, "language", ""),
            "segments": seg_list,
            "meta": {
                "rtf": float(rtf),
                "model": f"{self.model_name}-{self.compute_type}"
            }
        }

        # Persist if requested
        if output_path:
            import json
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(transcript, f, indent=2)

        metrics = {
            "duration_s": duration,
            "processing_time_s": elapsed,
            "realtime_ratio": rtf,
            "device": self.device
        }

        return transcript, metrics


def transcribe_audio(video_path: str, output_path: str | None = None) -> dict:
    transcriber = Transcriber()
    transcript, _ = transcriber.transcribe_video(video_path, output_path)
    return transcript


def transcribe_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe media to JSON (blueprint schema)")
    parser.add_argument("--audio", required=True, help="Video/audio file path")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    args = parser.parse_args()

    transcriber = Transcriber()
    transcript, metrics = transcriber.transcribe_video(args.audio, args.out)
    logger.info("Transcription complete: duration=%.1fs processing=%.1fs rtf=%.2fx segments=%d",
                metrics.get("duration_s", 0.0), metrics.get("processing_time_s", 0.0), metrics.get("realtime_ratio", 0.0), len(transcript["segments"]))


if __name__ == "__main__":
    transcribe_cli()
