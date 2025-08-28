import logging

logger = logging.getLogger(__name__)

"""
Blueprint3 Ops Module - Silence Removal (COMPLIANT)
Outputs cuts.json with keep_windows and params from [silence] in conf/ops.ini
"""
import numpy as np
import librosa
import configparser
import time
import os
import json

CFG = configparser.ConfigParser()
CFG.read(os.getenv("OPS_INI", "conf/ops.ini"))


class SilenceRemover:
    def __init__(self):
        self.rms_thresh_db = float(CFG.get("silence", "rms_thresh_db", fallback="-34"))
        self.min_silence_s = float(CFG.get("silence", "min_silence_s", fallback="0.35"))
        self.min_keep_s = float(CFG.get("silence", "min_keep_s", fallback="0.40"))
        self.pad_s = float(CFG.get("silence", "pad_s", fallback="0.05"))

    def detect_speech_windows(self, audio_path: str | None = None, y=None, sr: int | None = None):
        if y is None or sr is None:
            if not audio_path or not os.path.exists(audio_path):
                raise FileNotFoundError("Audio path not found and no samples provided")
            y, sr = librosa.load(audio_path, sr=22050)
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)

        is_speech = rms_db > self.rms_thresh_db
        speech = []
        in_speech = False
        start = 0.0
        for i, flag in enumerate(is_speech):
            if flag and not in_speech:
                start = times[i]
                in_speech = True
            elif not flag and in_speech:
                end = times[i]
                if end - start >= self.min_keep_s:
                    speech.append((max(0.0, start - self.pad_s), end + self.pad_s))
                in_speech = False
        if in_speech:
            end = times[-1]
            if end - start >= self.min_keep_s:
                speech.append((max(0.0, start - self.pad_s), end))
        return speech

    def remove_silence(self, video_path: str, output_path: str | None = None):
        start_time = time.time()

        # Extract audio using ffmpeg-python (robust), fallback to PyAV
        samples = []
        rate = 22050
        tmp_wav = "/tmp/autoresolve_silence.wav"
        try:
            import ffmpeg as _ff
            (
                _ff
                .input(video_path)
                .output(tmp_wav, ac=1, ar=22050, f='wav')
                .overwrite_output()
                .run(quiet=True)
            )
            if os.path.exists(tmp_wav):
                try:
                    y, rate = librosa.load(tmp_wav, sr=22050)
                    samples = [y]
                finally:
                    try:
                        os.remove(tmp_wav)
                    except:
                        pass
        except Exception:
            container = None
            try:
                import av
                container = av.open(video_path)
                astreams = [s for s in container.streams if s.type == 'audio']
                if astreams:
                    stream = astreams[0]
                    rate = int(stream.rate or rate)
                    for frame in container.decode(stream):
                        pcm = frame.to_ndarray()
                        if pcm.ndim > 1:
                            pcm = pcm.mean(axis=0)
                        samples.append(pcm.astype('float32') / (np.iinfo(pcm.dtype).max if np.issubdtype(pcm.dtype, np.integer) else 1.0))
            except Exception as e:
                logger.warning(f"Audio extraction failed: {e}")
                samples = []
            finally:
                if container is not None:
                    try:
                        container.close()
                    except:
                        pass

        if samples:
            y = np.concatenate(samples)
            # Resample to 22050 if needed
            if rate != 22050:
                y = librosa.resample(y, orig_sr=rate, target_sr=22050)
                rate = 22050
            keep_windows = self.detect_speech_windows(y=y, sr=rate)
        else:
            # Graceful fallback: if no audio could be extracted, keep full duration
            duration_fallback = self._get_video_duration(video_path)
            keep_windows = [(0.0, duration_fallback)] if duration_fallback > 0 else []

        # Cleanup temp wav if created
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

        original_duration = self._get_video_duration(video_path)
        estimated_new_duration = sum(e - s for s, e in keep_windows)

        cuts = {
            "version": "3.0",
            "source_video": video_path,
            "keep_windows": [{"start": float(s), "end": float(e)} for s, e in keep_windows],
            "params": {
                "rms_thresh_db": self.rms_thresh_db,
                "min_silence_s": self.min_silence_s,
                "min_keep_s": self.min_keep_s,
                "pad_s": self.pad_s
            }
        }

        metrics = {
            "processing_time_s": time.time() - start_time,
            "segments_found": len(keep_windows),
            "compression_ratio": (estimated_new_duration / original_duration) if original_duration > 0 else 1.0
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(cuts, f, indent=2)

        return cuts, metrics

    def _get_video_duration(self, video_path: str) -> float:
        try:
            import av
            container = av.open(video_path)
            duration = float(container.duration) / 1_000_000.0 if container.duration else 0.0
            container.close()
            return duration
        except Exception:
            return 0.0


def silence_cli():
    import sys
    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.ops.silence <video_path> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/cuts.json"

    remover = SilenceRemover()
    cuts_data, metrics = remover.remove_silence(video_path, output_path)

    logger.info("Silence removal complete: segments=%d compression=%.2f", metrics["segments_found"], metrics["compression_ratio"])


if __name__ == "__main__":
    silence_cli()
