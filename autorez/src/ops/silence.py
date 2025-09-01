import logging
import asyncio
import subprocess
from typing import Optional, List, Tuple, Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import configparser
import time
import os
import json
import tempfile
import wave

logger = logging.getLogger(__name__)

"""
Blueprint3 Ops Module - Silence Removal (COMPLIANT)
Outputs cuts.json with keep_windows and params from [silence] in conf/ops.ini
"""

CFG = configparser.ConfigParser()
CFG.read(os.getenv("OPS_INI", "conf/ops.ini"))


class SilenceRemover:
    def __init__(self):
        self.rms_thresh_db = float(CFG.get("silence", "rms_thresh_db", fallback="-34"))
        self.min_silence_s = float(CFG.get("silence", "min_silence_s", fallback="0.35"))
        self.min_keep_s = float(CFG.get("silence", "min_keep_s", fallback="0.40"))
        self.pad_s = float(CFG.get("silence", "pad_s", fallback="0.05"))
        self.duration = 0.0  # Store video duration for backend

    def detect_speech_windows(
        self, 
        audio_path: Optional[str] = None, 
        y: Optional[np.ndarray] = None, 
        sr: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        if y is None or sr is None:
            if not audio_path or not os.path.exists(audio_path):
                raise FileNotFoundError("Audio path not found and no samples provided")
            # Use standard library + NumPy only (no SciPy/LibROSA)
            sr, y = _read_wav_mono_float32(audio_path)
        
        hop_length = 512
        frame_length = 2048
        
        # Vectorized RMS calculation using sliding window
        # Pad signal
        y_padded = np.pad(y, (frame_length//2, frame_length//2), mode='constant')
        
        # Use sliding window view for efficient computation
        try:
            # Create sliding windows efficiently
            windows = sliding_window_view(y_padded, frame_length)[::hop_length]
            # Vectorized RMS calculation
            rms = np.sqrt(np.mean(windows ** 2, axis=1))
        except (ValueError, MemoryError):
            # Fallback for very large files or memory issues
            n_frames = 1 + (len(y_padded) - frame_length) // hop_length
            rms = np.zeros(n_frames)
            # Process in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, n_frames, chunk_size):
                end_idx = min(i + chunk_size, n_frames)
                chunk_starts = np.arange(i, end_idx) * hop_length
                chunk_ends = chunk_starts + frame_length
                # Vectorized within chunk
                chunk_frames = np.array([y_padded[s:e] for s, e in zip(chunk_starts, chunk_ends)])
                rms[i:end_idx] = np.sqrt(np.mean(chunk_frames ** 2, axis=1))
        
        # Convert to dB (NumPy only)
        rms_db = 20 * np.log10(rms + 1e-10) - 20 * np.log10(np.max(rms) + 1e-10)
        
        # Calculate time array
        times = np.arange(len(rms_db)) * hop_length / sr

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

    async def remove_silence_async(self, video_path: str, output_path: Optional[str] = None) -> dict:
        """Async version using subprocess for non-blocking I/O"""
        start_time = time.time()
        
        # Use tempfile for cross-platform compatibility
        tmp_wav_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="autoresolve_silence_")
        os.close(tmp_wav_fd)
        
        try:
            # Use async subprocess for non-blocking ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ac', '1', '-ar', '22050', '-f', 'wav',
                '-y', tmp_wav
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg failed: {stderr.decode()}")
                raise RuntimeError("Audio extraction failed")
            
            # Read the extracted audio
            if os.path.exists(tmp_wav):
                rate, y = _read_wav_mono_float32(tmp_wav)
                
                # Detect speech windows
                speech_windows = self.detect_speech_windows(y=y, sr=rate)
                
                # Calculate statistics
                total_duration = len(y) / rate
                self.duration = total_duration
                speech_duration = sum(end - start for start, end in speech_windows)
                silence_duration = total_duration - speech_duration
                
                result = {
                    "keep_windows": speech_windows,
                    "silence_segments": self._get_silence_segments(speech_windows, total_duration),
                    "total_duration": total_duration,
                    "speech_duration": speech_duration,
                    "silence_duration": silence_duration,
                    "processing_time": time.time() - start_time,
                    "params": {
                        "rms_thresh_db": self.rms_thresh_db,
                        "min_silence_s": self.min_silence_s,
                        "min_keep_s": self.min_keep_s,
                        "pad_s": self.pad_s
                    }
                }
                
                # Save to output if specified
                if output_path:
                    with open(output_path, 'w') as f:
                        json.dump(result, f, indent=2)
                
                return result
            else:
                raise FileNotFoundError("Audio extraction failed")
                
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)
    
    def remove_silence(self, video_path: str, output_path: Optional[str] = None) -> dict:
        """Sync wrapper for backward compatibility"""
        return asyncio.run(self.remove_silence_async(video_path, output_path))
    
    def _get_silence_segments(self, speech_windows: List[Tuple[float, float]], total_duration: float) -> List[dict]:
        """Calculate silence segments from speech windows"""
        silence_segments = []
        
        if not speech_windows:
            # Entire file is silence
            silence_segments.append({"start": 0.0, "end": total_duration})
            return silence_segments
        
        # Before first speech
        if speech_windows[0][0] > 0:
            silence_segments.append({"start": 0.0, "end": speech_windows[0][0]})
        
        # Between speech segments
        for i in range(len(speech_windows) - 1):
            silence_start = speech_windows[i][1]
            silence_end = speech_windows[i + 1][0]
            if silence_end > silence_start:
                silence_segments.append({"start": silence_start, "end": silence_end})
        
        # After last speech
        if speech_windows[-1][1] < total_duration:
            silence_segments.append({"start": speech_windows[-1][1], "end": total_duration})
        
        return silence_segments
        start_time = time.time()

        # Extract audio using ffmpeg-python (robust), fallback to PyAV
        samples = []
        rate = 22050
        # Use tempfile for cross-platform compatibility
        tmp_wav_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="autoresolve_silence_")
        os.close(tmp_wav_fd)  # Close the file descriptor, we'll use the path
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
                    # Read WAV using standard library (wave) and NumPy
                    rate, y = _read_wav_mono_float32(tmp_wav)
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
            # Simple resampling using NumPy (linear interpolation)
            if rate != 22050:
                # Calculate new length
                new_length = int(len(y) * 22050 / rate)
                # Create indices for interpolation
                old_indices = np.linspace(0, len(y) - 1, len(y))
                new_indices = np.linspace(0, len(y) - 1, new_length)
                # Linear interpolation
                y = np.interp(new_indices, old_indices, y)
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
        self.duration = original_duration  # Store for backend access
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

# === NumPy-only WAV reader (standard library + NumPy) ===
def _read_wav_mono_float32(path: str) -> tuple[int, np.ndarray]:
    """
    Read a WAV file using the standard library and return (sample_rate, mono float32 array in [-1, 1]).
    Supports 8/16/24/32-bit PCM. Avoids SciPy/LibROSA as mandated by the blueprint.
    """
    wf = wave.open(path, 'rb')
    try:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
    finally:
        wf.close()

    if sampwidth == 1:
        # Unsigned 8-bit PCM [0,255] -> [-1,1]
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        # Signed 16-bit PCM
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # Signed 24-bit PCM little-endian
        bytes_arr = np.frombuffer(frames, dtype=np.uint8)
        n = bytes_arr.size // 3
        bytes_arr = bytes_arr[: n * 3].reshape(n, 3)
        pad = np.zeros((n, 4), dtype=np.uint8)
        pad[:, :3] = bytes_arr
        int32 = pad.view('<i4').reshape(n)
        # Sign-extend 24-bit by shifting
        int32 = (int32 << 8) >> 8
        data = int32.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        # Signed 32-bit PCM
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        # Fallback: assume 16-bit PCM
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return framerate, data
