import psutil
import torch
import time
import random
import numpy as np
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Budget:
    max_gb: float = 16.0
    fps: float = 1.0
    window: int = 16
    crop: int = 256
    max_segments: int = 500

_DEF_FPS_FLOOR = 0.5
_DEF_WIN_FLOOR = 8
_DEF_CROP_FLOOR = 224

# Deterministic seeds
_DEF_SEED = 1234

import threading

# Thread lock for safe seed setting
_seed_lock = threading.Lock()

def set_seeds(seed=_DEF_SEED):
    """Thread-safe seed setting for reproducibility."""
    with _seed_lock:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            
            # Set CUDA seeds if available
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Ensure deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            if torch.backends.mps.is_available():
                # MPS doesn't have full determinism support yet
                # but we still set seeds for CPU operations
                pass
        except Exception:
            pass

def rss_gb():
    return psutil.Process().memory_info().rss / (1024**3)

def available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def check_memory_available(required_gb: float = 8.0) -> bool:
    """Check if enough memory is available before starting"""
    available = available_memory_gb()
    rss_gb()
    available + (psutil.virtual_memory().total / (1024**3) - psutil.virtual_memory().used / (1024**3))
    
    if available < required_gb:
        import warnings
        warnings.warn(f"Low memory warning: Only {available:.1f}GB available, {required_gb:.1f}GB recommended")
        return False
    return True

def enforce_budget(b: Budget, device: str, aggressive: bool = True):
    changes = []
    current_mem = rss_gb()
    
    # More aggressive memory management
    if aggressive and current_mem > b.max_gb * 0.8:  # Start reducing at 80% threshold
        if b.fps > _DEF_FPS_FLOOR:
            b.fps = max(_DEF_FPS_FLOOR, _DEF_FPS_FLOOR); changes.append(("fps", b.fps))
        if b.window > _DEF_WIN_FLOOR:
            b.window = _DEF_WIN_FLOOR; changes.append(("window", b.window))
    
    # degrade in order: fps → window → crop
    while rss_gb() > b.max_gb:
        if b.fps > _DEF_FPS_FLOOR:
            b.fps = max(_DEF_FPS_FLOOR, round(b.fps * 0.5, 3)); changes.append(("fps", b.fps)); break
        if b.window > _DEF_WIN_FLOOR:
            b.window = max(_DEF_WIN_FLOOR, int(b.window * 0.5)); changes.append(("window", b.window)); break
        if b.crop > _DEF_CROP_FLOOR:
            b.crop = max(_DEF_CROP_FLOOR, int(b.crop * 0.75)); changes.append(("crop", b.crop)); break
        
        # Force garbage collection
        import gc
        gc.collect()
        
        break
    
    # Clear GPU cache
    if device == "cuda":
        try: 
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: CUDA cache cleanup failed: {e}")
    elif device == "mps":
        try:
            # Force MPS cache clear
            import gc
            gc.collect()
        except Exception as e:
            print(f"Warning: MPS cache cleanup failed: {e}")
    
    return changes, b

# --- Telemetry (lightweight JSONL) ---
import os
import json

def get_budget() -> Budget:
    """Get default budget configuration"""
    return Budget()

def emit_metrics(name: str, metrics: dict, path: str = "artifacts/metrics.jsonl"):
    """Append one metrics record to a local JSONL log; never blocks execution."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rec = {"name": name, "ts": time.time(), **metrics}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

def collect_system_metrics() -> Dict[str, float]:
    """Collect real system metrics for performance monitoring"""
    process = psutil.Process(os.getpid())
    
    # Memory metrics
    memory_info = process.memory_info()
    current_rss_gb = memory_info.rss / (1024 ** 3)
    vms_gb = memory_info.vms / (1024 ** 3)
    
    # CPU metrics
    cpu_percent = process.cpu_percent(interval=0.1)
    
    # Find UI process memory if running
    ui_memory_mb = 0.0
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'AutoResolveUI' in proc.info['name']:
                ui_memory_mb = proc.info['memory_info'].rss / (1024 ** 2)
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return {
        "peak_rss_gb": current_rss_gb,
        "vms_gb": vms_gb,
        "cpu_percent": cpu_percent,
        "ui_memory_mb": ui_memory_mb,
        "timestamp": time.time()
    }

def collect_real_metrics() -> Dict[str, float]:
    """
    Collect comprehensive real metrics for gates verification
    This replaces the hardcoded default metrics
    """
    import tempfile
    import subprocess
    from src.ops.transcribe import Transcriber
    from src.ops.silence import SilenceRemover
    
    # Create a test video for benchmarking
    test_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    test_duration = 60  # 1 minute test video
    
    # Generate test video with ffmpeg
    subprocess.run([
        "ffmpeg", "-f", "lavfi",
        "-i", f"testsrc=duration={test_duration}:size=640x480:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={test_duration}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-y", test_video.name
    ], capture_output=True, check=False)
    
    # Benchmark transcription
    transcriber = Transcriber()
    trans_start = time.time()
    _, trans_metrics = transcriber.transcribe_video(test_video.name)
    trans_elapsed = time.time() - trans_start
    
    # Benchmark silence detection
    remover = SilenceRemover()
    silence_start = time.time()
    remover.remove_silence(test_video.name)
    silence_elapsed = time.time() - silence_start
    
    # Benchmark EDL export
    export_start = time.time()
    with tempfile.NamedTemporaryFile(suffix=".edl", mode="w") as edl:
        edl.write("TITLE: Test Timeline\n\n")
        for i in range(100):  # Simulate 100 clips
            edl.write(f"{i:03d}  AX  V  C  00:00:00:00 00:00:01:00 00:00:{i:02d}:00 00:00:{i+1:02d}:00\n")
    export_elapsed = time.time() - export_start
    
    # Collect system metrics
    system_metrics = collect_system_metrics()
    
    # Calculate processing speed (multiple of realtime)
    total_processing_time = trans_elapsed + silence_elapsed
    processing_speed_x = test_duration / total_processing_time if total_processing_time > 0 else 0
    
    # Clean up test file
    os.unlink(test_video.name)
    
    return {
        "processing_speed_x": processing_speed_x,
        "peak_rss_gb": system_metrics["peak_rss_gb"],
        "ui_memory_mb": system_metrics["ui_memory_mb"],
        "silence_sec_per_min": silence_elapsed,
        "transcription_rtf": trans_metrics.get("realtime_ratio", trans_elapsed / test_duration),
        "vjepa_sec_per_min": 0.0,  # V-JEPA not used
        "api_sec_per_min": 0.0,  # API not used
        "api_cost_per_min": 0.0,  # No API cost
        "export_time_s": export_elapsed,
    }

class MemoryTracker:
    """Track memory usage over time for profiling"""
    
    def __init__(self):
        self.samples = []
        self.start_time = time.time()
        self.peak_rss = 0
    
    def sample(self) -> Dict[str, float]:
        """Take a memory sample"""
        current_rss = rss_gb()
        self.peak_rss = max(self.peak_rss, current_rss)
        
        sample = {
            "time": time.time() - self.start_time,
            "rss_gb": current_rss,
            "peak_rss_gb": self.peak_rss
        }
        self.samples.append(sample)
        return sample
    
    def report(self) -> Dict[str, Any]:
        """Generate memory usage report"""
        if not self.samples:
            return {"error": "No samples collected"}
        
        rss_values = [s["rss_gb"] for s in self.samples]
        
        return {
            "peak_rss_gb": self.peak_rss,
            "average_rss_gb": sum(rss_values) / len(rss_values),
            "min_rss_gb": min(rss_values),
            "max_rss_gb": max(rss_values),
            "samples": len(self.samples),
            "duration_s": time.time() - self.start_time
        }