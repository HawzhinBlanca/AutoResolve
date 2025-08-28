import psutil, torch, time, random, numpy as np
from dataclasses import dataclass

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
        time.sleep(0.1)
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
import os, json

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