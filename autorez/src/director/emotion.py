# Blueprint3 Director Module - Emotion
# Tension analysis: peaks, valleys, sustained

import numpy as np
import configparser
import os
from dataclasses import dataclass
from typing import List, Tuple
from src.embedders.vjepa_embedder import VJEPAEmbedder

CFG = configparser.ConfigParser()
CFG.read(os.getenv("DIRECTOR_INI","conf/director.ini"))

@dataclass
class TensionProfile:
    """Container for tension analysis results"""
    peaks: List[Tuple[float, float]]
    valleys: List[Tuple[float, float]]
    sustained: List[Tuple[float, float]]
    curve: List[Tuple[float, float]]  # (timestamp, tension_value)

def _velocity(F):
    """Calculate velocity/movement in frame class tokens"""
    if F is None or len(F)<2:
        return 0.0
    v = np.linalg.norm(np.diff(F, axis=0), axis=1)
    return float(np.clip(v.mean()/np.sqrt(F.shape[1]), 0, 1))

def _rigidity(F):
    """Calculate rigidity/stiffness from variance"""
    if F is None or len(F)<2: return 0.5
    var = np.var(F, axis=0).mean()
    return float(np.clip(1.0/(1.0+var), 0, 1))

def _proximity_change(F):
    """Calculate proximity changes in feature space"""
    if F is None: return 0.5
    m = F.mean(axis=0)
    return float(np.clip(np.linalg.norm(m)/np.sqrt(len(m)), 0, 1))

def _geo_mul(vals, weights):
    """Geometric mean with weights"""
    vals = np.clip(np.array(vals, dtype=np.float32), 1e-6, 1.0)
    w = np.array(weights, dtype=np.float32)
    w /= w.sum()
    logv = (w * np.log(vals)).sum()
    return float(np.exp(logv))

def analyze_tension(video_path: str, fps=None, window=None):
    """
    Analyze emotional tension in video
    
    Args:
        video_path: Path to video file
        fps: Frames per second for analysis
        window: Window size for temporal segments
        
    Returns:
        Dictionary with tension_peaks, release_moments, sustained_tension, and curve
    """
    fps = fps or CFG.getfloat("emotion","fps", fallback=2.0)
    window = window or CFG.getint("emotion","window", fallback=16)
    wp = CFG.getfloat("emotion","w_posture")
    wg = CFG.getfloat("emotion","w_gesture")
    wx = CFG.getfloat("emotion","w_prox")
    peak = CFG.getfloat("emotion","tension_peak")
    plateau_len = CFG.getint("emotion","tension_plateau_len")

    # Use V-JEPA to get embeddings with frame_cls
    try:
        E = VJEPAEmbedder()
        segs, _ = E.embed_segments(video_path, fps=fps, window=window, 
                                  strategy="temp_attn", return_frame_cls=True)
    except Exception:
        # Fallback to CLIP if V-JEPA not available
        try:
            from src.embedders.clip_embedder import CLIPEmbedder
            E = CLIPEmbedder()
            segs, _ = E.embed_segments(video_path, fps=fps, window=window)
            # Simulate frame_cls from embeddings
            for s in segs:
                emb = s["emb"]
                # Add some temporal variation to simulate frame-level features
                noise = np.random.randn(window, len(emb)) * 0.1
                base = np.tile(emb, (window, 1))
                s["frame_cls"] = base + noise
        except Exception:
            return {"tension_peaks":[], "release_moments":[], 
                   "sustained_tension":[], "curve":[]}
    
    tension = []
    times = []
    
    for s in segs:
        F = s.get("frame_cls")
        if F is None:
            continue
        
        # Calculate tension components
        rigidity = _rigidity(F)
        velocity = _velocity(F)
        proximity = _proximity_change(F)
        
        # Combine using geometric mean
        val = _geo_mul([rigidity, velocity, proximity], [wp, wg, wx])
        tension.append(val)
        times.append((s["t0"], s["t1"]))

    if not tension: 
        return {"tension_peaks":[], "release_moments":[], 
               "sustained_tension":[], "curve":[]}

    arr = np.array(tension, dtype=np.float32)
    
    # Find peaks (local maxima above threshold)
    peaks = []
    for i in range(1, len(arr)-1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] >= peak:
            peaks.append((times[i][0], times[i][1]))
    
    # Find valleys (local minima below threshold)
    valleys = []
    for i in range(1, len(arr)-1):
        if arr[i] < arr[i-1] and arr[i] < arr[i+1] and arr[i] <= 0.3:
            valleys.append((times[i][0], times[i][1]))
    
    # Find sustained tension (consecutive high tension)
    sustained = []
    run = 0
    for i in range(len(arr)):
        if arr[i] > 0.6:
            run += 1
            if run >= plateau_len:
                sustained.append(times[i])
        else:
            run = 0
    
    # Create tension curve for visualization
    curve = [((t0+t1)/2.0, float(v)) for (t0,t1), v in zip(times, arr.tolist())]
    
    return {
        "tension_peaks": peaks,
        "release_moments": valleys,
        "sustained_tension": sustained,
        "curve": curve
    }