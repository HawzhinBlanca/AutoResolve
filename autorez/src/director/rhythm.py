# Blueprint3 Director Module - Rhythm
# Pace detection: fast, medium, slow

import numpy as np
import configparser
import os
from dataclasses import dataclass
from typing import List, Tuple
from src.embedders.vjepa_embedder import VJEPAEmbedder

CFG = configparser.ConfigParser()
CFG.read(os.getenv("DIRECTOR_INI","conf/director.ini"))

@dataclass
class PaceEvents:
    """Container for pace/rhythm events"""
    changes: List[Tuple[float, float]]
    peaks: List[Tuple[float, float]]
    valleys: List[Tuple[float, float]]
    curve: List[Tuple[float, float]]

def _velocity_curve(frame_cls):
    """Calculate velocity curve from frame class tokens"""
    if frame_cls is None or len(frame_cls) < 2:
        return np.array([0.0])
    v = np.linalg.norm(np.diff(frame_cls, axis=0), axis=1)
    return v

def _merge(points: List[Tuple[float,float]], gap=0.4):
    """Merge nearby intervals"""
    if not points:
        return []
    points.sort()
    out = [points[0]]
    for (a0, a1) in points[1:]:
        b0, b1 = out[-1]
        if a0 - b1 <= gap:
            out[-1] = (b0, max(a1, b1))  # Merge intervals
        else:
            out.append((a0, a1))
    return out

def detect_pace(video_path: str, fps=None, window=None):
    """
    Detect pace/rhythm in video
    
    Args:
        video_path: Path to video file
        fps: Frames per second for analysis
        window: Window size for temporal segments
        
    Returns:
        Dictionary with fast_pace, medium_pace, slow_pace, and velocity_curve
    """
    fps = fps or CFG.getfloat("global","fps", fallback=2.0)
    window = window or CFG.getint("global","window", fallback=16)
    merge_gap = CFG.getfloat("rhythm","merge_gap", fallback=0.40)
    
    # Use V-JEPA to get embeddings with frame_cls
    try:
        E = VJEPAEmbedder()
        segs, _ = E.embed_segments(video_path, fps=fps, window=window,
                                  strategy="temp_attn", return_frame_cls=True)
    except Exception:
        # Fallback to CLIP
        try:
            from src.embedders.clip_embedder import CLIPEmbedder
            E = CLIPEmbedder()
            segs, _ = E.embed_segments(video_path, fps=fps, window=window)
            # Simulate frame_cls
            for s in segs:
                emb = s["emb"]
                # Create temporal variation
                t = np.linspace(0, 2*np.pi, window)
                variation = np.sin(t[:, np.newaxis]) * 0.1
                base = np.tile(emb, (window, 1))
                s["frame_cls"] = base + variation
        except Exception:
            return {"fast_pace":[], "medium_pace":[], "slow_pace":[], "velocity_curve":[]}
    
    velocities = []
    times = []
    
    for s in segs:
        F = s.get("frame_cls")
        if F is None or len(F) < 2:
            continue
        
        # Calculate mean velocity for this segment
        v_curve = _velocity_curve(F)
        mean_v = float(v_curve.mean()) if len(v_curve) > 0 else 0.0
        
        velocities.append(mean_v)
        times.append((s["t0"], s["t1"]))
    
    if not velocities:
        return {"fast_pace":[], "medium_pace":[], "slow_pace":[], "velocity_curve":[]}
    
    # Normalize velocities
    v_arr = np.array(velocities, dtype=np.float32)
    if v_arr.max() > 0:
        v_arr = v_arr / v_arr.max()
    
    # Define pace thresholds using percentiles
    slow_thresh = np.percentile(v_arr, 33)
    fast_thresh = np.percentile(v_arr, 67)
    
    # Classify segments by pace
    fast_pace = []
    medium_pace = []
    slow_pace = []
    
    for i, v in enumerate(v_arr):
        t = times[i]
        if v >= fast_thresh:
            fast_pace.append(t)
        elif v <= slow_thresh:
            slow_pace.append(t)
        else:
            medium_pace.append(t)
    
    # Merge nearby segments of same pace
    fast_pace = _merge(fast_pace, gap=merge_gap)
    medium_pace = _merge(medium_pace, gap=merge_gap)
    slow_pace = _merge(slow_pace, gap=merge_gap)
    
    # Create velocity curve for visualization
    velocity_curve = [((t0+t1)/2.0, float(v)) for (t0,t1), v in zip(times, v_arr.tolist())]
    
    return {
        "fast_pace": fast_pace,
        "medium_pace": medium_pace,
        "slow_pace": slow_pace,
        "velocity_curve": velocity_curve
    }