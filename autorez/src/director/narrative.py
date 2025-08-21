# Blueprint3 Director Module - Narrative
# Story beats detection: setup, incidents, rising, climax, resolution

import numpy as np
import configparser
import os
from dataclasses import dataclass
from typing import List, Tuple
from src.embedders.vjepa_embedder import VJEPAEmbedder

CFG = configparser.ConfigParser()
CFG.read(os.getenv("DIRECTOR_INI","conf/director.ini"))

@dataclass
class StoryBeats:
    setup: List[Tuple[float,float]]
    incidents: List[Tuple[float,float]]
    rising: List[Tuple[float,float]]
    climax: List[Tuple[float,float]]
    resolution: List[Tuple[float,float]]
    energy_curve: List[Tuple[float,float]]  # (t, energy)
    novelty_curve: List[Tuple[float,float]] # (t, novelty)

def _complexity_entropy(frame_cls: np.ndarray) -> float:
    """Calculate complexity entropy from frame class tokens"""
    X = frame_cls - frame_cls.mean(0, keepdims=True)
    C = (X.T @ X) / max(1, X.shape[0]-1)
    vals = np.clip(np.linalg.eigvalsh(C), 1e-9, None)
    p = vals / vals.sum()
    H = -(p * np.log(p)).sum()
    return float(H / np.log(len(vals)))

def _grad(x):
    """Compute gradient with edge case handling"""
    return np.gradient(x) if len(x) > 1 else np.array([0.0])

def _cos(a,b):
    """Cosine similarity with zero handling"""
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

def _novelty_vs_context(emb, ctx):
    """Compute novelty score against context"""
    return 1.0 - (1.0 + _cos(emb, ctx)) / 2.0

def _find_climax_better(energy, momentum, pct=75):
    """Find climax points using energy and momentum"""
    hi = np.percentile(energy, pct)
    candidates = []
    for i in range(len(energy)-3):
        if energy[i] >= hi and np.all(momentum[i+1:i+4] < momentum[i]):
            candidates.append(i)
    return candidates if candidates else [int(np.argmax(energy))]

def detect_story_beats(video_path: str, fps=None, window=None):
    """
    Detect narrative story beats in video
    
    Args:
        video_path: Path to video file
        fps: Frames per second for analysis
        window: Window size for temporal segments
        
    Returns:
        Beats dataclass with detected story elements
    """
    fps = fps or CFG.getfloat("global","fps", fallback=2.0)
    window = window or CFG.getint("global","window", fallback=16)
    stable_m = CFG.getfloat("narrative","stable_momentum", fallback=0.10)
    spike_m  = CFG.getfloat("narrative","spike_momentum",  fallback=0.80)
    high_pct = CFG.getint("narrative","high_energy_pct", fallback=75)

    # Use V-JEPA to get embeddings with frame_cls
    try:
        E = VJEPAEmbedder()
        segs, _ = E.embed_segments(video_path, fps=fps, window=window, 
                                  strategy="temp_attn", return_frame_cls=True)
    except Exception:
        # Fallback for when V-JEPA model is not available
        # Use CLIP embedder as backup
        try:
            from src.embedders.clip_embedder import CLIPEmbedder
            E = CLIPEmbedder()
            segs, _ = E.embed_segments(video_path, fps=fps, window=window)
            # CLIP doesn't provide frame_cls, so we'll simulate with embeddings
            for s in segs:
                # Create fake frame_cls from embedding
                emb = s["emb"]
                s["frame_cls"] = np.tile(emb, (window, 1))
        except Exception:
            return StoryBeats([],[],[],[],[],[],[])

    energy=[]; times=[]; nov=[]; ctx=None
    for s in segs:
        F = s.get("frame_cls")
        if F is None or len(F)<2: continue
        e = _complexity_entropy(np.array(F, dtype=np.float32))
        energy.append(e); times.append((s["t0"], s["t1"]))
        emb = np.array(s["emb"], dtype=np.float32)
        if ctx is None: ctx = emb.copy()
        novelty = _novelty_vs_context(emb, ctx)
        ctx = 0.9*ctx + 0.1*emb
        nov.append(float(novelty))

    if not energy: 
        return StoryBeats([],[],[],[],[],[],[])

    energy = np.array(energy, dtype=np.float32)
    novelty = np.array(nov, dtype=np.float32)
    momentum = _grad(energy)

    # Detect different beat types based on momentum
    setup      = [(t0,t1) for (t0,t1),m in zip(times, momentum) if abs(m) < stable_m]
    incidents  = [(t0,t1) for (t0,t1),m in zip(times, momentum) if m > spike_m]
    rising     = [(t0,t1) for (t0,t1),m in zip(times, momentum) if m >= 0.3 and m <= spike_m]
    cand_idx   = _find_climax_better(energy, momentum, pct=high_pct)
    climax     = [times[i] for i in cand_idx[:1]] if cand_idx else []
    
    # Resolution comes after climax
    if cand_idx:
        cidx = cand_idx[0]
        resolution = [(t0,t1) for k,(t0,t1) in enumerate(times) 
                     if k>cidx and abs(momentum[k]) < stable_m]
    else:
        resolution = []

    # Create energy and novelty curves for visualization
    curve = [((t0+t1)/2.0, float(e)) for (t0,t1),e in zip(times, energy)]
    nov_curve = [((t0+t1)/2.0, float(n)) for (t0,t1),n in zip(times, novelty)]
    
    return StoryBeats(setup, incidents, rising, climax, resolution, curve, nov_curve)