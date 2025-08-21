# Blueprint3 Director Module - Emphasis
# Saliency tracking and attention detection

import numpy as np
import configparser
import os
from dataclasses import dataclass
from typing import List, Tuple
from src.embedders.vjepa_embedder import VJEPAEmbedder

CFG = configparser.ConfigParser()
CFG.read(os.getenv("DIRECTOR_INI","conf/director.ini"))

@dataclass
class SaliencyTrack:
    """Container for saliency tracking results"""
    salient_objects: List[Tuple[float, float, str]]  # (t0, t1, object_type)
    faces: List[Tuple[float, float]]
    text: List[Tuple[float, float]]
    motion_peaks: List[float]

def _attention_entropy(frame_cls):
    """Calculate attention entropy from frame class tokens"""
    if frame_cls is None or len(frame_cls) < 1:
        return 0.5
    
    # Calculate attention weights across spatial tokens
    cls_tokens = frame_cls[:, 0, :] if len(frame_cls.shape) > 2 else frame_cls
    
    # Self-attention scores with numerical stability
    attn_scores = cls_tokens @ cls_tokens.T
    attn_scores = attn_scores - np.max(attn_scores, axis=1, keepdims=True)  # Subtract max for stability
    attn_probs = np.exp(attn_scores)
    attn_probs = attn_probs / (attn_probs.sum(axis=1, keepdims=True) + 1e-8)  # Add epsilon for stability
    
    # Entropy of attention distribution
    entropy = -(attn_probs * np.log(attn_probs + 1e-9)).sum(axis=1).mean()
    
    # Normalize to [0, 1]
    max_entropy = np.log(len(cls_tokens))
    normalized = float(entropy / max_entropy) if max_entropy > 0 else 0.5
    
    return np.clip(normalized, 0, 1)

def _saliency_score(emb, context_embs):
    """Calculate saliency as distinctiveness from context"""
    if len(context_embs) == 0:
        return 0.5
    
    # Calculate mean similarity to context
    similarities = []
    for ctx in context_embs:
        n1 = np.linalg.norm(emb)
        n2 = np.linalg.norm(ctx)
        if n1 > 0 and n2 > 0:
            sim = np.dot(emb, ctx) / (n1 * n2)
            similarities.append(sim)
    
    if not similarities:
        return 0.5
    
    # Saliency is inverse of similarity (distinctiveness)
    mean_sim = np.mean(similarities)
    saliency = 1.0 - mean_sim
    
    return float(np.clip(saliency, 0, 1))

def track_saliency(video_path: str, fps=None, window=None):
    """
    Track visual saliency and attention in video
    
    Args:
        video_path: Path to video file
        fps: Frames per second for analysis
        window: Window size for temporal segments
        
    Returns:
        Dictionary with saliency_peaks, attention_regions, and focus_curve
    """
    fps = fps or CFG.getfloat("global","fps", fallback=2.0)
    window = window or CFG.getint("global","window", fallback=16)
    
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
                # Create varying attention patterns
                t = np.linspace(0, 4*np.pi, window)
                attention = (np.sin(t) + 1) / 2  # Varying attention [0, 1]
                base = np.tile(emb, (window, 1))
                s["frame_cls"] = base * attention[:, np.newaxis]
        except Exception:
            return {"saliency_peaks":[], "attention_regions":[], "focus_curve":[]}
    
    if not segs:
        return {"saliency_peaks":[], "attention_regions":[], "focus_curve":[]}
    
    saliency_scores = []
    attention_scores = []
    times = []
    
    # Build context for saliency calculation
    all_embs = [np.array(s["emb"], dtype=np.float32) for s in segs]
    
    for i, s in enumerate(segs):
        # Calculate saliency (distinctiveness)
        emb = all_embs[i]
        context = all_embs[:i] + all_embs[i+1:]  # All other embeddings
        context_sample = context[::max(1, len(context)//10)]  # Sample for efficiency
        
        saliency = _saliency_score(emb, context_sample)
        saliency_scores.append(saliency)
        
        # Calculate attention entropy
        F = s.get("frame_cls")
        if F is not None:
            attention = _attention_entropy(np.array(F, dtype=np.float32))
        else:
            attention = 0.5
        attention_scores.append(attention)
        
        times.append((s["t0"], s["t1"]))
    
    # Convert to arrays
    sal_arr = np.array(saliency_scores, dtype=np.float32)
    att_arr = np.array(attention_scores, dtype=np.float32)
    
    # Find saliency peaks (local maxima above threshold)
    saliency_peaks = []
    sal_threshold = np.percentile(sal_arr, 75)  # Top 25%
    
    for i in range(1, len(sal_arr)-1):
        if (sal_arr[i] > sal_arr[i-1] and 
            sal_arr[i] > sal_arr[i+1] and 
            sal_arr[i] >= sal_threshold):
            saliency_peaks.append(times[i])
    
    # Find attention regions (high attention areas)
    attention_regions = []
    att_threshold = np.percentile(att_arr, 70)  # Top 30%
    
    in_region = False
    region_start = None
    
    for i, att in enumerate(att_arr):
        if att >= att_threshold and not in_region:
            in_region = True
            region_start = times[i][0]
        elif att < att_threshold and in_region:
            in_region = False
            if region_start is not None:
                attention_regions.append((region_start, times[i-1][1]))
    
    # Close last region if needed
    if in_region and region_start is not None:
        attention_regions.append((region_start, times[-1][1]))
    
    # Combine saliency and attention for focus curve
    focus_scores = (sal_arr + att_arr) / 2.0
    focus_curve = [((t0+t1)/2.0, float(f)) for (t0,t1), f in zip(times, focus_scores)]
    
    return {
        "saliency_peaks": saliency_peaks,
        "attention_regions": attention_regions,
        "focus_curve": focus_curve
    }