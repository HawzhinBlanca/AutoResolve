# Blueprint3 Director Module - Continuity
# Shot boundaries and transitions detection

import numpy as np
import configparser
import os
from dataclasses import dataclass
from typing import List, Tuple
from src.embedders.vjepa_embedder import VJEPAEmbedder

CFG = configparser.ConfigParser()
CFG.read(os.getenv("DIRECTOR_INI","conf/director.ini"))

@dataclass
class ShotList:
    """Container for detected shots"""
    boundaries: List[float]
    shots: List[Tuple[float, float]]
    transitions: List[str]  # cut, dissolve, etc

def _similarity(emb1, emb2):
    """Calculate cosine similarity between embeddings"""
    n1 = np.linalg.norm(emb1)
    n2 = np.linalg.norm(emb2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (n1 * n2))

def _detect_cuts(similarities, threshold=0.5):
    """Detect hard cuts from similarity drops"""
    cuts = []
    for i in range(1, len(similarities)):
        # Detect significant drop in similarity
        if similarities[i-1] > threshold and similarities[i] < threshold:
            cuts.append(i)
        # Also detect when similarity suddenly drops by large amount
        elif i > 0 and (similarities[i-1] - similarities[i]) > 0.3:
            cuts.append(i)
    return cuts

def detect_shot_boundaries(video_path: str, fps=None, window=None):
    """
    Detect shot boundaries and transitions in video
    
    Args:
        video_path: Path to video file
        fps: Frames per second for analysis
        window: Window size for temporal segments
        
    Returns:
        Dictionary with cuts, boundaries, and similarity_curve
    """
    fps = fps or CFG.getfloat("global","fps", fallback=2.0)
    window = window or CFG.getint("global","window", fallback=16)
    iou_threshold = CFG.getfloat("global","iou_threshold", fallback=0.5)
    
    # Use V-JEPA to get embeddings
    try:
        E = VJEPAEmbedder()
        segs, _ = E.embed_segments(video_path, fps=fps, window=window,
                                  strategy="temp_attn", return_frame_cls=False)
    except Exception:
        # Fallback to CLIP
        try:
            from src.embedders.clip_embedder import CLIPEmbedder
            E = CLIPEmbedder()
            segs, _ = E.embed_segments(video_path, fps=fps, window=window)
        except Exception:
            return {"cuts":[], "boundaries":[], "similarity_curve":[]}
    
    if len(segs) < 2:
        return {"cuts":[], "boundaries":[], "similarity_curve":[]}
    
    # Calculate similarities between consecutive segments
    similarities = []
    times = []
    
    for i in range(len(segs)-1):
        emb1 = np.array(segs[i]["emb"], dtype=np.float32)
        emb2 = np.array(segs[i+1]["emb"], dtype=np.float32)
        
        sim = _similarity(emb1, emb2)
        similarities.append(sim)
        
        # Time is between the two segments
        t = (segs[i]["t1"] + segs[i+1]["t0"]) / 2.0
        times.append(t)
    
    # Detect cuts (hard transitions)
    cut_indices = _detect_cuts(similarities, threshold=iou_threshold)
    cuts = []
    for idx in cut_indices:
        if idx < len(times):
            # Create a time interval around the cut
            t = times[idx]
            cuts.append((t-0.5, t+0.5))
    
    # Detect boundaries (all transitions including gradual)
    boundaries = []
    mean_sim = np.mean(similarities) if similarities else 0.5
    std_sim = np.std(similarities) if len(similarities) > 1 else 0.1
    
    for i, sim in enumerate(similarities):
        # Boundary if similarity is significantly below mean
        if sim < mean_sim - std_sim:
            t = times[i]
            boundaries.append((t-0.5, t+0.5))
    
    # Remove duplicate boundaries that overlap with cuts
    unique_boundaries = []
    for b in boundaries:
        is_duplicate = False
        for c in cuts:
            if abs(b[0] - c[0]) < 0.1:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_boundaries.append(b)
    
    # Create similarity curve for visualization
    similarity_curve = [(t, float(s)) for t, s in zip(times, similarities)]
    
    # Add start and end points for completeness
    if segs:
        similarity_curve.insert(0, (segs[0]["t0"], 1.0))
        similarity_curve.append((segs[-1]["t1"], similarities[-1] if similarities else 1.0))
    
    return {
        "cuts": cuts,
        "boundaries": unique_boundaries,
        "similarity_curve": similarity_curve
    }