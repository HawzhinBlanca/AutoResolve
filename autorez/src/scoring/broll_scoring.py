import numpy as np, json, os
from src.utils.common import cosine

def score_candidate(emb_vjepa_proj, emb_clip, qtext_clip, mask_iou, beat_sync,
                    wv=.45, wc=.25, wm=.15, wb=.15, agree_thresh=.25):
    cos_v = cosine(emb_vjepa_proj, qtext_clip)
    cos_c = cosine(emb_clip,       qtext_clip)
    if cos_v < agree_thresh or cos_c < agree_thresh:
        wv = min(wv, .15)
    score = wv*cos_v + wc*cos_c + wm*mask_iou + wb*beat_sync
    return score, {"cos_v":cos_v,"cos_c":cos_c,"wm":wm,"wb":wb}

def log_decision(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f: f.write(json.dumps(record)+"\n")

def mmr(candidates, selected, lambda_diversity=0.5):
    """
    Maximal Marginal Relevance for diverse selection
    """
    if not candidates or not selected:
        return candidates
    
    scores = []
    for cand in candidates:
        relevance = cand.get("score", 0.0)
        
        # Compute max similarity to already selected items
        max_sim = 0.0
        for sel in selected:
            sim = cosine(cand["emb"], sel["emb"])
            max_sim = max(max_sim, sim)
        
        # MMR score
        mmr_score = lambda_diversity * relevance - (1 - lambda_diversity) * max_sim
        scores.append((mmr_score, cand))
    
    # Sort by MMR score
    scores.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scores]

def retrieval_score(segment, query_emb, weights=None):
    """
    Calculate retrieval score for a segment
    """
    if weights is None:
        weights = {"similarity": 0.7, "temporal": 0.2, "quality": 0.1}
    
    # Cosine similarity
    sim = cosine(segment["emb"], query_emb)
    
    # Temporal position score (prefer middle segments)
    t_mid = (segment["t0"] + segment["t1"]) / 2
    temporal_score = 1.0 - abs(0.5 - t_mid / max(segment.get("duration", 100), 1)) * 0.5
    
    # Quality score (placeholder)
    quality_score = segment.get("quality", 0.5)
    
    score = (weights["similarity"] * sim + 
             weights["temporal"] * temporal_score +
             weights["quality"] * quality_score)
    
    return score

def director_boost(segment, director_signals):
    """
    Apply director signal boosts to segment score
    """
    boost = 1.0
    
    # Check narrative beats
    if "narrative" in director_signals:
        for beat in director_signals["narrative"].get("beats", []):
            if segment["t0"] <= beat["t"] <= segment["t1"]:
                if beat["type"] == "climax":
                    boost *= 1.5
                elif beat["type"] == "incident":
                    boost *= 1.2
    
    # Check emphasis moments
    if "emphasis" in director_signals:
        for emphasis in director_signals["emphasis"].get("moments", []):
            if segment["t0"] <= emphasis["t"] <= segment["t1"]:
                boost *= 1.1 + emphasis.get("strength", 0.1)
    
    # Check rhythm peaks
    if "rhythm" in director_signals:
        for peak in director_signals["rhythm"].get("peaks", []):
            if segment["t0"] <= peak["t"] <= segment["t1"]:
                boost *= 1.15
    
    return boost

def nms(segments, iou_threshold=0.5):
    """
    Non-maximum suppression for overlapping segments
    """
    if not segments:
        return []
    
    # Sort by score
    segments = sorted(segments, key=lambda x: x.get("score", 0), reverse=True)
    
    kept = []
    for seg in segments:
        # Check overlap with already kept segments
        keep = True
        for kept_seg in kept:
            # Calculate IoU
            overlap_start = max(seg["t0"], kept_seg["t0"])
            overlap_end = min(seg["t1"], kept_seg["t1"])
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                seg_duration = seg["t1"] - seg["t0"]
                kept_duration = kept_seg["t1"] - kept_seg["t0"]
                union_duration = seg_duration + kept_duration - overlap_duration
                
                iou = overlap_duration / union_duration if union_duration > 0 else 0
                
                if iou > iou_threshold:
                    keep = False
                    break
        
        if keep:
            kept.append(seg)
    
    return kept

def calculate_top_k(relevance_scores, k=3):
    """Calculate top-k accuracy"""
    sorted_scores = sorted(relevance_scores, reverse=True)
    return sum(sorted_scores[:k]) / len(sorted_scores)

def calculate_mrr(relevance_scores):
    """Calculate Mean Reciprocal Rank"""
    sorted_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)
    for rank, idx in enumerate(sorted_indices, 1):
        if relevance_scores[idx] > 0.5:  # Consider relevant if score > 0.5
            return 1.0 / rank
    return 0.0