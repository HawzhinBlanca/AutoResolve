import numpy as np

def cos(a, b):
    """Cosine similarity with zero and dimension mismatch handling"""
    # Handle dimension mismatch
    if hasattr(a, 'shape') and hasattr(b, 'shape'):
        if a.shape != b.shape:
            # Log warning and return 0 for dimension mismatch
            import logging
            logging.debug(f"Cosine similarity dimension mismatch: {a.shape} vs {b.shape}")
            return 0.0
    
    # Add epsilon to prevent division by zero
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    
    # Check for zero vectors
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    
    # Compute similarity and clip to valid range
    similarity = np.dot(a, b) / (na * nb)
    return float(np.clip(similarity, -1.0, 1.0))

def iou(a, b):
    """Intersection over union for time ranges"""
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    if end <= start:
        return 0.0
    intersection = end - start
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union > 0 else 0.0

def set_global_seed(seed=1234):
    """Set all random seeds for determinism"""
    import random
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
