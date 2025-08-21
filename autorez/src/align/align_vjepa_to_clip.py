import numpy as np
from typing import Tuple

def fit_linear_head(v_embs, t_embs, lam=1e-2):
    V = np.asarray(v_embs, dtype=np.float32)  # [N,Dv]
    T = np.asarray(t_embs, dtype=np.float32)  # [N,Dt]
    Dv = V.shape[1]
    A = V.T @ V + lam * np.eye(Dv, dtype=np.float32)
    B = V.T @ T
    W = np.linalg.solve(A, B)                 # [Dv,Dt]
    return W

def fit_linear_head_kfold(v_embs, t_embs, k=5, lam=1e-2) -> Tuple[np.ndarray, float]:
    V = np.asarray(v_embs, dtype=np.float32); T = np.asarray(t_embs, dtype=np.float32)
    N = len(V); idx = np.arange(N)
    best_W, best_score = None, -1
    for fold in range(k):
        val_mask = (idx % k) == fold
        tr, va = ~val_mask, val_mask
        if tr.sum() == 0 or va.sum() == 0: continue
        W = fit_linear_head(V[tr], T[tr], lam=lam)
        P = project(V[va], W)
        # cosine to targets as quick proxy
        cos = (P * T[va]).sum(axis=1)
        score = float(np.mean(cos))
        if score > best_score: best_W, best_score = W, score
    return best_W if best_W is not None else fit_linear_head(V, T, lam=lam), float(best_score)

def project(V, W):
    P = V @ W
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    return P

def load_projection(path: str) -> np.ndarray:
    """Load projection matrix from file"""
    import os
    if os.path.exists(path):
        data = np.load(path)
        if 'W' in data:
            return data['W']
        elif 'projection' in data:
            return data['projection']
    return None