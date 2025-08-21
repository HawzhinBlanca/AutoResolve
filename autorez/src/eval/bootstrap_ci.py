import numpy as np

def bootstrap_ci(samples, iters=1000, alpha=0.05, seed=1234):
    np.random.seed(seed)  # Ensure deterministic CI bounds
    samples = np.array(samples, dtype=np.float32)
    n = len(samples); stats=[]
    for _ in range(iters):
        idx = np.random.randint(0, n, n)
        stats.append(np.mean(samples[idx]))
    stats.sort()
    # Ensure stable CI bounds with proper indexing
    lower_idx = max(0, int((alpha/2)*iters))
    upper_idx = min(len(stats)-1, int((1-alpha/2)*iters))
    return float(stats[lower_idx]), float(np.mean(stats)), float(stats[upper_idx])