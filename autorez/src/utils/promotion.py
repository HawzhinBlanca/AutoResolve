"""
Promotion logic for V-JEPA vs CLIP model selection.
Handles numerical stability and correct CI calculations.
"""

def promote_vjepa(results, sec_per_min):
    """
    Safely evaluate V-JEPA promotion with numerical stability.
    
    Args:
        results: Dict containing top3 and mrr metrics with confidence intervals
        sec_per_min: Performance metric (seconds per minute of video)
    
    Returns:
        bool: True if V-JEPA should be promoted, False otherwise
    """
    EPSILON = 0.001  # Prevent near-zero division with meaningful threshold
    
    # Validate CI structure (expect [lower, upper] for 95% CI)
    if not all(key in results for key in ["top3", "mrr"]):
        raise ValueError("Results must contain 'top3' and 'mrr' keys")
    
    for metric in ["top3", "mrr"]:
        for model in ["vjepa", "clip"]:
            ci_key = f"{model}_ci"
            if ci_key not in results[metric]:
                raise ValueError(f"Missing CI data for {metric}.{ci_key}")
            if len(results[metric][ci_key]) != 2:
                raise ValueError(f"CI must be [lower, upper], got {results[metric][ci_key]}")
    
    # Safe division with meaningful epsilon to avoid numerical instability
    clip_top3 = max(EPSILON, results["top3"]["clip"])
    clip_mrr = max(EPSILON, results["mrr"]["clip"])
    
    # Calculate relative gains
    top3_gain = (results["top3"]["vjepa"] / clip_top3) - 1.0
    mrr_gain = (results["mrr"]["vjepa"] / clip_mrr) - 1.0
    
    # Correct CI calculation: vjepa_lower - clip_upper for conservative estimate
    ci_lower_t3 = results["top3"]["vjepa_ci"][0] - results["top3"]["clip_ci"][1]
    ci_lower_mr = results["mrr"]["vjepa_ci"][0] - results["mrr"]["clip_ci"][1]
    
    # All conditions must be met for promotion
    return (top3_gain >= 0.15 and 
            mrr_gain >= 0.15 and
            ci_lower_t3 > 0 and 
            ci_lower_mr > 0 and 
            sec_per_min <= 5.0)