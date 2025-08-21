"""
Score normalization for video segment scoring.
Ensures weights are properly normalized and validated.
"""

class ScoreNormalizer:
    """Handles score calculation with normalized weights."""
    
    def __init__(self):
        """Initialize with default weights that sum to 1.0."""
        self.weights = {
            'content': 0.45,
            'narrative': 0.25,
            'tension': 0.15,
            'emphasis': 0.10,
            'continuity': 0.10,  # Added missing weight for proper normalization
            'rhythm_penalty': -0.05  # Negative weight for penalty
        }
        self._validate_weights()
    
    def _validate_weights(self):
        """Ensure weights sum to 1.0 for proper normalization."""
        # Sum absolute values for positive contribution
        positive_sum = sum(w for w in self.weights.values() if w > 0)
        negative_sum = sum(w for w in self.weights.values() if w < 0)
        total = positive_sum + negative_sum
        
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights sum to {total:.3f}, not 1.0. "
                           f"Positive: {positive_sum:.3f}, Negative: {negative_sum:.3f}")
    
    def calculate_score(self, metrics):
        """
        Calculate normalized score from metrics.
        
        Args:
            metrics: Dict containing values for each weight key
            
        Returns:
            float: Normalized score between 0 and 1
        """
        score = 0.0
        for key, weight in self.weights.items():
            if key in metrics:
                # Handle rhythm penalty as subtraction
                if key == 'rhythm_penalty':
                    score += weight * metrics[key]
                else:
                    score += weight * metrics[key]
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, score))
    
    def get_weights_info(self):
        """Return human-readable weights information."""
        info = []
        for key, weight in sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True):
            sign = "+" if weight >= 0 else ""
            info.append(f"{key}: {sign}{weight:.2f}")
        return ", ".join(info)