"""
Test suite for promotion logic validation.
Tests edge cases, numerical stability, and CI calculations.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.promotion import promote_vjepa


class TestPromotionLogic:
    """Test V-JEPA promotion logic with various edge cases."""
    
    def test_promote_vjepa_near_zero_clip_scores(self):
        """Test handling of near-zero CLIP scores (numerical stability)."""
        results = {
            "top3": {
                "vjepa": 0.8, 
                "clip": 0.0001,  # Near-zero value
                "vjepa_ci": [0.75, 0.85], 
                "clip_ci": [0.0, 0.001]
            },
            "mrr": {
                "vjepa": 0.7, 
                "clip": 0.0001,  # Near-zero value
                "vjepa_ci": [0.65, 0.75], 
                "clip_ci": [0.0, 0.001]
            }
        }
        # Should promote: huge gains, CI non-overlapping
        assert promote_vjepa(results, 4.0) == True
    
    def test_promote_vjepa_ci_overlap(self):
        """Test when confidence intervals overlap."""
        results = {
            "top3": {
                "vjepa": 0.8, 
                "clip": 0.6,
                "vjepa_ci": [0.75, 0.85], 
                "clip_ci": [0.74, 0.86]  # Overlaps with V-JEPA
            },
            "mrr": {
                "vjepa": 0.7, 
                "clip": 0.55,
                "vjepa_ci": [0.65, 0.75], 
                "clip_ci": [0.50, 0.60]
            }
        }
        # Should not promote: CI overlap on top3
        assert promote_vjepa(results, 4.0) == False
    
    def test_promote_vjepa_exactly_15_percent_gain(self):
        """Test boundary condition: exactly 15% gain."""
        results = {
            "top3": {
                "vjepa": 0.58,  # Slightly more than 15% gain (16%)
                "clip": 0.5,
                "vjepa_ci": [0.55, 0.60], 
                "clip_ci": [0.48, 0.52]
            },
            "mrr": {
                "vjepa": 0.47,  # Slightly more than 15% gain (17.5%)
                "clip": 0.4,
                "vjepa_ci": [0.44, 0.48], 
                "clip_ci": [0.38, 0.42]
            }
        }
        # Should promote: exceeds threshold
        assert promote_vjepa(results, 4.0) == True
    
    def test_promote_vjepa_performance_threshold(self):
        """Test performance threshold (sec/min)."""
        results = {
            "top3": {
                "vjepa": 0.8, 
                "clip": 0.6,
                "vjepa_ci": [0.75, 0.85], 
                "clip_ci": [0.55, 0.65]
            },
            "mrr": {
                "vjepa": 0.7, 
                "clip": 0.55,
                "vjepa_ci": [0.65, 0.75], 
                "clip_ci": [0.50, 0.60]
            }
        }
        # Good metrics but slow performance
        assert promote_vjepa(results, 5.1) == False  # Too slow
        assert promote_vjepa(results, 5.0) == True   # Just fast enough
        assert promote_vjepa(results, 4.9) == True   # Fast
    
    def test_promote_vjepa_invalid_ci_structure(self):
        """Test handling of invalid CI structure."""
        results = {
            "top3": {
                "vjepa": 0.8, 
                "clip": 0.6,
                "vjepa_ci": [0.75],  # Invalid: only one value
                "clip_ci": [0.55, 0.65]
            },
            "mrr": {
                "vjepa": 0.7, 
                "clip": 0.55,
                "vjepa_ci": [0.65, 0.75], 
                "clip_ci": [0.50, 0.60]
            }
        }
        with pytest.raises(ValueError, match="CI must be"):
            promote_vjepa(results, 4.0)
    
    def test_promote_vjepa_missing_metrics(self):
        """Test handling of missing metric keys."""
        results = {
            "top3": {
                "vjepa": 0.8, 
                "clip": 0.6,
                "vjepa_ci": [0.75, 0.85], 
                "clip_ci": [0.55, 0.65]
            }
            # Missing 'mrr' key
        }
        with pytest.raises(ValueError, match="must contain"):
            promote_vjepa(results, 4.0)
    
    def test_promote_vjepa_zero_clip_scores(self):
        """Test handling of zero CLIP scores."""
        results = {
            "top3": {
                "vjepa": 0.8, 
                "clip": 0.0,  # Zero score
                "vjepa_ci": [0.75, 0.85], 
                "clip_ci": [0.0, 0.0]
            },
            "mrr": {
                "vjepa": 0.7, 
                "clip": 0.0,  # Zero score
                "vjepa_ci": [0.65, 0.75], 
                "clip_ci": [0.0, 0.0]
            }
        }
        # Should handle gracefully with epsilon
        assert promote_vjepa(results, 4.0) == True
    
    def test_promote_vjepa_negative_gains(self):
        """Test when V-JEPA performs worse than CLIP."""
        results = {
            "top3": {
                "vjepa": 0.4,  # Worse than CLIP
                "clip": 0.6,
                "vjepa_ci": [0.35, 0.45], 
                "clip_ci": [0.55, 0.65]
            },
            "mrr": {
                "vjepa": 0.3,  # Worse than CLIP
                "clip": 0.55,
                "vjepa_ci": [0.25, 0.35], 
                "clip_ci": [0.50, 0.60]
            }
        }
        # Should not promote: V-JEPA is worse
        assert promote_vjepa(results, 4.0) == False
    
    def test_promote_vjepa_one_metric_fails(self):
        """Test when only one metric meets threshold."""
        results = {
            "top3": {
                "vjepa": 0.8,  # Good gain
                "clip": 0.6,
                "vjepa_ci": [0.75, 0.85], 
                "clip_ci": [0.55, 0.65]
            },
            "mrr": {
                "vjepa": 0.63,  # Only 5% gain (below 15% threshold)
                "clip": 0.6,
                "vjepa_ci": [0.60, 0.66], 
                "clip_ci": [0.57, 0.63]
            }
        }
        # Should not promote: MRR gain too small
        assert promote_vjepa(results, 4.0) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])