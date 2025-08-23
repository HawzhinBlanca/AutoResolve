"""
Test suite for input validators.
Tests segment and duration validation logic.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.validators.duration_validator import DurationValidator


class TestSegmentValidator:
    """Test segment validation logic."""
    
    def test_validate_max_segments_normal_case(self):
        """Test normal case with reasonable video duration."""
        # For a 60-second video at 1 fps, theoretical max is 60 segments
        # If config says 500, should return 60
        # Note: This would need a mock video file for actual testing
    
    def test_validate_max_segments_exceeds_theoretical(self):
        """Test when config max_segments exceeds theoretical maximum."""
        # Would need mock implementation


class TestDurationValidator:
    """Test duration validation logic."""
    
    def test_validate_segment_bounds_normal_video(self):
        """Test segment bounds for normal duration video."""
        video_duration = 60.0  # 1 minute
        min_seg = 3.0
        max_seg = 18.0
        
        adjusted_min, adjusted_max = DurationValidator.validate_segment_bounds(
            video_duration, min_seg, max_seg
        )
        
        # Max should be adjusted to min of original or 50% of video
        assert adjusted_min == 3.0
        assert adjusted_max == min(18.0, 30.0)  # min of 18.0 and 50% of 60s
    
    def test_validate_segment_bounds_short_video(self):
        """Test segment bounds for very short video."""
        video_duration = 2.0  # 2 seconds
        min_seg = 3.0  # Longer than video!
        max_seg = 18.0
        
        adjusted_min, adjusted_max = DurationValidator.validate_segment_bounds(
            video_duration, min_seg, max_seg
        )
        
        # Should adjust to proportional segments (but with MIN_VIABLE_DURATION constraint)
        assert adjusted_min == max(1.0, 2.0 * 0.2)  # max of 1.0 (MIN_VIABLE) and 0.4s
        assert adjusted_max == 2.0 * 0.8  # 1.6s
    
    def test_validate_segment_bounds_too_short_video(self):
        """Test video that's too short for any segmentation."""
        video_duration = 0.5  # Half second
        min_seg = 3.0
        max_seg = 18.0
        
        with pytest.raises(ValueError, match="Video too short"):
            DurationValidator.validate_segment_bounds(
                video_duration, min_seg, max_seg
            )
    
    def test_validate_segment_bounds_conflicting_constraints(self):
        """Test when min_seg > max allowed after adjustment."""
        video_duration = 10.0  # 10 seconds
        min_seg = 8.0  # Would require segments > 50% of video
        max_seg = 18.0
        
        adjusted_min, adjusted_max = DurationValidator.validate_segment_bounds(
            video_duration, min_seg, max_seg
        )
        
        # Should use proportional segments
        assert adjusted_min == 10.0 * 0.1  # 1.0s
        assert adjusted_max == 10.0 * 0.3  # 3.0s
    
    def test_validate_silence_params_normal_video(self):
        """Test silence parameter validation for normal video."""
        video_duration = 60.0
        min_silence_s = 0.35
        min_keep_s = 0.40
        pad_s = 0.05
        
        adj_silence, adj_keep, adj_pad = DurationValidator.validate_silence_params(
            video_duration, min_silence_s, min_keep_s, pad_s
        )
        
        # All should remain unchanged for reasonable video
        assert adj_silence == 0.35
        assert adj_keep == 0.40
        assert adj_pad == 0.05
    
    def test_validate_silence_params_short_video(self):
        """Test silence parameter validation for short video."""
        video_duration = 5.0  # 5 seconds
        min_silence_s = 2.0  # 40% of video!
        min_keep_s = 3.0  # 60% of video!
        pad_s = 0.5  # 10% of video!
        
        adj_silence, adj_keep, adj_pad = DurationValidator.validate_silence_params(
            video_duration, min_silence_s, min_keep_s, pad_s
        )
        
        # Should be adjusted to percentages
        assert adj_silence <= 5.0 * 0.1  # Max 10% (0.5s)
        assert adj_keep <= 5.0 * 0.3  # Max 30% (1.5s)
        assert adj_pad <= 5.0 * 0.01  # Max 1% (0.05s)
    
    def test_validate_broll_timing_normal_video(self):
        """Test B-roll timing validation for normal video."""
        video_duration = 60.0
        max_overlay_s = 7.0
        min_gap_s = 4.0
        dissolve_s = 0.25
        
        adj_overlay, adj_gap, adj_dissolve = DurationValidator.validate_broll_timing(
            video_duration, max_overlay_s, min_gap_s, dissolve_s
        )
        
        # All should remain unchanged for reasonable video
        assert adj_overlay == 7.0
        assert adj_gap == 4.0
        assert adj_dissolve == 0.25
    
    def test_validate_broll_timing_short_video(self):
        """Test B-roll timing validation for short video."""
        video_duration = 10.0
        max_overlay_s = 7.0  # 70% of video!
        min_gap_s = 4.0
        dissolve_s = 1.0  # Long dissolve
        
        adj_overlay, adj_gap, adj_dissolve = DurationValidator.validate_broll_timing(
            video_duration, max_overlay_s, min_gap_s, dissolve_s
        )
        
        # Overlay should be limited to 20% of video
        assert adj_overlay == 10.0 * 0.2  # 2.0s
        
        # Gap should allow at least 3 B-rolls
        assert adj_gap <= 10.0 / 4  # 2.5s max
        
        # Dissolve should be subtle
        assert adj_dissolve <= 0.5  # Max 0.5s
    
    def test_validate_broll_timing_edge_cases(self):
        """Test B-roll timing edge cases."""
        video_duration = 100.0
        max_overlay_s = 50.0  # Very long overlay
        min_gap_s = 60.0  # Very long gap
        dissolve_s = 5.0  # Very long dissolve
        
        adj_overlay, adj_gap, adj_dissolve = DurationValidator.validate_broll_timing(
            video_duration, max_overlay_s, min_gap_s, dissolve_s
        )
        
        # Check all constraints are applied
        assert adj_overlay <= 100.0 * 0.2  # 20s max
        assert adj_gap <= 100.0 / 4  # 25s max
        assert adj_dissolve <= 0.5  # 0.5s max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])