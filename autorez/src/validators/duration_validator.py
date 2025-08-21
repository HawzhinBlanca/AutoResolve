"""
Duration validation for video segments and clips.
Ensures segment boundaries are appropriate for video length.
"""

from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DurationValidator:
    """Validates duration-based constraints for video processing."""
    
    @staticmethod
    def validate_segment_bounds(
        video_duration: float, 
        min_seg: float, 
        max_seg: float
    ) -> Tuple[float, float]:
        """
        Ensure segment bounds are valid for video duration.
        
        Args:
            video_duration: Total video duration in seconds
            min_seg: Minimum segment duration
            max_seg: Maximum segment duration
            
        Returns:
            Tuple[float, float]: Adjusted (min_seg, max_seg) values
            
        Raises:
            ValueError: If video is too short for any reasonable segmentation
        """
        # Minimum viable video duration
        MIN_VIABLE_DURATION = 1.0
        
        if video_duration < MIN_VIABLE_DURATION:
            raise ValueError(
                f"Video too short ({video_duration:.2f}s) for segmentation. "
                f"Minimum required: {MIN_VIABLE_DURATION}s"
            )
        
        # Check if video is shorter than minimum segment
        if video_duration < min_seg:
            logger.warning(
                f"Video duration ({video_duration:.1f}s) shorter than min_seg ({min_seg}s). "
                f"Adjusting boundaries."
            )
            # For very short videos, use proportional segments
            adjusted_min = max(MIN_VIABLE_DURATION, video_duration * 0.2)
            adjusted_max = video_duration * 0.8
            return adjusted_min, adjusted_max
        
        # Max segment can't exceed 50% of video duration (to allow multiple segments)
        max_allowed = min(max_seg, video_duration * 0.5)
        
        # Ensure min <= max after adjustment
        if min_seg > max_allowed:
            # If constraints conflict, use proportional segments
            adjusted_min = video_duration * 0.1
            adjusted_max = video_duration * 0.3
            logger.info(
                f"Segment bounds adjusted to {adjusted_min:.1f}-{adjusted_max:.1f}s "
                f"for {video_duration:.1f}s video"
            )
        else:
            adjusted_min = min_seg
            adjusted_max = max_allowed
            
            if max_allowed < max_seg:
                logger.info(
                    f"Max segment reduced from {max_seg}s to {adjusted_max:.1f}s "
                    f"for {video_duration:.1f}s video"
                )
        
        return adjusted_min, adjusted_max
    
    @staticmethod
    def validate_silence_params(
        video_duration: float,
        min_silence_s: float,
        min_keep_s: float,
        pad_s: float
    ) -> Tuple[float, float, float]:
        """
        Validate silence detection parameters against video duration.
        
        Args:
            video_duration: Total video duration in seconds
            min_silence_s: Minimum silence duration to cut
            min_keep_s: Minimum segment to keep
            pad_s: Padding around cuts
            
        Returns:
            Tuple[float, float, float]: Adjusted parameters
        """
        # Ensure parameters are reasonable for video length
        max_silence = video_duration * 0.1  # Max 10% of video as single silence
        max_keep = video_duration * 0.3     # Max 30% as single keep segment
        
        adjusted_silence = min(min_silence_s, max_silence)
        adjusted_keep = min(min_keep_s, max_keep)
        adjusted_pad = min(pad_s, video_duration * 0.01)  # Max 1% padding
        
        if adjusted_silence != min_silence_s:
            logger.info(f"Adjusted min_silence from {min_silence_s}s to {adjusted_silence:.2f}s")
        if adjusted_keep != min_keep_s:
            logger.info(f"Adjusted min_keep from {min_keep_s}s to {adjusted_keep:.2f}s")
        if adjusted_pad != pad_s:
            logger.info(f"Adjusted pad from {pad_s}s to {adjusted_pad:.3f}s")
        
        return adjusted_silence, adjusted_keep, adjusted_pad
    
    @staticmethod
    def validate_broll_timing(
        video_duration: float,
        max_overlay_s: float,
        min_gap_s: float,
        dissolve_s: float
    ) -> Tuple[float, float, float]:
        """
        Validate B-roll overlay timing parameters.
        
        Args:
            video_duration: Total video duration
            max_overlay_s: Maximum B-roll overlay duration
            min_gap_s: Minimum gap between overlays
            dissolve_s: Cross-dissolve duration
            
        Returns:
            Tuple[float, float, float]: Adjusted parameters
        """
        # B-roll shouldn't exceed 20% of scene duration
        adjusted_overlay = min(max_overlay_s, video_duration * 0.2)
        
        # Gap should allow at least 3 B-rolls in the video
        max_gap = video_duration / 4
        adjusted_gap = min(min_gap_s, max_gap)
        
        # Dissolve should be subtle (max 5% of overlay)
        max_dissolve = min(dissolve_s, adjusted_overlay * 0.05, 0.5)
        adjusted_dissolve = max_dissolve
        
        return adjusted_overlay, adjusted_gap, adjusted_dissolve