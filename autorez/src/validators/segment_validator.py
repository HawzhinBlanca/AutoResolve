"""
Video segment validation for boundary and constraint checking.
Ensures segments are within valid ranges for video duration.
"""

import av
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SegmentValidator:
    """Validates segment parameters against video constraints."""
    
    @staticmethod
    def validate_max_segments(video_path: str, fps: float, max_segments_config: int = 500) -> int:
        """
        Validate and adjust max_segments based on actual video duration.
        
        Args:
            video_path: Path to video file
            fps: Frames per second for segment extraction
            max_segments_config: Configured maximum segments
            
        Returns:
            int: Adjusted max_segments that fits video duration
        """
        try:
            with av.open(video_path) as container:
                # Get video stream
                video_stream = next((s for s in container.streams if s.type == 'video'), None)
                if not video_stream:
                    logger.warning(f"No video stream found in {video_path}")
                    return max_segments_config
                
                # Calculate duration in seconds
                if container.duration:
                    duration = container.duration / av.time_base  # Convert to seconds
                elif video_stream.duration and video_stream.time_base:
                    duration = float(video_stream.duration * video_stream.time_base)
                else:
                    # Fallback: count frames
                    frame_count = video_stream.frames
                    if frame_count and video_stream.average_rate:
                        duration = frame_count / float(video_stream.average_rate)
                    else:
                        logger.warning("Cannot determine video duration, using config max")
                        return max_segments_config
                
                # Calculate theoretical maximum segments
                theoretical_max = int(duration * fps)
                
                # Return the minimum of config and theoretical max
                actual_max = min(max_segments_config, theoretical_max)
                
                if actual_max < theoretical_max:
                    logger.info(f"Adjusted max_segments from {max_segments_config} to {actual_max} "
                              f"(video duration: {duration:.1f}s at {fps} fps)")
                
                return max(1, actual_max)  # Ensure at least 1 segment
                
        except Exception as e:
            logger.error(f"Error validating segments for {video_path}: {e}")
            return max_segments_config
    
    @staticmethod
    def get_video_duration(video_path: str) -> Optional[float]:
        """
        Get video duration in seconds.
        
        Args:
            video_path: Path to video file
            
        Returns:
            float: Duration in seconds, or None if cannot determine
        """
        try:
            with av.open(video_path) as container:
                if container.duration:
                    return container.duration / av.time_base
                
                # Fallback to stream duration
                video_stream = next((s for s in container.streams if s.type == 'video'), None)
                if video_stream and video_stream.duration and video_stream.time_base:
                    return float(video_stream.duration * video_stream.time_base)
                
        except Exception as e:
            logger.error(f"Error getting duration for {video_path}: {e}")
        
        return None