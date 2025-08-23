"""
Real B-roll quality scoring implementation
Replaces placeholder with actual video quality metrics
"""

import numpy as np
import cv2
import torch
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoQualityAnalyzer:
    """
    Analyzes video quality for B-roll selection.
    Implements real metrics instead of placeholders.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
    def calculate_sharpness(self, frame: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        Higher values indicate sharper images.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (empirically determined)
        normalized = min(variance / 1000.0, 1.0)
        return float(normalized)
    
    def calculate_motion_stability(self, frames: List[np.ndarray]) -> float:
        """
        Calculate motion stability between consecutive frames.
        Lower optical flow variance indicates more stable footage.
        """
        if len(frames) < 2:
            return 0.5
            
        flows = []
        for i in range(len(frames) - 1):
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) if len(frames[i].shape) == 3 else frames[i]
            next_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY) if len(frames[i+1].shape) == 3 else frames[i+1]
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flows.append(magnitude.mean())
        
        # Lower variance is better (more stable)
        flow_variance = np.var(flows)
        stability = 1.0 / (1.0 + flow_variance)  # Inverse relationship
        return float(stability)
    
    def calculate_composition_score(self, frame: np.ndarray) -> float:
        """
        Calculate composition quality using rule of thirds and balance.
        """
        h, w = frame.shape[:2]
        
        # Rule of thirds analysis
        thirds_h = [h // 3, 2 * h // 3]
        thirds_w = [w // 3, 2 * w // 3]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, 50, 150)
        
        # Check edge density near thirds lines
        thirds_score = 0.0
        for y in thirds_h:
            region = edges[max(0, y-5):min(h, y+5), :]
            thirds_score += np.mean(region > 0)
        for x in thirds_w:
            region = edges[:, max(0, x-5):min(w, x+5)]
            thirds_score += np.mean(region > 0)
        
        thirds_score /= 4.0  # Normalize
        
        # Visual balance (compare left/right halves)
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        left_weight = np.mean(left_half)
        right_weight = np.mean(right_half)
        balance = 1.0 - abs(left_weight - right_weight) / 255.0
        
        # Combine scores
        composition = 0.6 * thirds_score + 0.4 * balance
        return float(composition)
    
    def calculate_color_consistency(self, frames: List[np.ndarray]) -> float:
        """
        Calculate color consistency across frames.
        """
        if len(frames) < 2:
            return 1.0
            
        histograms = []
        for frame in frames:
            # Calculate histogram for each channel
            hist_r = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([frame], [2], None, [256], [0, 256])
            
            # Normalize and concatenate
            hist = np.concatenate([
                hist_r.flatten() / hist_r.sum(),
                hist_g.flatten() / hist_g.sum(),
                hist_b.flatten() / hist_b.sum()
            ])
            histograms.append(hist)
        
        # Calculate consistency as inverse of histogram variance
        hist_array = np.array(histograms)
        variance = np.mean(np.var(hist_array, axis=0))
        consistency = 1.0 / (1.0 + variance * 100)  # Scale factor for normalization
        
        return float(consistency)
    
    def calculate_exposure_quality(self, frame: np.ndarray) -> float:
        """
        Calculate exposure quality (not over/under exposed).
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Check for over/under exposure
        overexposed = np.mean(l_channel > 240) 
        underexposed = np.mean(l_channel < 15)
        
        # Ideal exposure has minimal over/under exposed pixels
        exposure_score = 1.0 - (overexposed + underexposed)
        
        # Also check histogram distribution
        hist, _ = np.histogram(l_channel, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        
        # Entropy of histogram (higher is better - more distributed)
        entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))
        entropy_normalized = entropy / np.log(256)  # Normalize to 0-1
        
        # Combine scores
        quality = 0.7 * exposure_score + 0.3 * entropy_normalized
        return float(quality)
    
    def analyze_segment(self, segment: Dict[str, Any], video_path: str = None) -> Dict[str, float]:
        """
        Analyze a video segment and return quality metrics.
        
        Args:
            segment: Segment dictionary with t0, t1
            video_path: Optional path to video file
            
        Returns:
            Dictionary of quality metrics
        """
        # If we have cached frame data in segment, use it
        if "frames" in segment:
            frames = segment["frames"]
        elif video_path:
            # Extract frames from video
            frames = self.extract_frames(video_path, segment["t0"], segment["t1"])
        else:
            # Return default scores if no frame data
            return {
                "sharpness": 0.5,
                "stability": 0.5,
                "composition": 0.5,
                "color_consistency": 0.5,
                "exposure": 0.5,
                "overall": 0.5
            }
        
        if not frames or len(frames) == 0:
            return {
                "sharpness": 0.5,
                "stability": 0.5,
                "composition": 0.5,
                "color_consistency": 0.5,
                "exposure": 0.5,
                "overall": 0.5
            }
        
        # Calculate individual metrics
        metrics = {
            "sharpness": np.mean([self.calculate_sharpness(f) for f in frames]),
            "stability": self.calculate_motion_stability(frames),
            "composition": np.mean([self.calculate_composition_score(f) for f in frames]),
            "color_consistency": self.calculate_color_consistency(frames),
            "exposure": np.mean([self.calculate_exposure_quality(f) for f in frames])
        }
        
        # Calculate weighted overall score
        weights = {
            "sharpness": 0.25,
            "stability": 0.20,
            "composition": 0.20,
            "color_consistency": 0.15,
            "exposure": 0.20
        }
        
        metrics["overall"] = sum(metrics[k] * weights[k] for k in weights)
        
        return metrics
    
    def extract_frames(self, video_path: str, t0: float, t1: float, max_frames: int = 10) -> List[np.ndarray]:
        """
        Extract frames from video segment.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        frame_indices = np.linspace(t0 * fps, t1 * fps, min(max_frames, int((t1 - t0) * fps)))
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize for consistent processing
                frame = cv2.resize(frame, (640, 360))
                frames.append(frame)
        
        cap.release()
        return frames


def calculate_broll_quality_score(segment: Dict[str, Any], video_path: Optional[str] = None) -> float:
    """
    Main function to calculate B-roll quality score.
    Replaces the placeholder in broll_scoring.py.
    
    Args:
        segment: Segment dictionary
        video_path: Optional path to video
        
    Returns:
        Quality score between 0 and 1
    """
    analyzer = VideoQualityAnalyzer()
    metrics = analyzer.analyze_segment(segment, video_path)
    return metrics["overall"]


def test_quality_scoring():
    """Test the quality scoring with a sample video"""
    import glob
    
    # Find a test video
    videos = glob.glob("assets/pilots/*.mp4")
    if not videos:
        logger.error("No test videos found")
        return
    
    test_video = videos[0]
    logger.info(f"Testing with video: {test_video}")
    
    # Create test segment
    segment = {
        "t0": 10.0,
        "t1": 20.0
    }
    
    # Calculate quality
    analyzer = VideoQualityAnalyzer()
    metrics = analyzer.analyze_segment(segment, test_video)
    
    logger.info("Quality Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.3f}")
    
    # Test the main function
    score = calculate_broll_quality_score(segment, test_video)
    logger.info(f"\nOverall B-roll Quality Score: {score:.3f}")
    
    # Validate score is in valid range
    assert 0.0 <= score <= 1.0, f"Score {score} out of range"
    logger.info("✅ Quality scoring test passed")
    
    return metrics


if __name__ == "__main__":
    test_quality_scoring()