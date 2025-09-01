#!/usr/bin/env python3
"""
Advanced Motion Tracking and Stabilization System
Implements optical flow, feature tracking, and AI-powered stabilization
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class TrackingMethod(Enum):
    """Motion tracking algorithms"""
    OPTICAL_FLOW = "optical_flow"
    FEATURE_BASED = "feature_based"
    TEMPLATE_MATCHING = "template_matching"
    AI_TRACKER = "ai_tracker"
    DENSE_FLOW = "dense_flow"

class StabilizationMode(Enum):
    """Video stabilization modes"""
    SMOOTH = "smooth"          # Smooth camera movement
    LOCKED = "locked"          # Lock to fixed position
    CINEMATIC = "cinematic"    # Cinema-style stabilization
    TRIPOD = "tripod"          # Simulate tripod shot
    HANDHELD = "handheld"      # Natural handheld look

@dataclass
class TrackingPoint:
    """Single tracking point data"""
    id: int
    position: Tuple[float, float]
    confidence: float
    frame: int
    velocity: Optional[Tuple[float, float]] = None
    acceleration: Optional[Tuple[float, float]] = None
    is_lost: bool = False

@dataclass
class MotionPath:
    """Complete motion path for tracked object"""
    points: List[TrackingPoint]
    object_id: int
    start_frame: int
    end_frame: int
    smoothed_path: Optional[List[Tuple[float, float]]] = None
    
    def get_position_at_frame(self, frame: int) -> Optional[Tuple[float, float]]:
        """Get interpolated position at specific frame"""
        for point in self.points:
            if point.frame == frame:
                return point.position
        
        # Interpolate if between points
        prev, next = None, None
        for point in self.points:
            if point.frame < frame:
                prev = point
            elif point.frame > frame and prev:
                next = point
                break
        
        if prev and next:
            # Linear interpolation
            t = (frame - prev.frame) / (next.frame - prev.frame)
            x = prev.position[0] + t * (next.position[0] - prev.position[0])
            y = prev.position[1] + t * (next.position[1] - prev.position[1])
            return (x, y)
        
        return None

class OpticalFlowTracker:
    """Lucas-Kanade optical flow tracker"""
    
    def __init__(self, max_corners: int = 100):
        self.max_corners = max_corners
        self.feature_params = {
            'maxCorners': max_corners,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7
        }
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
    def initialize(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Initialize tracking points"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if roi:
            x, y, w, h = roi
            mask = np.zeros_like(gray)
            mask[y:y+h, x:x+w] = 255
        else:
            mask = None
        
        # Detect good features to track
        corners = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        return corners if corners is not None else np.array([])
    
    def track(self, prev_frame: np.ndarray, curr_frame: np.ndarray, 
              prev_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track points between frames"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )
        
        # Filter good points
        if next_pts is not None:
            good_new = next_pts[status == 1]
            good_old = prev_pts[status == 1]
            return good_new.reshape(-1, 1, 2), good_old.reshape(-1, 1, 2), error
        
        return np.array([]), np.array([]), np.array([])

class FeatureTracker:
    """Feature-based tracking using ORB/SIFT descriptors"""
    
    def __init__(self, detector_type: str = "ORB"):
        if detector_type == "ORB":
            self.detector = cv2.ORB_create()
        elif detector_type == "SIFT":
            self.detector = cv2.SIFT_create()
        else:
            self.detector = cv2.AKAZE_create()
        
        # Feature matcher
        if detector_type == "ORB":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def detect_and_compute(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect keypoints and compute descriptors"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features between frames"""
        if desc1 is None or desc2 is None:
            return []
        
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:50]  # Keep top 50 matches

class AIMotionTracker(nn.Module):
    """Deep learning-based motion tracker"""
    
    def __init__(self, input_channels: int = 6, hidden_dim: int = 256):
        super().__init__()
        
        # CNN backbone for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Correlation layer for matching
        self.correlation = nn.Conv2d(512, hidden_dim, 1)
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_dim * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)  # dx, dy motion
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Predict motion between frames"""
        # Concatenate frames
        x = torch.cat([frame1, frame2], dim=1)
        
        # Extract features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global pooling
        x = F.adaptive_avg_pool2d(x, (16, 16))
        
        # Flatten and predict
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class VideoStabilizer:
    """Advanced video stabilization system"""
    
    def __init__(self, mode: StabilizationMode = StabilizationMode.SMOOTH):
        self.mode = mode
        self.smoothing_radius = 30
        self.transforms = []
        
    def estimate_transform(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> np.ndarray:
        """Estimate affine transform between point sets"""
        if len(prev_pts) < 3 or len(curr_pts) < 3:
            return np.eye(3)
        
        # Find transformation matrix
        transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        
        if transform is not None:
            # Convert to 3x3 homography
            h = np.eye(3)
            h[:2, :] = transform
            return h
        
        return np.eye(3)
    
    def smooth_trajectory(self, trajectory: List[np.ndarray]) -> List[np.ndarray]:
        """Apply smoothing to camera trajectory"""
        if not trajectory:
            return []
        
        # Convert to cumulative transforms
        cumulative = [trajectory[0]]
        for i in range(1, len(trajectory)):
            cumulative.append(cumulative[-1] @ trajectory[i])
        
        # Apply moving average filter
        smoothed = []
        for i in range(len(cumulative)):
            start = max(0, i - self.smoothing_radius)
            end = min(len(cumulative), i + self.smoothing_radius + 1)
            
            # Average transforms in window
            avg_transform = np.zeros((3, 3))
            for j in range(start, end):
                avg_transform += cumulative[j]
            avg_transform /= (end - start)
            
            smoothed.append(avg_transform)
        
        # Convert back to frame-to-frame transforms
        smooth_transforms = [smoothed[0]]
        for i in range(1, len(smoothed)):
            smooth_transforms.append(
                np.linalg.inv(smoothed[i-1]) @ smoothed[i]
            )
        
        return smooth_transforms
    
    def stabilize_frame(self, frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply stabilization transform to frame"""
        h, w = frame.shape[:2]
        
        if self.mode == StabilizationMode.SMOOTH:
            # Apply smoothed transform
            stabilized = cv2.warpPerspective(frame, transform, (w, h))
            
        elif self.mode == StabilizationMode.LOCKED:
            # Lock to first frame position
            stabilized = cv2.warpPerspective(frame, np.eye(3), (w, h))
            
        elif self.mode == StabilizationMode.CINEMATIC:
            # Apply cinematic smoothing
            transform = self._apply_cinematic_constraints(transform)
            stabilized = cv2.warpPerspective(frame, transform, (w, h))
            
        elif self.mode == StabilizationMode.TRIPOD:
            # Remove all motion
            stabilized = frame.copy()
            
        else:  # HANDHELD
            # Apply subtle smoothing
            transform = self._apply_handheld_feel(transform)
            stabilized = cv2.warpPerspective(frame, transform, (w, h))
        
        return stabilized
    
    def _apply_cinematic_constraints(self, transform: np.ndarray) -> np.ndarray:
        """Apply cinematic motion constraints"""
        # Limit rotation
        angle = np.arctan2(transform[0, 1], transform[0, 0])
        max_angle = np.deg2rad(2)  # Max 2 degrees per frame
        angle = np.clip(angle, -max_angle, max_angle)
        
        # Rebuild transform with constraints
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        constrained = np.array([
            [cos_a, -sin_a, transform[0, 2]],
            [sin_a, cos_a, transform[1, 2]],
            [0, 0, 1]
        ])
        
        return constrained
    
    def _apply_handheld_feel(self, transform: np.ndarray) -> np.ndarray:
        """Add natural handheld motion"""
        # Add slight random motion
        noise = np.random.randn(2) * 0.5
        transform[0, 2] += noise[0]
        transform[1, 2] += noise[1]
        
        return transform

class MotionTrackingSystem:
    """Complete motion tracking and stabilization system"""
    
    def __init__(self):
        self.optical_flow = OpticalFlowTracker()
        self.feature_tracker = FeatureTracker()
        self.stabilizer = VideoStabilizer()
        self.ai_tracker = None  # Lazy load
        self.tracking_data: Dict[int, MotionPath] = {}
        
    def track_object(
        self,
        video_path: str,
        roi: Tuple[int, int, int, int],
        method: TrackingMethod = TrackingMethod.OPTICAL_FLOW,
        output_path: Optional[str] = None
    ) -> MotionPath:
        """Track object through video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        tracking_points = []
        frame_idx = 0
        
        # Initialize tracker
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Cannot read first frame")
        
        if method == TrackingMethod.OPTICAL_FLOW:
            prev_pts = self.optical_flow.initialize(first_frame, roi)
            prev_frame = first_frame
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Track points
                curr_pts, prev_pts_filtered, errors = self.optical_flow.track(
                    prev_frame, frame, prev_pts
                )
                
                # Update tracking data
                if len(curr_pts) > 0:
                    avg_pos = np.mean(curr_pts.reshape(-1, 2), axis=0)
                    confidence = 1.0 - np.mean(errors) / 100.0 if len(errors) > 0 else 0.0
                    
                    tracking_points.append(TrackingPoint(
                        id=0,
                        position=(float(avg_pos[0]), float(avg_pos[1])),
                        confidence=confidence,
                        frame=frame_idx
                    ))
                
                prev_frame = frame
                prev_pts = curr_pts
                frame_idx += 1
                
        elif method == TrackingMethod.FEATURE_BASED:
            prev_kp, prev_desc = self.feature_tracker.detect_and_compute(first_frame)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect and match features
                curr_kp, curr_desc = self.feature_tracker.detect_and_compute(frame)
                matches = self.feature_tracker.match_features(prev_desc, curr_desc)
                
                if matches:
                    # Calculate average motion
                    motions = []
                    for match in matches:
                        pt1 = prev_kp[match.queryIdx].pt
                        pt2 = curr_kp[match.trainIdx].pt
                        motions.append((pt2[0] - pt1[0], pt2[1] - pt1[1]))
                    
                    avg_motion = np.mean(motions, axis=0)
                    
                    # Update position based on ROI center
                    if tracking_points:
                        last_pos = tracking_points[-1].position
                        new_pos = (
                            last_pos[0] + avg_motion[0],
                            last_pos[1] + avg_motion[1]
                        )
                    else:
                        new_pos = (roi[0] + roi[2]/2, roi[1] + roi[3]/2)
                    
                    tracking_points.append(TrackingPoint(
                        id=0,
                        position=new_pos,
                        confidence=len(matches) / 100.0,
                        frame=frame_idx
                    ))
                
                prev_kp, prev_desc = curr_kp, curr_desc
                frame_idx += 1
        
        cap.release()
        
        # Create motion path
        motion_path = MotionPath(
            points=tracking_points,
            object_id=0,
            start_frame=0,
            end_frame=frame_idx - 1
        )
        
        # Smooth path if needed
        motion_path.smoothed_path = self._smooth_path(tracking_points)
        
        # Save tracking data if output specified
        if output_path:
            self._save_tracking_data(motion_path, output_path)
        
        return motion_path
    
    def stabilize_video(
        self,
        input_path: str,
        output_path: str,
        mode: StabilizationMode = StabilizationMode.SMOOTH,
        crop_ratio: float = 0.9
    ) -> bool:
        """Stabilize entire video"""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {input_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize stabilizer
        self.stabilizer.mode = mode
        transforms = []
        
        # First pass: calculate transforms
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=30
            )
            
            if prev_pts is not None:
                # Track features
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, prev_pts, None
                )
                
                # Filter good points
                idx = np.where(status == 1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                
                # Estimate transform
                transform = self.stabilizer.estimate_transform(prev_pts, curr_pts)
                transforms.append(transform)
            else:
                transforms.append(np.eye(3))
            
            prev_gray = curr_gray
        
        # Smooth trajectory
        smoothed_transforms = self.stabilizer.smooth_trajectory(transforms)
        
        # Second pass: apply stabilization
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for transform in smoothed_transforms:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply stabilization
            stabilized = self.stabilizer.stabilize_frame(frame, transform)
            
            # Crop to remove borders
            if crop_ratio < 1.0:
                h, w = stabilized.shape[:2]
                crop_h = int(h * crop_ratio)
                crop_w = int(w * crop_ratio)
                y1 = (h - crop_h) // 2
                x1 = (w - crop_w) // 2
                stabilized = stabilized[y1:y1+crop_h, x1:x1+crop_w]
                stabilized = cv2.resize(stabilized, (width, height))
            
            out.write(stabilized)
        
        cap.release()
        out.release()
        
        logger.info(f"Video stabilized: {output_path}")
        return True
    
    def _smooth_path(self, points: List[TrackingPoint], window: int = 5) -> List[Tuple[float, float]]:
        """Apply smoothing to tracking path"""
        if len(points) < window:
            return [(p.position[0], p.position[1]) for p in points]
        
        smoothed = []
        for i in range(len(points)):
            start = max(0, i - window // 2)
            end = min(len(points), i + window // 2 + 1)
            
            # Average positions in window
            x_avg = np.mean([points[j].position[0] for j in range(start, end)])
            y_avg = np.mean([points[j].position[1] for j in range(start, end)])
            
            smoothed.append((x_avg, y_avg))
        
        return smoothed
    
    def _save_tracking_data(self, motion_path: MotionPath, output_path: str):
        """Save tracking data to JSON"""
        data = {
            "object_id": motion_path.object_id,
            "start_frame": motion_path.start_frame,
            "end_frame": motion_path.end_frame,
            "points": [
                {
                    "frame": p.frame,
                    "x": p.position[0],
                    "y": p.position[1],
                    "confidence": p.confidence
                }
                for p in motion_path.points
            ]
        }
        
        if motion_path.smoothed_path:
            data["smoothed_path"] = [
                {"x": p[0], "y": p[1]} for p in motion_path.smoothed_path
            ]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Tracking data saved: {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize system
    tracker = MotionTrackingSystem()
    
    # Track object
    motion_path = tracker.track_object(
        "/Users/hawzhin/Videos/test_30s.mp4",
        roi=(100, 100, 200, 200),  # x, y, width, height
        method=TrackingMethod.OPTICAL_FLOW,
        output_path="/tmp/tracking_data.json"
    )
    
    print(f"✅ Tracked {len(motion_path.points)} points")
    
    # Stabilize video
    success = tracker.stabilize_video(
        "/Users/hawzhin/Videos/test_30s.mp4",
        "/tmp/stabilized_output.mp4",
        mode=StabilizationMode.SMOOTH
    )
    
    if success:
        print("✅ Video stabilized successfully")