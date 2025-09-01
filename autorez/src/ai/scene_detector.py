#!/usr/bin/env python3
"""
AI-Powered Scene Detection with Multiple ML Models
Advanced scene boundary detection, shot classification, and visual analysis
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
from collections import deque
import time
from scipy import signal
from sklearn.cluster import DBSCAN
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ShotType(Enum):
    """Types of camera shots"""
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSEUP = "closeup"
    EXTREME_CLOSEUP = "extreme_closeup"
    ESTABLISHING = "establishing"
    POV = "pov"
    OVER_SHOULDER = "over_shoulder"
    TWO_SHOT = "two_shot"
    AERIAL = "aerial"

class TransitionType(Enum):
    """Types of transitions between scenes"""
    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    MATCH_CUT = "match_cut"
    JUMP_CUT = "jump_cut"
    L_CUT = "l_cut"
    J_CUT = "j_cut"

@dataclass
class SceneBoundary:
    """Represents a scene boundary/cut point"""
    frame_num: int
    timestamp: float
    confidence: float
    transition_type: TransitionType
    visual_change: float
    audio_change: float
    motion_change: float

@dataclass
class Scene:
    """Represents a detected scene"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    shot_type: ShotType
    dominant_color: Tuple[int, int, int]
    avg_motion: float
    key_objects: List[str]
    emotional_tone: str
    confidence: float

class SceneDetectionCNN(nn.Module):
    """CNN for scene boundary detection"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Temporal convolution for sequence analysis
        self.temporal_conv = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        
        # Classification layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)  # Binary: scene boundary or not
        
    def forward(self, x):
        # Spatial features
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

class TransformerSceneAnalyzer(nn.Module):
    """Transformer-based scene understanding model"""
    
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.shot_classifier = nn.Linear(d_model, len(ShotType))
        self.emotion_classifier = nn.Linear(d_model, 7)  # 7 basic emotions
        
    def forward(self, features, mask=None):
        # Add positional encoding
        seq_len = features.size(1)
        pos = torch.arange(seq_len, device=features.device).unsqueeze(0)
        features = features + self.pos_encoder(pos)
        
        # Transformer encoding
        features = features.transpose(0, 1)  # (seq, batch, features)
        encoded = self.transformer(features, mask=mask)
        encoded = encoded.transpose(0, 1)  # (batch, seq, features)
        
        # Classifications
        shot_logits = self.shot_classifier(encoded)
        emotion_logits = self.emotion_classifier(encoded)
        
        return shot_logits, emotion_logits

class AISceneDetector:
    """Main AI-powered scene detection system"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize models
        self.cnn_detector = SceneDetectionCNN().to(self.device)
        self.transformer_analyzer = TransformerSceneAnalyzer().to(self.device)
        
        # Load pretrained weights if available
        if model_path and Path(model_path).exists():
            self.load_models(model_path)
        else:
            logger.info("Using randomly initialized models")
        
        self.cnn_detector.eval()
        self.transformer_analyzer.eval()
        
        # Detection parameters
        self.min_scene_duration = 0.5  # seconds
        self.threshold = 0.7
        self.window_size = 5  # frames
        
    def detect_scenes(self, video_path: str, progress_callback=None) -> Tuple[List[Scene], List[SceneBoundary]]:
        """Detect scenes in video using multiple techniques"""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scenes = []
        boundaries = []
        
        # Feature buffers
        frame_buffer = deque(maxlen=self.window_size)
        feature_buffer = []
        motion_buffer = []
        
        prev_frame = None
        prev_features = None
        
        frame_num = 0
        current_scene_start = 0
        
        logger.info(f"Processing {total_frames} frames at {fps} fps")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for processing
            processed_frame = cv2.resize(frame, (224, 224))
            frame_buffer.append(processed_frame)
            
            # Extract features
            features = self._extract_features(processed_frame)
            feature_buffer.append(features)
            
            # Calculate motion
            if prev_frame is not None:
                motion = self._calculate_optical_flow(prev_frame, processed_frame)
                motion_buffer.append(motion)
            
            # Detect scene boundary
            if len(frame_buffer) == self.window_size:
                is_boundary, confidence, transition_type = self._detect_boundary(
                    frame_buffer, feature_buffer[-self.window_size:]
                )
                
                if is_boundary and confidence > self.threshold:
                    # Calculate change metrics
                    visual_change = self._calculate_visual_change(frame_buffer)
                    audio_change = 0.0  # Placeholder for audio analysis
                    motion_change = np.mean(motion_buffer[-3:]) if motion_buffer else 0.0
                    
                    boundary = SceneBoundary(
                        frame_num=frame_num,
                        timestamp=frame_num / fps,
                        confidence=confidence,
                        transition_type=transition_type,
                        visual_change=visual_change,
                        audio_change=audio_change,
                        motion_change=motion_change
                    )
                    boundaries.append(boundary)
                    
                    # Create scene
                    if frame_num - current_scene_start > self.min_scene_duration * fps:
                        scene = self._analyze_scene(
                            feature_buffer[current_scene_start:frame_num],
                            current_scene_start,
                            frame_num,
                            fps
                        )
                        scenes.append(scene)
                        current_scene_start = frame_num
            
            # Progress callback
            if progress_callback and frame_num % 30 == 0:
                progress = frame_num / total_frames
                progress_callback(progress)
            
            prev_frame = processed_frame
            prev_features = features
            frame_num += 1
        
        # Add final scene
        if frame_num - current_scene_start > self.min_scene_duration * fps:
            scene = self._analyze_scene(
                feature_buffer[current_scene_start:],
                current_scene_start,
                frame_num,
                fps
            )
            scenes.append(scene)
        
        cap.release()
        
        logger.info(f"Detected {len(scenes)} scenes with {len(boundaries)} boundaries")
        return scenes, boundaries
    
    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract visual features from frame"""
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device) / 255.0
        
        with torch.no_grad():
            # Use CNN for feature extraction
            features = self.cnn_detector.conv3(
                self.cnn_detector.pool2(
                    F.relu(self.cnn_detector.bn2(
                        self.cnn_detector.conv2(
                            self.cnn_detector.pool1(
                                F.relu(self.cnn_detector.bn1(
                                    self.cnn_detector.conv1(frame_tensor)
                                ))
                            )
                        )
                    ))
                )
            )
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.squeeze().cpu().numpy()
        
        return features
    
    def _detect_boundary(
        self,
        frames: deque,
        features: List[np.ndarray]
    ) -> Tuple[bool, float, TransitionType]:
        """Detect if current position is a scene boundary"""
        
        # Method 1: Feature similarity
        if len(features) >= 2:
            feature_diff = np.linalg.norm(features[-1] - features[-2])
            
            # Method 2: Histogram comparison
            hist_diff = self._compare_histograms(frames[-1], frames[-2])
            
            # Method 3: SSIM
            ssim_score = ssim(
                cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frames[-2], cv2.COLOR_BGR2GRAY)
            )
            
            # Method 4: Edge detection
            edge_diff = self._compare_edges(frames[-1], frames[-2])
            
            # Combine metrics
            combined_score = (
                feature_diff * 0.3 +
                hist_diff * 0.2 +
                (1 - ssim_score) * 0.3 +
                edge_diff * 0.2
            )
            
            # Determine transition type
            transition_type = self._classify_transition(frames, combined_score)
            
            # Threshold for boundary
            is_boundary = combined_score > 0.4
            confidence = min(combined_score / 0.6, 1.0)
            
            return is_boundary, confidence, transition_type
        
        return False, 0.0, TransitionType.CUT
    
    def _calculate_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate optical flow between frames"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.mean(magnitude)
    
    def _calculate_visual_change(self, frames: deque) -> float:
        """Calculate overall visual change in frame buffer"""
        if len(frames) < 2:
            return 0.0
        
        changes = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i-1])
            changes.append(np.mean(diff))
        
        return np.mean(changes)
    
    def _compare_histograms(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compare color histograms of two frames"""
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        return 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def _compare_edges(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compare edge maps of two frames"""
        edges1 = cv2.Canny(frame1, 50, 150)
        edges2 = cv2.Canny(frame2, 50, 150)
        
        diff = cv2.absdiff(edges1, edges2)
        return np.mean(diff) / 255.0
    
    def _classify_transition(self, frames: deque, score: float) -> TransitionType:
        """Classify the type of transition"""
        if score > 0.8:
            return TransitionType.CUT
        elif score > 0.6:
            # Check for fade
            if self._is_fade(frames):
                return TransitionType.FADE
            else:
                return TransitionType.DISSOLVE
        elif score > 0.4:
            return TransitionType.MATCH_CUT
        else:
            return TransitionType.JUMP_CUT
    
    def _is_fade(self, frames: deque) -> bool:
        """Check if transition is a fade"""
        if len(frames) < 3:
            return False
        
        # Check for consistent darkening or brightening
        brightnesses = [np.mean(frame) for frame in frames]
        diff = np.diff(brightnesses)
        
        # All increasing or all decreasing indicates fade
        return np.all(diff > 0) or np.all(diff < 0)
    
    def _analyze_scene(
        self,
        features: List[np.ndarray],
        start_frame: int,
        end_frame: int,
        fps: float
    ) -> Scene:
        """Analyze a scene segment"""
        
        # Stack features for transformer analysis
        if features:
            feature_tensor = torch.tensor(np.array(features)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                shot_logits, emotion_logits = self.transformer_analyzer(feature_tensor)
                
                # Get predictions
                shot_type_idx = torch.argmax(shot_logits[0, -1]).item()
                emotion_idx = torch.argmax(emotion_logits[0, -1]).item()
        else:
            shot_type_idx = 0
            emotion_idx = 0
        
        # Map to enums
        shot_types = list(ShotType)
        emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
        
        shot_type = shot_types[min(shot_type_idx, len(shot_types)-1)]
        emotional_tone = emotions[min(emotion_idx, len(emotions)-1)]
        
        # Calculate scene properties
        duration = (end_frame - start_frame) / fps
        
        # Placeholder values for demo
        dominant_color = (128, 128, 128)
        avg_motion = 0.5
        key_objects = ["person", "background"]
        confidence = 0.85
        
        return Scene(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_frame / fps,
            end_time=end_frame / fps,
            duration=duration,
            shot_type=shot_type,
            dominant_color=dominant_color,
            avg_motion=avg_motion,
            key_objects=key_objects,
            emotional_tone=emotional_tone,
            confidence=confidence
        )
    
    def cluster_scenes(self, scenes: List[Scene], min_samples=2) -> List[List[Scene]]:
        """Cluster similar scenes together"""
        if len(scenes) < min_samples:
            return [scenes]
        
        # Extract features for clustering
        features = []
        for scene in scenes:
            feature = [
                scene.duration,
                scene.avg_motion,
                float(scene.shot_type == ShotType.WIDE),
                float(scene.shot_type == ShotType.CLOSEUP),
                len(scene.key_objects)
            ]
            features.append(feature)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=min_samples)
        labels = clustering.fit_predict(features)
        
        # Group scenes by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(scenes[i])
        
        return list(clusters.values())
    
    def save_timeline(self, scenes: List[Scene], boundaries: List[SceneBoundary], output_path: str):
        """Save scene detection results to JSON"""
        timeline = {
            "scenes": [
                {
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration": s.duration,
                    "shot_type": s.shot_type.value,
                    "emotional_tone": s.emotional_tone,
                    "confidence": s.confidence
                }
                for s in scenes
            ],
            "boundaries": [
                {
                    "frame": b.frame_num,
                    "timestamp": b.timestamp,
                    "confidence": b.confidence,
                    "transition_type": b.transition_type.value,
                    "visual_change": b.visual_change
                }
                for b in boundaries
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(timeline, f, indent=2)
        
        logger.info(f"Saved timeline to {output_path}")
    
    def load_models(self, model_path: str):
        """Load pretrained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'cnn_state_dict' in checkpoint:
            self.cnn_detector.load_state_dict(checkpoint['cnn_state_dict'])
        if 'transformer_state_dict' in checkpoint:
            self.transformer_analyzer.load_state_dict(checkpoint['transformer_state_dict'])
        logger.info(f"Loaded models from {model_path}")
    
    def save_models(self, model_path: str):
        """Save model weights"""
        torch.save({
            'cnn_state_dict': self.cnn_detector.state_dict(),
            'transformer_state_dict': self.transformer_analyzer.state_dict()
        }, model_path)
        logger.info(f"Saved models to {model_path}")

# Integration function for pipeline
def detect_scenes_ai(video_path: str, output_path: Optional[str] = None) -> Dict:
    """Main entry point for AI scene detection"""
    
    detector = AISceneDetector()
    
    def progress_callback(progress):
        logger.info(f"Scene detection progress: {progress*100:.1f}%")
    
    scenes, boundaries = detector.detect_scenes(video_path, progress_callback)
    
    # Cluster similar scenes
    scene_clusters = detector.cluster_scenes(scenes)
    
    result = {
        "scenes": len(scenes),
        "boundaries": len(boundaries),
        "clusters": len(scene_clusters),
        "average_scene_duration": np.mean([s.duration for s in scenes]) if scenes else 0,
        "scene_data": scenes,
        "boundary_data": boundaries
    }
    
    if output_path:
        detector.save_timeline(scenes, boundaries, output_path)
    
    return result

if __name__ == "__main__":
    # Test scene detection
    test_video = "/Users/hawzhin/Videos/test_30s.mp4"
    if Path(test_video).exists():
        result = detect_scenes_ai(test_video, "scene_timeline.json")
        print(f"Detected {result['scenes']} scenes")
        print(f"Found {result['boundaries']} boundaries")
        print(f"Grouped into {result['clusters']} clusters")