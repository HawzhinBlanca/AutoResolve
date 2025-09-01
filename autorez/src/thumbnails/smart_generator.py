#!/usr/bin/env python3
"""
Smart Thumbnail Generation System
AI-powered selection of best frames for thumbnails with composition analysis
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import colorsys

logger = logging.getLogger(__name__)

class ThumbnailStyle(Enum):
    """Thumbnail generation styles"""
    CLEAN = "clean"                # Simple, no overlays
    YOUTUBE = "youtube"            # YouTube-style with text
    NETFLIX = "netflix"            # Netflix-style grid
    CINEMATIC = "cinematic"        # Cinematic bars
    SOCIAL = "social"              # Social media optimized
    POSTER = "poster"              # Movie poster style
    MINIMAL = "minimal"            # Minimalist design
    DYNAMIC = "dynamic"            # Action-oriented

class CompositionRule(Enum):
    """Photographic composition rules"""
    RULE_OF_THIRDS = "rule_of_thirds"
    GOLDEN_RATIO = "golden_ratio"
    CENTER_WEIGHTED = "center_weighted"
    DIAGONAL_LINES = "diagonal_lines"
    SYMMETRY = "symmetry"
    LEADING_LINES = "leading_lines"

@dataclass
class ThumbnailCandidate:
    """Candidate frame for thumbnail"""
    frame: np.ndarray
    frame_number: int
    timestamp: float
    quality_score: float
    composition_score: float
    sharpness_score: float
    color_score: float
    face_score: float
    action_score: float
    metadata: Dict[str, Any]

@dataclass
class ThumbnailConfig:
    """Thumbnail generation configuration"""
    width: int = 1920
    height: int = 1080
    style: ThumbnailStyle = ThumbnailStyle.YOUTUBE
    text_overlay: Optional[str] = None
    logo_path: Optional[str] = None
    blur_background: bool = False
    add_vignette: bool = True
    color_correction: bool = True
    compression_quality: int = 95

class CompositionAnalyzer:
    """Analyze frame composition quality"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def analyze(self, frame: np.ndarray) -> Dict[str, float]:
        """Comprehensive composition analysis"""
        scores = {
            'rule_of_thirds': self._check_rule_of_thirds(frame),
            'golden_ratio': self._check_golden_ratio(frame),
            'symmetry': self._check_symmetry(frame),
            'leading_lines': self._detect_leading_lines(frame),
            'depth': self._analyze_depth(frame),
            'balance': self._check_balance(frame)
        }
        
        # Overall composition score
        scores['overall'] = np.mean(list(scores.values()))
        return scores
    
    def _check_rule_of_thirds(self, frame: np.ndarray) -> float:
        """Check rule of thirds composition"""
        h, w = frame.shape[:2]
        
        # Define thirds grid
        thirds_x = [w // 3, 2 * w // 3]
        thirds_y = [h // 3, 2 * h // 3]
        
        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check edge density at intersection points
        score = 0.0
        for x in thirds_x:
            for y in thirds_y:
                # Sample region around intersection
                region = edges[max(0, y-20):min(h, y+20), 
                              max(0, x-20):min(w, x+20)]
                score += np.sum(region > 0) / region.size
        
        return min(1.0, score * 4)  # Normalize
    
    def _check_golden_ratio(self, frame: np.ndarray) -> float:
        """Check golden ratio composition"""
        h, w = frame.shape[:2]
        phi = 1.618
        
        # Golden ratio points
        golden_x = [int(w / phi), int(w - w / phi)]
        golden_y = [int(h / phi), int(h - h / phi)]
        
        # Similar to rule of thirds but with golden ratio
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        score = 0.0
        for x in golden_x:
            for y in golden_y:
                region = edges[max(0, y-20):min(h, y+20),
                              max(0, x-20):min(w, x+20)]
                score += np.sum(region > 0) / region.size
        
        return min(1.0, score * 4)
    
    def _check_symmetry(self, frame: np.ndarray) -> float:
        """Check horizontal and vertical symmetry"""
        h, w = frame.shape[:2]
        
        # Horizontal symmetry
        top_half = frame[:h//2]
        bottom_half = cv2.flip(frame[h//2:], 0)
        
        # Resize to same size
        min_h = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_h]
        bottom_half = bottom_half[:min_h]
        
        h_sym = 1.0 - np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float))) / 255.0
        
        # Vertical symmetry
        left_half = frame[:, :w//2]
        right_half = cv2.flip(frame[:, w//2:], 1)
        
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]
        
        v_sym = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        
        return max(h_sym, v_sym)
    
    def _detect_leading_lines(self, frame: np.ndarray) -> float:
        """Detect leading lines in composition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return 0.0
        
        # Score based on line convergence
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        convergence_score = 0.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line points toward center
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            dist_to_center = np.sqrt((line_center[0] - center[0])**2 + 
                                     (line_center[1] - center[1])**2)
            
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            convergence_score += 1.0 - (dist_to_center / max_dist)
        
        return min(1.0, convergence_score / max(1, len(lines)))
    
    def _analyze_depth(self, frame: np.ndarray) -> float:
        """Analyze depth and layering"""
        # Simple depth estimation using blur detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Divide into regions
        h, w = frame.shape[:2]
        regions = [
            gray[:h//3],           # Top
            gray[h//3:2*h//3],     # Middle
            gray[2*h//3:]          # Bottom
        ]
        
        # Calculate sharpness variance
        sharpness_values = []
        for region in regions:
            laplacian = cv2.Laplacian(region, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_values.append(sharpness)
        
        # Good depth has varying sharpness
        depth_score = np.std(sharpness_values) / (np.mean(sharpness_values) + 1e-6)
        return min(1.0, depth_score)
    
    def _check_balance(self, frame: np.ndarray) -> float:
        """Check visual balance of composition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate visual weight of quadrants
        quadrants = [
            gray[:h//2, :w//2],      # Top-left
            gray[:h//2, w//2:],      # Top-right
            gray[h//2:, :w//2],      # Bottom-left
            gray[h//2:, w//2:]       # Bottom-right
        ]
        
        weights = [np.mean(q) for q in quadrants]
        
        # Good balance has similar weights
        balance_score = 1.0 - (np.std(weights) / (np.mean(weights) + 1e-6))
        return max(0.0, min(1.0, balance_score))

class QualityAnalyzer:
    """Analyze technical quality of frames"""
    
    def analyze_sharpness(self, frame: np.ndarray) -> float:
        """Measure image sharpness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize to 0-1 range
        return min(1.0, sharpness / 1000.0)
    
    def analyze_color_vibrancy(self, frame: np.ndarray) -> float:
        """Measure color vibrancy and appeal"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Saturation indicates vibrancy
        saturation = hsv[:, :, 1].mean() / 255.0
        
        # Value indicates brightness
        value = hsv[:, :, 2].mean() / 255.0
        
        # Color diversity
        hue = hsv[:, :, 0]
        hue_std = np.std(hue) / 180.0
        
        # Combine metrics
        vibrancy = (saturation * 0.5 + value * 0.3 + hue_std * 0.2)
        return min(1.0, vibrancy)
    
    def detect_faces(self, frame: np.ndarray) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """Detect faces and return score"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return 0.0, []
        
        # Score based on face size and position
        h, w = frame.shape[:2]
        face_score = 0.0
        
        for (x, y, fw, fh) in faces:
            # Size score
            face_area = (fw * fh) / (w * h)
            size_score = min(1.0, face_area * 10)
            
            # Position score (prefer centered faces)
            center_x = x + fw / 2
            center_y = y + fh / 2
            dist_from_center = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            position_score = 1.0 - (dist_from_center / max_dist)
            
            face_score += (size_score * 0.7 + position_score * 0.3)
        
        return min(1.0, face_score / len(faces)), faces.tolist()
    
    def analyze_motion_blur(self, frame: np.ndarray) -> float:
        """Detect motion blur amount"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # FFT to detect motion blur
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Motion blur creates directional patterns in frequency domain
        h, w = magnitude.shape
        center = (h // 2, w // 2)
        
        # Sample radial slices
        angles = np.linspace(0, np.pi, 8)
        variances = []
        
        for angle in angles:
            # Create line through center
            x = np.cos(angle) * np.arange(-min(center), min(center))
            y = np.sin(angle) * np.arange(-min(center), min(center))
            
            x = (x + center[1]).astype(int)
            y = (y + center[0]).astype(int)
            
            # Keep valid indices
            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x, y = x[valid], y[valid]
            
            if len(x) > 0:
                profile = magnitude[y, x]
                variances.append(np.var(profile))
        
        # High variance indicates less blur
        blur_score = 1.0 - (np.std(variances) / (np.mean(variances) + 1e-6))
        return max(0.0, min(1.0, blur_score))

class ThumbnailAISelector(nn.Module):
    """Deep learning model for thumbnail selection"""
    
    def __init__(self, input_size: int = 2048):
        super().__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Scoring head
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict thumbnail quality score"""
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.global_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Score prediction
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return torch.sigmoid(x)

class ThumbnailStyler:
    """Apply styling to thumbnails"""
    
    def __init__(self):
        self.font_cache = {}
        
    def apply_style(
        self,
        image: Image.Image,
        style: ThumbnailStyle,
        config: ThumbnailConfig
    ) -> Image.Image:
        """Apply style to thumbnail"""
        if style == ThumbnailStyle.YOUTUBE:
            return self._style_youtube(image, config)
        elif style == ThumbnailStyle.NETFLIX:
            return self._style_netflix(image, config)
        elif style == ThumbnailStyle.CINEMATIC:
            return self._style_cinematic(image, config)
        elif style == ThumbnailStyle.SOCIAL:
            return self._style_social(image, config)
        elif style == ThumbnailStyle.POSTER:
            return self._style_poster(image, config)
        elif style == ThumbnailStyle.MINIMAL:
            return self._style_minimal(image, config)
        elif style == ThumbnailStyle.DYNAMIC:
            return self._style_dynamic(image, config)
        else:
            return image
    
    def _style_youtube(self, image: Image.Image, config: ThumbnailConfig) -> Image.Image:
        """YouTube-style thumbnail with bold text"""
        img = image.copy()
        
        # Add gradient overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Bottom gradient for text
        for i in range(img.height // 3):
            alpha = int(180 * (i / (img.height // 3)))
            draw.rectangle(
                [(0, img.height - i), (img.width, img.height)],
                fill=(0, 0, 0, alpha)
            )
        
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        # Add text if specified
        if config.text_overlay:
            draw = ImageDraw.Draw(img)
            
            # Large, bold text
            font_size = img.height // 8
            try:
                font = ImageFont.truetype("Arial-Bold", font_size)
            except:
                font = ImageFont.load_default()
            
            # Text with outline
            text = config.text_overlay.upper()
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (img.width - text_width) // 2
            y = img.height - text_height - 50
            
            # Draw outline
            for adj_x in [-2, 0, 2]:
                for adj_y in [-2, 0, 2]:
                    draw.text((x + adj_x, y + adj_y), text, 
                             font=font, fill=(0, 0, 0, 255))
            
            # Draw main text
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        
        # Add vignette
        if config.add_vignette:
            img = self._add_vignette(img)
        
        return img.convert('RGB')
    
    def _style_netflix(self, image: Image.Image, config: ThumbnailConfig) -> Image.Image:
        """Netflix-style grid thumbnail"""
        img = image.copy()
        
        # Darken image slightly
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)
        
        # Add Netflix-style bottom bar
        draw = ImageDraw.Draw(img)
        bar_height = img.height // 10
        draw.rectangle(
            [(0, img.height - bar_height), (img.width, img.height)],
            fill=(20, 20, 20)
        )
        
        # Add progress indicator
        progress_width = img.width // 3
        draw.rectangle(
            [(0, img.height - 4), (progress_width, img.height)],
            fill=(229, 9, 20)  # Netflix red
        )
        
        return img
    
    def _style_cinematic(self, image: Image.Image, config: ThumbnailConfig) -> Image.Image:
        """Cinematic style with letterbox bars"""
        img = image.copy()
        
        # Add cinematic bars
        bar_height = img.height // 8
        draw = ImageDraw.Draw(img)
        
        # Top bar
        draw.rectangle([(0, 0), (img.width, bar_height)], fill=(0, 0, 0))
        
        # Bottom bar
        draw.rectangle(
            [(0, img.height - bar_height), (img.width, img.height)],
            fill=(0, 0, 0)
        )
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        
        # Add film grain
        noise = Image.effect_noise(img.size, 20)
        img = Image.blend(img, noise, 0.05)
        
        return img
    
    def _style_social(self, image: Image.Image, config: ThumbnailConfig) -> Image.Image:
        """Social media optimized style"""
        img = image.copy()
        
        # Square crop for Instagram
        size = min(img.width, img.height)
        left = (img.width - size) // 2
        top = (img.height - size) // 2
        img = img.crop((left, top, left + size, top + size))
        img = img.resize((1080, 1080), Image.Resampling.LANCZOS)
        
        # Increase saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.3)
        
        # Add subtle gradient overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for i in range(img.height // 4):
            alpha = int(60 * (1 - i / (img.height // 4)))
            draw.rectangle(
                [(0, i), (img.width, i + 1)],
                fill=(255, 200, 100, alpha)
            )
        
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        return img.convert('RGB')
    
    def _style_poster(self, image: Image.Image, config: ThumbnailConfig) -> Image.Image:
        """Movie poster style"""
        img = image.copy()
        
        # Portrait orientation
        target_ratio = 2 / 3
        current_ratio = img.width / img.height
        
        if current_ratio > target_ratio:
            # Crop width
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            # Crop height
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
        
        # Dramatic lighting
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.4)
        
        # Add title area at bottom
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for i in range(img.height // 3):
            alpha = int(200 * (i / (img.height // 3)))
            draw.rectangle(
                [(0, img.height - i), (img.width, img.height)],
                fill=(0, 0, 0, alpha)
            )
        
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        return img.convert('RGB')
    
    def _style_minimal(self, image: Image.Image, config: ThumbnailConfig) -> Image.Image:
        """Minimalist style"""
        img = image.copy()
        
        # Desaturate slightly
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.7)
        
        # Add white border
        border_size = 20
        img_with_border = Image.new('RGB', 
                                    (img.width + 2 * border_size,
                                     img.height + 2 * border_size),
                                    'white')
        img_with_border.paste(img, (border_size, border_size))
        
        return img_with_border
    
    def _style_dynamic(self, image: Image.Image, config: ThumbnailConfig) -> Image.Image:
        """Dynamic action-oriented style"""
        img = image.copy()
        
        # Motion blur effect on edges
        img_blur = img.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Create radial mask
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        center = (img.width // 2, img.height // 2)
        radius = min(img.width, img.height) // 3
        
        draw.ellipse(
            [(center[0] - radius, center[1] - radius),
             (center[0] + radius, center[1] + radius)],
            fill=255
        )
        
        mask = mask.filter(ImageFilter.GaussianBlur(radius=50))
        
        # Composite sharp center with blurred edges
        img = Image.composite(img, img_blur, mask)
        
        # Boost saturation and contrast
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.4)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        return img
    
    def _add_vignette(self, image: Image.Image) -> Image.Image:
        """Add vignette effect"""
        img = image.copy()
        
        # Create radial gradient
        w, h = img.size
        center = (w // 2, h // 2)
        max_radius = np.sqrt(center[0]**2 + center[1]**2)
        
        vignette = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(vignette)
        
        for i in range(int(max_radius)):
            alpha = int(100 * (i / max_radius)**2)
            draw.ellipse(
                [(center[0] - i, center[1] - i),
                 (center[0] + i, center[1] + i)],
                outline=(0, 0, 0, alpha)
            )
        
        return Image.alpha_composite(img.convert('RGBA'), vignette).convert('RGB')

class SmartThumbnailGenerator:
    """Complete smart thumbnail generation system"""
    
    def __init__(self):
        self.composition_analyzer = CompositionAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
        self.styler = ThumbnailStyler()
        self.ai_selector = None  # Lazy load
        self.cache_dir = Path("/tmp/thumbnail_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def generate_thumbnails(
        self,
        video_path: str,
        output_dir: str,
        count: int = 3,
        config: Optional[ThumbnailConfig] = None
    ) -> List[str]:
        """Generate smart thumbnails from video"""
        if not config:
            config = ThumbnailConfig()
        
        # Extract candidate frames
        candidates = self._extract_candidates(video_path, count * 10)
        
        # Score and rank candidates
        scored_candidates = self._score_candidates(candidates)
        
        # Select best thumbnails
        selected = sorted(scored_candidates, 
                         key=lambda x: x.quality_score, 
                         reverse=True)[:count]
        
        # Generate styled thumbnails
        output_paths = []
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for i, candidate in enumerate(selected):
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(candidate.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize to target dimensions
            img = img.resize((config.width, config.height), 
                           Image.Resampling.LANCZOS)
            
            # Apply style
            styled = self.styler.apply_style(img, config.style, config)
            
            # Save thumbnail
            output_path = output_dir / f"thumbnail_{i+1}.jpg"
            styled.save(output_path, quality=config.compression_quality)
            output_paths.append(str(output_path))
            
            logger.info(f"Generated thumbnail: {output_path}")
        
        return output_paths
    
    def _extract_candidates(
        self,
        video_path: str,
        count: int
    ) -> List[ThumbnailCandidate]:
        """Extract candidate frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, count, dtype=int)
        
        candidates = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                candidate = ThumbnailCandidate(
                    frame=frame,
                    frame_number=idx,
                    timestamp=idx / fps,
                    quality_score=0.0,
                    composition_score=0.0,
                    sharpness_score=0.0,
                    color_score=0.0,
                    face_score=0.0,
                    action_score=0.0,
                    metadata={}
                )
                candidates.append(candidate)
        
        cap.release()
        return candidates
    
    def _score_candidates(
        self,
        candidates: List[ThumbnailCandidate]
    ) -> List[ThumbnailCandidate]:
        """Score thumbnail candidates"""
        for candidate in candidates:
            # Composition analysis
            comp_scores = self.composition_analyzer.analyze(candidate.frame)
            candidate.composition_score = comp_scores['overall']
            
            # Quality analysis
            candidate.sharpness_score = self.quality_analyzer.analyze_sharpness(
                candidate.frame
            )
            candidate.color_score = self.quality_analyzer.analyze_color_vibrancy(
                candidate.frame
            )
            
            # Face detection
            face_score, faces = self.quality_analyzer.detect_faces(candidate.frame)
            candidate.face_score = face_score
            candidate.metadata['faces'] = faces
            
            # Motion blur
            blur_score = self.quality_analyzer.analyze_motion_blur(candidate.frame)
            candidate.metadata['blur_score'] = blur_score
            
            # Calculate overall quality score
            weights = {
                'composition': 0.3,
                'sharpness': 0.25,
                'color': 0.2,
                'face': 0.15,
                'blur': 0.1
            }
            
            candidate.quality_score = (
                candidate.composition_score * weights['composition'] +
                candidate.sharpness_score * weights['sharpness'] +
                candidate.color_score * weights['color'] +
                candidate.face_score * weights['face'] +
                blur_score * weights['blur']
            )
        
        return candidates
    
    def generate_contact_sheet(
        self,
        video_path: str,
        output_path: str,
        rows: int = 4,
        cols: int = 4,
        width: int = 1920
    ) -> bool:
        """Generate contact sheet of video frames"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate thumbnail size
        thumb_width = width // cols
        thumb_height = int(thumb_width * 9 / 16)  # Assume 16:9
        
        # Create contact sheet
        sheet_height = thumb_height * rows
        contact_sheet = Image.new('RGB', (width, sheet_height))
        
        # Sample frames
        frame_indices = np.linspace(0, total_frames - 1, rows * cols, dtype=int)
        
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert and resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((thumb_width, thumb_height), 
                               Image.Resampling.LANCZOS)
                
                # Calculate position
                row = i // cols
                col = i % cols
                x = col * thumb_width
                y = row * thumb_height
                
                # Paste into sheet
                contact_sheet.paste(img, (x, y))
        
        cap.release()
        
        # Save contact sheet
        contact_sheet.save(output_path, quality=95)
        logger.info(f"Generated contact sheet: {output_path}")
        
        return True

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = SmartThumbnailGenerator()
    
    # Generate thumbnails
    config = ThumbnailConfig(
        style=ThumbnailStyle.YOUTUBE,
        text_overlay="AMAZING VIDEO",
        add_vignette=True
    )
    
    thumbnails = generator.generate_thumbnails(
        "/Users/hawzhin/Videos/test_30s.mp4",
        "/tmp/thumbnails",
        count=3,
        config=config
    )
    
    print(f"✅ Generated {len(thumbnails)} thumbnails")
    
    # Generate contact sheet
    generator.generate_contact_sheet(
        "/Users/hawzhin/Videos/test_30s.mp4",
        "/tmp/contact_sheet.jpg"
    )
    
    print("✅ Generated contact sheet")