#!/usr/bin/env python3
"""
Advanced Color Grading Automation System
AI-powered color correction, LUT application, and style matching
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
from scipy import signal, ndimage
from sklearn.cluster import KMeans
import colorsys
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ColorGradeStyle(Enum):
    """Predefined color grading styles"""
    CINEMATIC = "cinematic"
    VINTAGE = "vintage"
    NOIR = "noir"
    TEAL_ORANGE = "teal_orange"
    BLEACH_BYPASS = "bleach_bypass"
    DAY_FOR_NIGHT = "day_for_night"
    HIGH_KEY = "high_key"
    LOW_KEY = "low_key"
    CROSS_PROCESS = "cross_process"
    FILM_EMULATION = "film_emulation"

@dataclass
class ColorProfile:
    """Color profile for a video/image"""
    dominant_colors: List[Tuple[int, int, int]]
    average_brightness: float
    contrast_ratio: float
    saturation_level: float
    color_temperature: float
    tint: float
    histogram: np.ndarray
    
@dataclass
class LUT:
    """Look-Up Table for color grading"""
    name: str
    size: int
    data: np.ndarray
    style: ColorGradeStyle
    intensity: float = 1.0

class ColorMatchingNetwork(nn.Module):
    """Neural network for color style matching"""
    
    def __init__(self, input_dim=256, hidden_dim=512):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 256)
        )
        
        self.style_classifier = nn.Linear(256, len(ColorGradeStyle))
        self.adjustment_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12)  # RGB lift, gamma, gain, offset
        )
        
    def forward(self, color_features):
        encoded = self.encoder(color_features)
        style_logits = self.style_classifier(encoded)
        adjustments = self.adjustment_predictor(encoded)
        return style_logits, adjustments

class AdvancedColorGrading:
    """Advanced color grading automation system"""
    
    def __init__(self, lut_directory: Optional[str] = None):
        self.lut_directory = Path(lut_directory) if lut_directory else Path("luts")
        self.luts = self._load_luts()
        
        # Initialize AI model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.color_matcher = ColorMatchingNetwork().to(self.device)
        self.color_matcher.eval()
        
        # Color science constants
        self.d65_white_point = np.array([0.95047, 1.0, 1.08883])
        
    def analyze_color_profile(self, frame: np.ndarray) -> ColorProfile:
        """Analyze color characteristics of a frame"""
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Extract dominant colors using K-means
        pixels = frame.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = [tuple(color.astype(int)) for color in kmeans.cluster_centers_]
        
        # Calculate metrics
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # Contrast using standard deviation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        # Saturation from HSV
        saturation = np.mean(hsv[:, :, 1])
        
        # Color temperature and tint from LAB
        avg_a = np.mean(lab[:, :, 1]) - 128  # Green-Red
        avg_b = np.mean(lab[:, :, 2]) - 128  # Blue-Yellow
        
        # Approximate color temperature
        color_temp = 6500 + (avg_b * 100)  # Kelvin
        tint = avg_a
        
        # Calculate histogram
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        histogram = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        
        return ColorProfile(
            dominant_colors=dominant_colors,
            average_brightness=brightness / 255.0,
            contrast_ratio=contrast / 128.0,
            saturation_level=saturation / 255.0,
            color_temperature=color_temp,
            tint=tint,
            histogram=histogram
        )
    
    def auto_color_correct(self, frame: np.ndarray) -> np.ndarray:
        """Automatic color correction"""
        
        # Auto white balance
        frame = self._auto_white_balance(frame)
        
        # Auto levels
        frame = self._auto_levels(frame)
        
        # Auto contrast
        frame = self._auto_contrast(frame)
        
        # Auto saturation
        frame = self._auto_saturation(frame)
        
        return frame
    
    def _auto_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """Automatic white balance using gray world assumption"""
        
        # Calculate average for each channel
        avg_b = np.mean(frame[:, :, 0])
        avg_g = np.mean(frame[:, :, 1])
        avg_r = np.mean(frame[:, :, 2])
        
        # Calculate gray value
        gray_value = (avg_b + avg_g + avg_r) / 3
        
        # Calculate scaling factors
        scale_b = gray_value / avg_b if avg_b > 0 else 1
        scale_g = gray_value / avg_g if avg_g > 0 else 1
        scale_r = gray_value / avg_r if avg_r > 0 else 1
        
        # Apply scaling
        balanced = frame.copy().astype(np.float32)
        balanced[:, :, 0] *= scale_b
        balanced[:, :, 1] *= scale_g
        balanced[:, :, 2] *= scale_r
        
        return np.clip(balanced, 0, 255).astype(np.uint8)
    
    def _auto_levels(self, frame: np.ndarray) -> np.ndarray:
        """Auto adjust levels using histogram stretching"""
        
        # Calculate percentiles for black and white points
        percentile_low = 0.5
        percentile_high = 99.5
        
        result = frame.copy()
        
        for i in range(3):
            channel = frame[:, :, i]
            
            # Find black and white points
            black_point = np.percentile(channel, percentile_low)
            white_point = np.percentile(channel, percentile_high)
            
            # Stretch histogram
            if white_point > black_point:
                stretched = (channel - black_point) * 255.0 / (white_point - black_point)
                result[:, :, i] = np.clip(stretched, 0, 255).astype(np.uint8)
        
        return result
    
    def _auto_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Auto adjust contrast using adaptive histogram equalization"""
        
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _auto_saturation(self, frame: np.ndarray) -> np.ndarray:
        """Auto adjust saturation based on content"""
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Calculate current saturation
        current_sat = np.mean(hsv[:, :, 1])
        
        # Target saturation (moderate)
        target_sat = 100
        
        # Calculate adjustment factor
        if current_sat > 0:
            factor = target_sat / current_sat
            factor = np.clip(factor, 0.5, 1.5)  # Limit adjustment
            
            # Apply saturation adjustment
            hsv[:, :, 1] *= factor
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def apply_style(self, frame: np.ndarray, style: ColorGradeStyle) -> np.ndarray:
        """Apply predefined color grading style"""
        
        if style == ColorGradeStyle.CINEMATIC:
            return self._apply_cinematic(frame)
        elif style == ColorGradeStyle.VINTAGE:
            return self._apply_vintage(frame)
        elif style == ColorGradeStyle.NOIR:
            return self._apply_noir(frame)
        elif style == ColorGradeStyle.TEAL_ORANGE:
            return self._apply_teal_orange(frame)
        elif style == ColorGradeStyle.BLEACH_BYPASS:
            return self._apply_bleach_bypass(frame)
        elif style == ColorGradeStyle.DAY_FOR_NIGHT:
            return self._apply_day_for_night(frame)
        elif style == ColorGradeStyle.HIGH_KEY:
            return self._apply_high_key(frame)
        elif style == ColorGradeStyle.LOW_KEY:
            return self._apply_low_key(frame)
        elif style == ColorGradeStyle.CROSS_PROCESS:
            return self._apply_cross_process(frame)
        else:
            return frame
    
    def _apply_cinematic(self, frame: np.ndarray) -> np.ndarray:
        """Apply cinematic color grading"""
        
        # Slightly desaturate
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.85
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Adjust curves - lift shadows, compress highlights
        frame = self._apply_curves(frame, 
            shadows_lift=0.1,
            midtones_gamma=0.9,
            highlights_gain=0.95
        )
        
        # Add slight blue tint to shadows
        frame = self._color_balance(frame, 
            shadows=(0, 0, 5),
            midtones=(0, 0, 0),
            highlights=(5, 3, 0)
        )
        
        return frame
    
    def _apply_vintage(self, frame: np.ndarray) -> np.ndarray:
        """Apply vintage/retro look"""
        
        # Reduce contrast
        frame = self._adjust_contrast(frame, 0.8)
        
        # Warm color temperature
        frame = self._adjust_temperature(frame, 200)
        
        # Add grain
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        # Vignetting
        frame = self._add_vignette(frame, 0.3)
        
        # Sepia tone
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        frame = cv2.transform(frame, kernel)
        
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    def _apply_noir(self, frame: np.ndarray) -> np.ndarray:
        """Apply film noir style"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # High contrast
        gray = self._adjust_contrast_gray(gray, 1.5)
        
        # Crush blacks
        gray[gray < 30] = 0
        
        # Add grain
        noise = np.random.normal(0, 3, gray.shape).astype(np.uint8)
        gray = cv2.add(gray, noise)
        
        # Convert back to BGR
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return frame
    
    def _apply_teal_orange(self, frame: np.ndarray) -> np.ndarray:
        """Apply popular teal and orange look"""
        
        # Split into shadows and highlights
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = (gray < 128).astype(np.float32)
        inv_mask = 1.0 - mask
        
        # Apply teal to shadows
        teal_adjustment = np.zeros_like(frame).astype(np.float32)
        teal_adjustment[:, :, 0] = 10  # Blue
        teal_adjustment[:, :, 1] = 5   # Green
        
        # Apply orange to highlights  
        orange_adjustment = np.zeros_like(frame).astype(np.float32)
        orange_adjustment[:, :, 2] = 10  # Red
        orange_adjustment[:, :, 1] = 5   # Green
        
        # Blend adjustments
        frame = frame.astype(np.float32)
        for i in range(3):
            frame[:, :, i] += mask * teal_adjustment[:, :, i]
            frame[:, :, i] += inv_mask * orange_adjustment[:, :, i]
        
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    def _apply_bleach_bypass(self, frame: np.ndarray) -> np.ndarray:
        """Apply bleach bypass effect"""
        
        # Desaturate
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Overlay with original
        result = cv2.addWeighted(frame, 0.6, gray_bgr, 0.4, 0)
        
        # Increase contrast
        result = self._adjust_contrast(result, 1.3)
        
        return result
    
    def _apply_day_for_night(self, frame: np.ndarray) -> np.ndarray:
        """Convert day scene to night look"""
        
        # Darken overall
        frame = (frame * 0.3).astype(np.uint8)
        
        # Blue color cast
        frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.3, 0, 255)  # Blue channel
        
        # Reduce saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.5
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Add moonlight effect to highlights
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        highlights = (gray > 100).astype(np.float32)
        frame[:, :, 0] = np.clip(frame[:, :, 0] + highlights * 20, 0, 255)
        frame[:, :, 1] = np.clip(frame[:, :, 1] + highlights * 20, 0, 255)
        
        return frame.astype(np.uint8)
    
    def _apply_high_key(self, frame: np.ndarray) -> np.ndarray:
        """Apply high key lighting style"""
        
        # Brighten image
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
        
        # Reduce contrast
        frame = self._adjust_contrast(frame, 0.7)
        
        # Slight overexposure on highlights
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        highlights = (gray > 200).astype(np.float32)
        frame = frame.astype(np.float32)
        frame += highlights[:, :, np.newaxis] * 20
        
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    def _apply_low_key(self, frame: np.ndarray) -> np.ndarray:
        """Apply low key dramatic lighting"""
        
        # Darken image
        frame = cv2.convertScaleAbs(frame, alpha=0.7, beta=-30)
        
        # Increase contrast
        frame = self._adjust_contrast(frame, 1.5)
        
        # Crush shadows
        frame[frame < 20] = 0
        
        return frame
    
    def _apply_cross_process(self, frame: np.ndarray) -> np.ndarray:
        """Apply cross processing effect"""
        
        # Split channels
        b, g, r = cv2.split(frame)
        
        # Apply different curves to each channel
        r = self._apply_curve_to_channel(r, 'convex')
        g = self._apply_curve_to_channel(g, 'linear')
        b = self._apply_curve_to_channel(b, 'concave')
        
        # Merge and add color shift
        frame = cv2.merge([b, g, r])
        
        # Increase saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.3
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return frame
    
    def _apply_curves(
        self,
        frame: np.ndarray,
        shadows_lift: float = 0,
        midtones_gamma: float = 1,
        highlights_gain: float = 1
    ) -> np.ndarray:
        """Apply lift/gamma/gain adjustments"""
        
        frame = frame.astype(np.float32) / 255.0
        
        # Apply adjustments
        frame = frame * highlights_gain + shadows_lift
        frame = np.power(frame, 1.0 / midtones_gamma)
        
        return (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    
    def _color_balance(
        self,
        frame: np.ndarray,
        shadows: Tuple[int, int, int],
        midtones: Tuple[int, int, int],
        highlights: Tuple[int, int, int]
    ) -> np.ndarray:
        """Apply color balance adjustments"""
        
        # Create luminance mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Create masks for different tonal ranges
        shadow_mask = np.clip(1.0 - gray * 2, 0, 1)
        highlight_mask = np.clip(gray * 2 - 1, 0, 1)
        midtone_mask = 1.0 - shadow_mask - highlight_mask
        
        # Apply adjustments
        result = frame.astype(np.float32)
        
        for i in range(3):
            result[:, :, i] += shadow_mask * shadows[i]
            result[:, :, i] += midtone_mask * midtones[i]
            result[:, :, i] += highlight_mask * highlights[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _adjust_contrast(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast of image"""
        
        # Convert to float
        frame = frame.astype(np.float32)
        
        # Calculate mean
        mean = np.mean(frame)
        
        # Adjust contrast
        frame = (frame - mean) * factor + mean
        
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    def _adjust_contrast_gray(self, gray: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast of grayscale image"""
        
        mean = np.mean(gray)
        gray = gray.astype(np.float32)
        gray = (gray - mean) * factor + mean
        
        return np.clip(gray, 0, 255).astype(np.uint8)
    
    def _adjust_temperature(self, frame: np.ndarray, kelvin_shift: float) -> np.ndarray:
        """Adjust color temperature"""
        
        # Simple temperature adjustment
        frame = frame.astype(np.float32)
        
        if kelvin_shift > 0:  # Warmer
            frame[:, :, 2] *= 1 + (kelvin_shift / 1000)  # Increase red
            frame[:, :, 0] *= 1 - (kelvin_shift / 2000)  # Decrease blue
        else:  # Cooler
            frame[:, :, 0] *= 1 + (abs(kelvin_shift) / 1000)  # Increase blue
            frame[:, :, 2] *= 1 - (abs(kelvin_shift) / 2000)  # Decrease red
        
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    def _add_vignette(self, frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Add vignette effect"""
        
        rows, cols = frame.shape[:2]
        
        # Create radial gradient
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols/2)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/2)
        
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        mask = 1.0 - (1.0 - mask) * strength
        
        # Apply vignette
        frame = frame.astype(np.float32)
        for i in range(3):
            frame[:, :, i] *= mask
        
        return frame.astype(np.uint8)
    
    def _apply_curve_to_channel(self, channel: np.ndarray, curve_type: str) -> np.ndarray:
        """Apply curve transformation to single channel"""
        
        normalized = channel.astype(np.float32) / 255.0
        
        if curve_type == 'convex':
            result = np.power(normalized, 0.7)
        elif curve_type == 'concave':
            result = np.power(normalized, 1.5)
        else:  # linear
            result = normalized
        
        return (result * 255).astype(np.uint8)
    
    def apply_lut(self, frame: np.ndarray, lut: LUT) -> np.ndarray:
        """Apply 3D LUT to frame"""
        
        if lut.size == 17:
            return self._apply_lut_17(frame, lut)
        elif lut.size == 33:
            return self._apply_lut_33(frame, lut)
        elif lut.size == 65:
            return self._apply_lut_65(frame, lut)
        else:
            logger.warning(f"Unsupported LUT size: {lut.size}")
            return frame
    
    def _apply_lut_17(self, frame: np.ndarray, lut: LUT) -> np.ndarray:
        """Apply 17x17x17 LUT"""
        # Implementation for 17-point LUT
        return cv2.LUT(frame, lut.data)
    
    def _apply_lut_33(self, frame: np.ndarray, lut: LUT) -> np.ndarray:
        """Apply 33x33x33 LUT"""
        # Implementation for 33-point LUT
        return cv2.LUT(frame, lut.data)
    
    def _apply_lut_65(self, frame: np.ndarray, lut: LUT) -> np.ndarray:
        """Apply 65x65x65 LUT"""
        # Implementation for 65-point LUT
        return cv2.LUT(frame, lut.data)
    
    def match_reference_style(
        self,
        frame: np.ndarray,
        reference: np.ndarray
    ) -> np.ndarray:
        """Match color style of reference image"""
        
        # Analyze both images
        source_profile = self.analyze_color_profile(frame)
        ref_profile = self.analyze_color_profile(reference)
        
        # Calculate adjustments needed
        brightness_adj = ref_profile.average_brightness / (source_profile.average_brightness + 0.001)
        contrast_adj = ref_profile.contrast_ratio / (source_profile.contrast_ratio + 0.001)
        saturation_adj = ref_profile.saturation_level / (source_profile.saturation_level + 0.001)
        
        # Apply adjustments
        result = frame.copy()
        
        # Brightness
        result = cv2.convertScaleAbs(result, alpha=brightness_adj, beta=0)
        
        # Contrast
        result = self._adjust_contrast(result, contrast_adj)
        
        # Saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation_adj
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Color temperature matching
        temp_diff = ref_profile.color_temperature - source_profile.color_temperature
        result = self._adjust_temperature(result, temp_diff)
        
        return result
    
    def _load_luts(self) -> Dict[str, LUT]:
        """Load LUT files from directory"""
        luts = {}
        
        if not self.lut_directory.exists():
            self.lut_directory.mkdir(parents=True)
            self._generate_default_luts()
        
        for lut_file in self.lut_directory.glob("*.cube"):
            try:
                lut = self._parse_cube_lut(lut_file)
                luts[lut.name] = lut
            except Exception as e:
                logger.warning(f"Failed to load LUT {lut_file}: {e}")
        
        return luts
    
    def _parse_cube_lut(self, filepath: Path) -> LUT:
        """Parse .cube LUT file"""
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        size = 17  # Default
        data_lines = []
        title = filepath.stem
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('TITLE'):
                title = line.split('"')[1] if '"' in line else title
            elif line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[-1])
            elif line and not line.startswith('#'):
                # Data line
                data_lines.append(line)
        
        # Parse data
        lut_data = np.zeros((size, size, size, 3))
        idx = 0
        
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    if idx < len(data_lines):
                        values = list(map(float, data_lines[idx].split()))
                        lut_data[b, g, r] = values[:3]
                        idx += 1
        
        return LUT(
            name=title,
            size=size,
            data=lut_data,
            style=ColorGradeStyle.CINEMATIC  # Default
        )
    
    def _generate_default_luts(self):
        """Generate default LUT files"""
        
        # Generate basic LUTs for each style
        for style in ColorGradeStyle:
            lut_data = self._generate_style_lut(style)
            self._save_lut(lut_data, style.value)
    
    def _generate_style_lut(self, style: ColorGradeStyle) -> np.ndarray:
        """Generate LUT for specific style"""
        
        size = 17
        lut = np.zeros((size, size, size, 3))
        
        # Create identity LUT and modify based on style
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    # Normalized coordinates
                    rn = r / (size - 1)
                    gn = g / (size - 1)
                    bn = b / (size - 1)
                    
                    # Apply style-specific transformation
                    if style == ColorGradeStyle.CINEMATIC:
                        # Slight S-curve
                        rn = self._s_curve(rn, 0.1)
                        gn = self._s_curve(gn, 0.1)
                        bn = self._s_curve(bn, 0.15)
                    elif style == ColorGradeStyle.VINTAGE:
                        # Lifted blacks, compressed highlights
                        rn = rn * 0.9 + 0.05
                        gn = gn * 0.85 + 0.05
                        bn = bn * 0.8 + 0.05
                    
                    lut[b, g, r] = [rn, gn, bn]
        
        return lut
    
    def _s_curve(self, x: float, strength: float) -> float:
        """Apply S-curve transformation"""
        
        # Sigmoid function
        midpoint = 0.5
        
        if x < midpoint:
            return midpoint * np.power(x / midpoint, 1 + strength)
        else:
            return 1 - midpoint * np.power((1 - x) / midpoint, 1 + strength)
    
    def _save_lut(self, lut_data: np.ndarray, name: str):
        """Save LUT as .cube file"""
        
        filepath = self.lut_directory / f"{name}.cube"
        size = lut_data.shape[0]
        
        with open(filepath, 'w') as f:
            f.write(f'TITLE "{name}"\n')
            f.write(f'LUT_3D_SIZE {size}\n')
            f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
            f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
            
            for b in range(size):
                for g in range(size):
                    for r in range(size):
                        values = lut_data[b, g, r]
                        f.write(f'{values[0]:.6f} {values[1]:.6f} {values[2]:.6f}\n')
    
    def process_video(
        self,
        video_path: str,
        output_path: str,
        style: Optional[ColorGradeStyle] = None,
        reference_image: Optional[str] = None,
        auto_correct: bool = True
    ):
        """Process entire video with color grading"""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load reference if provided
        ref_frame = None
        if reference_image:
            ref_frame = cv2.imread(reference_image)
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Auto color correct
            if auto_correct:
                frame = self.auto_color_correct(frame)
            
            # Apply style
            if style:
                frame = self.apply_style(frame, style)
            
            # Match reference
            if ref_frame is not None:
                frame = self.match_reference_style(frame, ref_frame)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        
        logger.info(f"Color grading complete: {output_path}")

# Integration function
def apply_color_grading(
    video_path: str,
    style: str = "cinematic",
    output_path: Optional[str] = None
) -> str:
    """Apply color grading to video"""
    
    grader = AdvancedColorGrading()
    
    if not output_path:
        output_path = video_path.replace('.mp4', '_graded.mp4')
    
    style_enum = ColorGradeStyle[style.upper()] if style else None
    
    grader.process_video(
        video_path,
        output_path,
        style=style_enum,
        auto_correct=True
    )
    
    return output_path