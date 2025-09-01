#!/usr/bin/env python3
"""
Advanced Effects Library
Comprehensive collection of video and audio effects with GPU acceleration
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import signal, ndimage
from PIL import Image, ImageFilter, ImageEnhance
import cupy as cp  # GPU acceleration

logger = logging.getLogger(__name__)

class EffectType(Enum):
    """Effect categories"""
    BLUR = "blur"
    SHARPEN = "sharpen"
    DISTORTION = "distortion"
    COLOR = "color"
    STYLIZE = "stylize"
    TRANSITION = "transition"
    GENERATIVE = "generative"
    CORRECTION = "correction"
    ARTISTIC = "artistic"
    TEMPORAL = "temporal"

@dataclass
class EffectParameters:
    """Parameters for an effect"""
    intensity: float = 1.0
    blend_mode: str = "normal"
    mask: Optional[np.ndarray] = None
    keyframes: Optional[List[Tuple[float, float]]] = None
    gpu_enabled: bool = True

class BlurEffects:
    """Collection of blur effects"""
    
    @staticmethod
    def gaussian_blur(frame: np.ndarray, radius: float = 5.0, gpu: bool = True) -> np.ndarray:
        """Gaussian blur effect"""
        if gpu and cp:
            frame_gpu = cp.asarray(frame)
            kernel_size = int(radius * 2) | 1
            blurred = cp.asnumpy(ndimage.gaussian_filter(frame_gpu, sigma=radius))
            return blurred
        else:
            kernel_size = int(radius * 2) | 1
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), radius)
    
    @staticmethod
    def motion_blur(frame: np.ndarray, angle: float = 0, length: int = 15) -> np.ndarray:
        """Directional motion blur"""
        # Create motion kernel
        kernel = np.zeros((length, length))
        center = length // 2
        
        # Draw line at angle
        angle_rad = np.deg2rad(angle)
        for i in range(length):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
        
        kernel = kernel / np.sum(kernel)
        return cv2.filter2D(frame, -1, kernel)
    
    @staticmethod
    def radial_blur(frame: np.ndarray, center: Tuple[int, int], strength: float = 0.1) -> np.ndarray:
        """Radial blur effect"""
        h, w = frame.shape[:2]
        cx, cy = center
        
        # Create radial blur
        result = frame.copy()
        iterations = int(strength * 10)
        
        for i in range(1, iterations + 1):
            scale = 1 + (i * strength / iterations)
            
            # Create scaled version
            scaled = cv2.resize(frame, None, fx=scale, fy=scale)
            sh, sw = scaled.shape[:2]
            
            # Calculate crop region
            x1 = max(0, int((sw - w) / 2))
            y1 = max(0, int((sh - h) / 2))
            x2 = min(sw, x1 + w)
            y2 = min(sh, y1 + h)
            
            # Blend
            alpha = 1.0 / iterations
            cropped = scaled[y1:y2, x1:x2]
            if cropped.shape == result.shape:
                result = cv2.addWeighted(result, 1 - alpha, cropped, alpha, 0)
        
        return result
    
    @staticmethod
    def lens_blur(frame: np.ndarray, aperture: int = 5, focal_point: Tuple[int, int] = None) -> np.ndarray:
        """Lens blur with bokeh effect"""
        if focal_point is None:
            h, w = frame.shape[:2]
            focal_point = (w // 2, h // 2)
        
        # Create depth map based on distance from focal point
        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        
        fx, fy = focal_point
        distance = np.sqrt((x - fx)**2 + (y - fy)**2)
        depth_map = distance / np.max(distance)
        
        # Apply variable blur based on depth
        result = frame.copy()
        levels = 5
        
        for i in range(levels):
            depth_min = i / levels
            depth_max = (i + 1) / levels
            
            # Create mask for this depth level
            mask = ((depth_map >= depth_min) & (depth_map < depth_max)).astype(np.uint8)
            
            # Apply blur proportional to depth
            blur_amount = aperture * (i + 1) / levels
            if blur_amount > 0:
                blurred = cv2.GaussianBlur(frame, (int(blur_amount * 2) | 1, int(blur_amount * 2) | 1), blur_amount)
                
                # Blend using mask
                mask_3ch = cv2.merge([mask, mask, mask])
                result = np.where(mask_3ch > 0, blurred, result)
        
        return result

class ColorEffects:
    """Collection of color effects"""
    
    @staticmethod
    def chromatic_aberration(frame: np.ndarray, shift: int = 5) -> np.ndarray:
        """Chromatic aberration effect"""
        b, g, r = cv2.split(frame)
        
        # Shift channels
        h, w = frame.shape[:2]
        
        # Red channel shift right
        r_shifted = np.zeros_like(r)
        r_shifted[:, shift:] = r[:, :-shift]
        
        # Blue channel shift left
        b_shifted = np.zeros_like(b)
        b_shifted[:, :-shift] = b[:, shift:]
        
        return cv2.merge([b_shifted, g, r_shifted])
    
    @staticmethod
    def duotone(frame: np.ndarray, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> np.ndarray:
        """Duotone color effect"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create gradient map
        gradient = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            gradient[i, 0] = [
                int(color1[0] * (1 - t) + color2[0] * t),
                int(color1[1] * (1 - t) + color2[1] * t),
                int(color1[2] * (1 - t) + color2[2] * t)
            ]
        
        # Apply gradient map
        return cv2.LUT(cv2.merge([gray, gray, gray]), gradient)
    
    @staticmethod
    def color_isolation(frame: np.ndarray, target_color: Tuple[int, int, int], tolerance: int = 30) -> np.ndarray:
        """Isolate specific color, desaturate others"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for target color
        target_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0, 0]
        
        lower = np.array([max(0, target_hsv[0] - tolerance), 50, 50])
        upper = np.array([min(179, target_hsv[0] + tolerance), 255, 255])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Desaturate non-masked areas
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Combine
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch > 0, frame, gray_bgr)
        
        return result
    
    @staticmethod
    def gradient_map(frame: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Apply gradient map to image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ensure gradient is 256x3
        if gradient.shape[0] != 256:
            gradient = cv2.resize(gradient, (1, 256))
        
        # Apply LUT
        result = np.zeros_like(frame)
        for c in range(3):
            result[:, :, c] = gradient[gray, c]
        
        return result

class DistortionEffects:
    """Collection of distortion effects"""
    
    @staticmethod
    def wave_distortion(frame: np.ndarray, amplitude: float = 20, frequency: float = 0.05) -> np.ndarray:
        """Wave distortion effect"""
        h, w = frame.shape[:2]
        
        # Create displacement maps
        x_map = np.zeros((h, w), dtype=np.float32)
        y_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                x_map[i, j] = j + amplitude * np.sin(2 * np.pi * frequency * i)
                y_map[i, j] = i
        
        return cv2.remap(frame, x_map, y_map, cv2.INTER_LINEAR)
    
    @staticmethod
    def fisheye(frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Fisheye lens distortion"""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Create distortion maps
        x_map = np.zeros((h, w), dtype=np.float32)
        y_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                # Distance from center
                dx = j - cx
                dy = i - cy
                dist = np.sqrt(dx**2 + dy**2)
                
                # Apply distortion
                factor = 1.0
                if dist > 0:
                    factor = np.arctan(dist * strength) / (dist * strength)
                
                x_map[i, j] = cx + dx * factor
                y_map[i, j] = cy + dy * factor
        
        return cv2.remap(frame, x_map, y_map, cv2.INTER_LINEAR)
    
    @staticmethod
    def glitch(frame: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Digital glitch effect"""
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Random horizontal shifts
        num_glitches = int(intensity * 10)
        
        for _ in range(num_glitches):
            # Random row range
            y1 = np.random.randint(0, h - 10)
            y2 = y1 + np.random.randint(5, min(20, h - y1))
            
            # Random shift amount
            shift = np.random.randint(-w // 10, w // 10)
            
            # Shift rows
            if shift > 0:
                result[y1:y2, shift:] = frame[y1:y2, :-shift]
            elif shift < 0:
                result[y1:y2, :shift] = frame[y1:y2, -shift:]
            
            # Random color channel swap
            if np.random.random() < 0.3:
                channels = list(range(3))
                np.random.shuffle(channels)
                result[y1:y2] = result[y1:y2, :, channels]
        
        # Add digital noise
        noise = np.random.randint(-int(intensity * 50), int(intensity * 50), (h, w, 3))
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def pixelate(frame: np.ndarray, pixel_size: int = 10) -> np.ndarray:
        """Pixelation effect"""
        h, w = frame.shape[:2]
        
        # Downscale
        small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        
        # Upscale with nearest neighbor
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

class ArtisticEffects:
    """Artistic and stylization effects"""
    
    @staticmethod
    def oil_painting(frame: np.ndarray, brush_size: int = 5, levels: int = 30) -> np.ndarray:
        """Oil painting effect"""
        # Quantize colors
        div = 256 // levels
        quantized = (frame // div) * div
        
        # Apply bilateral filter for smooth areas
        smooth = cv2.bilateralFilter(quantized, brush_size * 2, 75, 75)
        
        # Edge enhancement
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = 255 - edges
        
        # Combine
        edges_3ch = cv2.merge([edges, edges, edges])
        result = cv2.bitwise_and(smooth, edges_3ch)
        
        return result
    
    @staticmethod
    def cartoon(frame: np.ndarray, num_colors: int = 8) -> np.ndarray:
        """Cartoon effect"""
        # Bilateral filter for smoothing
        smooth = cv2.bilateralFilter(frame, 15, 80, 80)
        
        # K-means color quantization
        data = smooth.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reconstruct image
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(frame.shape)
        
        # Edge detection
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine
        result = cv2.bitwise_and(quantized, edges)
        
        return result
    
    @staticmethod
    def pencil_sketch(frame: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Pencil sketch effect"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Invert
        inv = 255 - gray
        
        # Gaussian blur
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        
        # Blend
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        
        # Convert back to BGR
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def watercolor(frame: np.ndarray) -> np.ndarray:
        """Watercolor painting effect"""
        # Stylization
        stylized = cv2.stylization(frame, sigma_s=60, sigma_r=0.4)
        
        # Edge preserving filter
        result = cv2.edgePreservingFilter(stylized, flags=2, sigma_s=50, sigma_r=0.4)
        
        return result

class TemporalEffects:
    """Effects that work across multiple frames"""
    
    def __init__(self):
        self.frame_buffer = []
        self.max_buffer_size = 30
    
    def echo(self, frame: np.ndarray, delay: int = 5, decay: float = 0.5) -> np.ndarray:
        """Echo/trail effect"""
        self.frame_buffer.append(frame)
        
        # Limit buffer size
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        result = frame.copy().astype(np.float32)
        
        # Add echoes
        for i in range(len(self.frame_buffer) - delay - 1, -1, -delay):
            if i >= 0:
                alpha = decay ** ((len(self.frame_buffer) - i) / delay)
                result = result * (1 - alpha) + self.frame_buffer[i].astype(np.float32) * alpha
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def time_displacement(self, frame: np.ndarray, displacement_map: np.ndarray) -> np.ndarray:
        """Time displacement using previous frames"""
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        if len(self.frame_buffer) < 2:
            return frame
        
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)
        
        # Normalize displacement map
        disp_norm = (displacement_map * (len(self.frame_buffer) - 1)).astype(int)
        
        # Sample from different time points
        for y in range(h):
            for x in range(w):
                time_index = max(0, min(len(self.frame_buffer) - 1, disp_norm[y, x]))
                result[y, x] = self.frame_buffer[time_index][y, x]
        
        return result
    
    def motion_trail(self, frame: np.ndarray, num_trails: int = 5) -> np.ndarray:
        """Motion trail effect"""
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) > num_trails:
            self.frame_buffer.pop(0)
        
        if len(self.frame_buffer) == 1:
            return frame
        
        # Blend frames with decreasing opacity
        result = np.zeros_like(frame, dtype=np.float32)
        
        for i, past_frame in enumerate(self.frame_buffer):
            alpha = (i + 1) / len(self.frame_buffer)
            result += past_frame.astype(np.float32) * alpha
        
        result /= len(self.frame_buffer)
        
        return np.clip(result, 0, 255).astype(np.uint8)

class AdvancedEffectsProcessor:
    """Main effects processor with GPU support"""
    
    def __init__(self):
        self.blur_effects = BlurEffects()
        self.color_effects = ColorEffects()
        self.distortion_effects = DistortionEffects()
        self.artistic_effects = ArtisticEffects()
        self.temporal_effects = TemporalEffects()
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            logger.info("GPU acceleration available for effects")
    
    def apply_effect(
        self,
        frame: np.ndarray,
        effect_name: str,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Apply named effect to frame"""
        # Parse effect category and name
        if "." in effect_name:
            category, name = effect_name.split(".", 1)
        else:
            category = "custom"
            name = effect_name
        
        # Route to appropriate effect
        if category == "blur":
            if name == "gaussian":
                return self.blur_effects.gaussian_blur(frame, **parameters)
            elif name == "motion":
                return self.blur_effects.motion_blur(frame, **parameters)
            elif name == "radial":
                return self.blur_effects.radial_blur(frame, **parameters)
            elif name == "lens":
                return self.blur_effects.lens_blur(frame, **parameters)
        
        elif category == "color":
            if name == "chromatic":
                return self.color_effects.chromatic_aberration(frame, **parameters)
            elif name == "duotone":
                return self.color_effects.duotone(frame, **parameters)
            elif name == "isolation":
                return self.color_effects.color_isolation(frame, **parameters)
        
        elif category == "distortion":
            if name == "wave":
                return self.distortion_effects.wave_distortion(frame, **parameters)
            elif name == "fisheye":
                return self.distortion_effects.fisheye(frame, **parameters)
            elif name == "glitch":
                return self.distortion_effects.glitch(frame, **parameters)
            elif name == "pixelate":
                return self.distortion_effects.pixelate(frame, **parameters)
        
        elif category == "artistic":
            if name == "oil":
                return self.artistic_effects.oil_painting(frame, **parameters)
            elif name == "cartoon":
                return self.artistic_effects.cartoon(frame, **parameters)
            elif name == "sketch":
                return self.artistic_effects.pencil_sketch(frame, **parameters)
            elif name == "watercolor":
                return self.artistic_effects.watercolor(frame)
        
        elif category == "temporal":
            if name == "echo":
                return self.temporal_effects.echo(frame, **parameters)
            elif name == "trail":
                return self.temporal_effects.motion_trail(frame, **parameters)
        
        logger.warning(f"Unknown effect: {effect_name}")
        return frame
    
    def chain_effects(
        self,
        frame: np.ndarray,
        effects: List[Tuple[str, Dict[str, Any]]]
    ) -> np.ndarray:
        """Apply multiple effects in sequence"""
        result = frame
        
        for effect_name, parameters in effects:
            result = self.apply_effect(result, effect_name, parameters)
        
        return result
    
    def blend_effect(
        self,
        original: np.ndarray,
        effected: np.ndarray,
        blend_mode: str = "normal",
        opacity: float = 1.0
    ) -> np.ndarray:
        """Blend effect with original using various modes"""
        if blend_mode == "normal":
            return cv2.addWeighted(original, 1 - opacity, effected, opacity, 0)
        
        elif blend_mode == "multiply":
            result = (original.astype(np.float32) * effected.astype(np.float32)) / 255.0
            result = result * opacity + original.astype(np.float32) * (1 - opacity)
            return np.clip(result, 0, 255).astype(np.uint8)
        
        elif blend_mode == "screen":
            result = 255 - ((255 - original.astype(np.float32)) * (255 - effected.astype(np.float32))) / 255.0
            result = result * opacity + original.astype(np.float32) * (1 - opacity)
            return np.clip(result, 0, 255).astype(np.uint8)
        
        elif blend_mode == "overlay":
            result = np.where(
                original < 128,
                2 * original * effected / 255.0,
                255 - 2 * (255 - original) * (255 - effected) / 255.0
            )
            result = result * opacity + original.astype(np.float32) * (1 - opacity)
            return np.clip(result, 0, 255).astype(np.uint8)
        
        else:
            return effected

# Example usage
if __name__ == "__main__":
    # Create processor
    processor = AdvancedEffectsProcessor()
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (540, 380), (255, 100, 50), -1)
    
    # Apply single effect
    blurred = processor.apply_effect(test_frame, "blur.gaussian", {"radius": 10})
    
    # Chain multiple effects
    effects_chain = [
        ("blur.gaussian", {"radius": 3}),
        ("color.chromatic", {"shift": 5}),
        ("distortion.wave", {"amplitude": 10, "frequency": 0.02})
    ]
    
    result = processor.chain_effects(test_frame, effects_chain)
    
    print("✅ Advanced effects library ready")