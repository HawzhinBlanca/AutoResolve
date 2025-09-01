#!/usr/bin/env python3
"""
LUT (Look-Up Table) Management System
Comprehensive LUT handling with 1D, 3D support and real-time preview
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
import struct
import re
from scipy.interpolate import RegularGridInterpolator
import json
import hashlib

logger = logging.getLogger(__name__)

class LUTFormat(Enum):
    """Supported LUT formats"""
    CUBE = "cube"          # Adobe .cube format
    THREE_DL = "3dl"       # 3D LUT format
    CSP = "csp"           # Cinespace format
    ITX = "itx"           # Iridas format
    LOOK = "look"         # Lustre format
    MGA = "mga"           # Pandora format
    DAT = "dat"           # DaVinci format
    CUSTOM = "custom"      # Custom format

class LUTType(Enum):
    """LUT dimensions"""
    LUT_1D = "1D"
    LUT_3D = "3D"

@dataclass
class LUTMetadata:
    """LUT metadata information"""
    name: str
    format: LUTFormat
    lut_type: LUTType
    size: int  # Size per dimension (e.g., 33 for 33x33x33)
    input_range: Tuple[float, float]
    output_range: Tuple[float, float]
    title: Optional[str] = None
    creator: Optional[str] = None
    description: Optional[str] = None
    copyright: Optional[str] = None
    timestamp: Optional[str] = None

class LUT:
    """Base LUT class"""
    
    def __init__(self, data: np.ndarray, metadata: LUTMetadata):
        self.data = data
        self.metadata = metadata
        self._validate()
    
    def _validate(self):
        """Validate LUT data"""
        if self.metadata.lut_type == LUTType.LUT_1D:
            if len(self.data.shape) != 2 or self.data.shape[1] != 3:
                raise ValueError("1D LUT must be Nx3 array")
        elif self.metadata.lut_type == LUTType.LUT_3D:
            if len(self.data.shape) != 4 or self.data.shape[3] != 3:
                raise ValueError("3D LUT must be NxNxNx3 array")
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply LUT to image"""
        if self.metadata.lut_type == LUTType.LUT_1D:
            return self._apply_1d(image)
        else:
            return self._apply_3d(image)
    
    def _apply_1d(self, image: np.ndarray) -> np.ndarray:
        """Apply 1D LUT"""
        result = np.zeros_like(image)
        
        for c in range(3):
            # Create interpolation function
            x = np.linspace(0, 255, len(self.data))
            y = self.data[:, c] * 255
            
            # Apply to each channel
            result[:, :, c] = np.interp(image[:, :, c], x, y)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_3d(self, image: np.ndarray) -> np.ndarray:
        """Apply 3D LUT with trilinear interpolation"""
        h, w = image.shape[:2]
        
        # Normalize input to LUT range
        img_norm = image.astype(np.float32) / 255.0
        
        # Create interpolator
        size = self.metadata.size
        x = np.linspace(0, 1, size)
        interpolator = RegularGridInterpolator(
            (x, x, x),
            self.data,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        # Reshape for interpolation
        img_flat = img_norm.reshape(-1, 3)
        
        # Swap BGR to RGB for interpolation
        img_flat = img_flat[:, [2, 1, 0]]
        
        # Interpolate
        result_flat = interpolator(img_flat)
        
        # Swap back to BGR
        result_flat = result_flat[:, [2, 1, 0]]
        
        # Reshape and denormalize
        result = result_flat.reshape(h, w, 3)
        result = (result * 255).astype(np.uint8)
        
        return result

class CubeLUTParser:
    """Parser for .cube LUT files"""
    
    @staticmethod
    def parse(file_path: str) -> LUT:
        """Parse .cube file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        metadata = LUTMetadata(
            name=Path(file_path).stem,
            format=LUTFormat.CUBE,
            lut_type=LUTType.LUT_3D,
            size=0,
            input_range=(0.0, 1.0),
            output_range=(0.0, 1.0)
        )
        
        data_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse metadata
            if line.startswith('TITLE'):
                metadata.title = line.split('"')[1] if '"' in line else line.split()[1]
            elif line.startswith('LUT_3D_SIZE'):
                metadata.size = int(line.split()[1])
            elif line.startswith('LUT_1D_SIZE'):
                metadata.lut_type = LUTType.LUT_1D
                metadata.size = int(line.split()[1])
            elif line.startswith('DOMAIN_MIN'):
                values = line.split()[1:]
                metadata.input_range = (float(values[0]), metadata.input_range[1])
            elif line.startswith('DOMAIN_MAX'):
                values = line.split()[1:]
                metadata.input_range = (metadata.input_range[0], float(values[0]))
            else:
                # Data line
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == 3:
                        data_lines.append(values)
                except ValueError:
                    continue
        
        # Convert to numpy array
        if metadata.lut_type == LUTType.LUT_3D:
            size = metadata.size
            data = np.array(data_lines).reshape(size, size, size, 3)
        else:
            data = np.array(data_lines)
        
        return LUT(data, metadata)

class ThreeDLParser:
    """Parser for .3dl LUT files"""
    
    @staticmethod
    def parse(file_path: str) -> LUT:
        """Parse .3dl file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Detect mesh size
        mesh_line = next((l for l in lines if l.strip().isdigit()), None)
        if not mesh_line:
            raise ValueError("Invalid 3DL file: no mesh size found")
        
        size = int(mesh_line.strip())
        
        metadata = LUTMetadata(
            name=Path(file_path).stem,
            format=LUTFormat.THREE_DL,
            lut_type=LUTType.LUT_3D,
            size=size,
            input_range=(0, 1023),  # 10-bit by default
            output_range=(0, 1023)
        )
        
        # Parse data
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.isdigit():
                try:
                    values = [int(x) for x in line.split()]
                    if len(values) == 3:
                        # Normalize to 0-1 range
                        normalized = [v / 1023.0 for v in values]
                        data_lines.append(normalized)
                except ValueError:
                    continue
        
        # Reshape data
        data = np.array(data_lines).reshape(size, size, size, 3)
        
        return LUT(data, metadata)

class LUTManager:
    """Manages LUT collection and application"""
    
    def __init__(self, lut_dir: str):
        self.lut_dir = Path(lut_dir)
        self.lut_dir.mkdir(exist_ok=True)
        
        self.luts: Dict[str, LUT] = {}
        self.categories: Dict[str, List[str]] = {
            "Film": [],
            "Creative": [],
            "Technical": [],
            "Camera": [],
            "Custom": []
        }
        
        self.parsers = {
            ".cube": CubeLUTParser(),
            ".3dl": ThreeDLParser()
        }
        
        # Load built-in LUTs
        self._load_builtin_luts()
        
        # Scan directory for LUTs
        self.scan_directory()
    
    def _load_builtin_luts(self):
        """Load built-in LUTs"""
        # Create some basic LUTs programmatically
        
        # S-Curve contrast LUT
        self._create_contrast_lut("Contrast_S_Curve", 1.5)
        
        # Bleach bypass LUT
        self._create_bleach_bypass_lut("Bleach_Bypass")
        
        # Cross processing LUT
        self._create_cross_process_lut("Cross_Process")
        
        # Film emulation LUTs
        self._create_film_lut("Kodak_2383", warm=True, contrast=1.2)
        self._create_film_lut("Fuji_3510", warm=False, contrast=1.1)
    
    def _create_contrast_lut(self, name: str, contrast: float):
        """Create contrast adjustment LUT"""
        size = 256
        data = np.zeros((size, 3))
        
        for i in range(size):
            val = i / 255.0
            # S-curve formula
            adjusted = 1 / (1 + np.exp(-contrast * (val - 0.5)))
            data[i] = [adjusted, adjusted, adjusted]
        
        metadata = LUTMetadata(
            name=name,
            format=LUTFormat.CUSTOM,
            lut_type=LUTType.LUT_1D,
            size=size,
            input_range=(0, 1),
            output_range=(0, 1),
            description="S-curve contrast enhancement"
        )
        
        lut = LUT(data, metadata)
        self.luts[name] = lut
        self.categories["Technical"].append(name)
    
    def _create_bleach_bypass_lut(self, name: str):
        """Create bleach bypass effect LUT"""
        size = 33
        data = np.zeros((size, size, size, 3))
        
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    # Normalize to 0-1
                    rgb = np.array([r, g, b]) / (size - 1)
                    
                    # Desaturate
                    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    
                    # Mix with original
                    result = rgb * 0.5 + np.array([gray, gray, gray]) * 0.5
                    
                    # Increase contrast
                    result = np.clip(result * 1.2 - 0.1, 0, 1)
                    
                    data[r, g, b] = result
        
        metadata = LUTMetadata(
            name=name,
            format=LUTFormat.CUSTOM,
            lut_type=LUTType.LUT_3D,
            size=size,
            input_range=(0, 1),
            output_range=(0, 1),
            description="Bleach bypass film effect"
        )
        
        lut = LUT(data, metadata)
        self.luts[name] = lut
        self.categories["Film"].append(name)
    
    def _create_cross_process_lut(self, name: str):
        """Create cross processing effect LUT"""
        size = 33
        data = np.zeros((size, size, size, 3))
        
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    rgb = np.array([r, g, b]) / (size - 1)
                    
                    # Shift color channels
                    result = np.zeros(3)
                    result[0] = np.clip(rgb[0] * 1.2 - 0.1, 0, 1)  # Push reds
                    result[1] = np.clip(rgb[1] * 0.9, 0, 1)        # Pull greens
                    result[2] = np.clip(rgb[2] * 1.3 - 0.15, 0, 1) # Push blues
                    
                    # S-curve on each channel
                    for c in range(3):
                        result[c] = 1 / (1 + np.exp(-5 * (result[c] - 0.5)))
                    
                    data[r, g, b] = result
        
        metadata = LUTMetadata(
            name=name,
            format=LUTFormat.CUSTOM,
            lut_type=LUTType.LUT_3D,
            size=size,
            input_range=(0, 1),
            output_range=(0, 1),
            description="Cross processing effect"
        )
        
        lut = LUT(data, metadata)
        self.luts[name] = lut
        self.categories["Creative"].append(name)
    
    def _create_film_lut(self, name: str, warm: bool, contrast: float):
        """Create film emulation LUT"""
        size = 33
        data = np.zeros((size, size, size, 3))
        
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    rgb = np.array([r, g, b]) / (size - 1)
                    
                    # Apply film characteristics
                    result = rgb.copy()
                    
                    # Adjust contrast
                    result = np.clip((result - 0.5) * contrast + 0.5, 0, 1)
                    
                    # Color shift
                    if warm:
                        result[0] *= 1.05  # Warm: boost reds
                        result[2] *= 0.95  # Reduce blues
                    else:
                        result[0] *= 0.95  # Cool: reduce reds
                        result[2] *= 1.05  # Boost blues
                    
                    # Film curve
                    for c in range(3):
                        # Lifted blacks, compressed highlights
                        result[c] = result[c] * 0.9 + 0.05
                        result[c] = 1 - np.exp(-2 * result[c])
                    
                    data[r, g, b] = np.clip(result, 0, 1)
        
        metadata = LUTMetadata(
            name=name,
            format=LUTFormat.CUSTOM,
            lut_type=LUTType.LUT_3D,
            size=size,
            input_range=(0, 1),
            output_range=(0, 1),
            description=f"Film emulation - {name}"
        )
        
        lut = LUT(data, metadata)
        self.luts[name] = lut
        self.categories["Film"].append(name)
    
    def scan_directory(self):
        """Scan directory for LUT files"""
        for ext, parser in self.parsers.items():
            for lut_file in self.lut_dir.glob(f"*{ext}"):
                try:
                    lut = parser.parse(str(lut_file))
                    self.luts[lut.metadata.name] = lut
                    
                    # Auto-categorize
                    self._categorize_lut(lut.metadata.name)
                    
                    logger.info(f"Loaded LUT: {lut.metadata.name}")
                except Exception as e:
                    logger.error(f"Failed to load LUT {lut_file}: {e}")
    
    def _categorize_lut(self, name: str):
        """Auto-categorize LUT based on name"""
        name_lower = name.lower()
        
        if any(film in name_lower for film in ["kodak", "fuji", "film", "cineon"]):
            self.categories["Film"].append(name)
        elif any(cam in name_lower for cam in ["log", "slog", "clog", "vlog", "raw"]):
            self.categories["Camera"].append(name)
        elif any(tech in name_lower for tech in ["709", "2020", "linear", "gamma"]):
            self.categories["Technical"].append(name)
        else:
            self.categories["Creative"].append(name)
    
    def apply_lut(self, image: np.ndarray, lut_name: str, intensity: float = 1.0) -> np.ndarray:
        """Apply LUT to image with intensity control"""
        if lut_name not in self.luts:
            logger.error(f"LUT not found: {lut_name}")
            return image
        
        lut = self.luts[lut_name]
        
        # Apply LUT
        result = lut.apply(image)
        
        # Blend with original based on intensity
        if intensity < 1.0:
            result = cv2.addWeighted(image, 1 - intensity, result, intensity, 0)
        
        return result
    
    def chain_luts(self, image: np.ndarray, lut_names: List[str]) -> np.ndarray:
        """Apply multiple LUTs in sequence"""
        result = image
        
        for lut_name in lut_names:
            if lut_name in self.luts:
                result = self.apply_lut(result, lut_name)
        
        return result
    
    def create_lut_from_reference(
        self,
        reference_before: np.ndarray,
        reference_after: np.ndarray,
        name: str,
        size: int = 33
    ) -> LUT:
        """Create LUT from reference images"""
        # Sample color points from before image
        h, w = reference_before.shape[:2]
        
        # Create 3D LUT
        data = np.zeros((size, size, size, 3))
        
        # Sample colors uniformly
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    # Target color
                    target = np.array([b, g, r]) / (size - 1) * 255
                    
                    # Find closest match in before image
                    diff = np.sum((reference_before - target) ** 2, axis=2)
                    min_idx = np.unravel_index(np.argmin(diff), diff.shape)
                    
                    # Get corresponding color from after image
                    result_color = reference_after[min_idx] / 255.0
                    
                    data[r, g, b] = result_color[[2, 1, 0]]  # BGR to RGB
        
        metadata = LUTMetadata(
            name=name,
            format=LUTFormat.CUSTOM,
            lut_type=LUTType.LUT_3D,
            size=size,
            input_range=(0, 1),
            output_range=(0, 1),
            description="Created from reference images"
        )
        
        lut = LUT(data, metadata)
        self.luts[name] = lut
        self.categories["Custom"].append(name)
        
        return lut
    
    def export_lut(self, lut_name: str, output_path: str, format: LUTFormat = LUTFormat.CUBE):
        """Export LUT to file"""
        if lut_name not in self.luts:
            logger.error(f"LUT not found: {lut_name}")
            return
        
        lut = self.luts[lut_name]
        
        if format == LUTFormat.CUBE:
            self._export_cube(lut, output_path)
        elif format == LUTFormat.THREE_DL:
            self._export_3dl(lut, output_path)
        else:
            logger.error(f"Export format not supported: {format}")
    
    def _export_cube(self, lut: LUT, output_path: str):
        """Export LUT in .cube format"""
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"# Created by AutoResolve LUT Manager\n")
            f.write(f"TITLE \"{lut.metadata.name}\"\n")
            
            if lut.metadata.lut_type == LUTType.LUT_3D:
                f.write(f"LUT_3D_SIZE {lut.metadata.size}\n")
                
                # Write data
                size = lut.metadata.size
                for r in range(size):
                    for g in range(size):
                        for b in range(size):
                            values = lut.data[r, g, b]
                            f.write(f"{values[0]:.6f} {values[1]:.6f} {values[2]:.6f}\n")
            else:
                f.write(f"LUT_1D_SIZE {lut.metadata.size}\n")
                
                # Write data
                for i in range(len(lut.data)):
                    values = lut.data[i]
                    f.write(f"{values[0]:.6f} {values[1]:.6f} {values[2]:.6f}\n")
        
        logger.info(f"Exported LUT to: {output_path}")
    
    def get_lut_preview(self, lut_name: str, sample_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate LUT preview image"""
        if sample_image is None:
            # Create color gradient
            sample_image = self._create_gradient_image()
        
        if lut_name in self.luts:
            return self.apply_lut(sample_image, lut_name)
        
        return sample_image
    
    def _create_gradient_image(self, width: int = 512, height: int = 256) -> np.ndarray:
        """Create gradient test image"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Horizontal gradient
        for x in range(width):
            value = int(x * 255 / width)
            image[:height//3, x] = [value, value, value]
        
        # Color bars
        colors = [
            [255, 0, 0], [255, 255, 0], [0, 255, 0],
            [0, 255, 255], [0, 0, 255], [255, 0, 255]
        ]
        
        bar_width = width // len(colors)
        for i, color in enumerate(colors):
            x1 = i * bar_width
            x2 = (i + 1) * bar_width
            image[height//3:2*height//3, x1:x2] = color
        
        # Skin tones
        skin_tones = [
            [255, 224, 189], [255, 205, 148], [234, 192, 134],
            [255, 173, 96], [165, 126, 110], [143, 86, 59]
        ]
        
        bar_width = width // len(skin_tones)
        for i, color in enumerate(skin_tones):
            x1 = i * bar_width
            x2 = (i + 1) * bar_width
            image[2*height//3:, x1:x2] = color[::-1]  # RGB to BGR
        
        return image

# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = LUTManager("/tmp/luts")
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (540, 380), (100, 150, 200), -1)
    
    # Apply LUT
    result = manager.apply_lut(test_image, "Bleach_Bypass", intensity=0.8)
    
    # Chain multiple LUTs
    lut_chain = ["Contrast_S_Curve", "Cross_Process"]
    chained = manager.chain_luts(test_image, lut_chain)
    
    # Export LUT
    manager.export_lut("Film_Kodak_2383", "/tmp/kodak.cube")
    
    print(f"✅ LUT Manager ready with {len(manager.luts)} LUTs loaded")