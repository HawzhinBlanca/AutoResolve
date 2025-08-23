"""
Adaptive memory management system for AutoResolve
Ensures compliance with 16GB memory limit
"""

import gc
import os
import psutil
import logging
from typing import Callable, Any, Dict
from functools import wraps
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Centralized memory management with adaptive degradation.
    Monitors memory usage and triggers degradation strategies.
    """
    
    # Degradation thresholds (GB)
    WARNING_THRESHOLD = 12.0
    CRITICAL_THRESHOLD = 14.0
    MAX_THRESHOLD = 15.5
    
    # Degradation levels
    LEVEL_NORMAL = 0
    LEVEL_WARNING = 1
    LEVEL_CRITICAL = 2
    LEVEL_EMERGENCY = 3
    
    def __init__(self):
        self.process = psutil.Process()
        self.degradation_level = self.LEVEL_NORMAL
        self.strategies_applied = []
        self.peak_memory_gb = 0.0
        
        # Cache management
        self._cache_dirs = [
            Path.home() / ".cache" / "torch",
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".cache" / "whisper",
        ]
        
    def get_memory_gb(self) -> float:
        """Get current RSS memory in GB"""
        return self.process.memory_info().rss / (1024**3)
    
    def get_available_gb(self) -> float:
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024**3)
    
    def check_memory(self) -> int:
        """Check memory and return degradation level"""
        current_gb = self.get_memory_gb()
        self.peak_memory_gb = max(self.peak_memory_gb, current_gb)
        
        if current_gb >= self.MAX_THRESHOLD:
            return self.LEVEL_EMERGENCY
        elif current_gb >= self.CRITICAL_THRESHOLD:
            return self.LEVEL_CRITICAL
        elif current_gb >= self.WARNING_THRESHOLD:
            return self.LEVEL_WARNING
        else:
            return self.LEVEL_NORMAL
    
    def apply_degradation(self, level: int) -> bool:
        """Apply degradation strategies based on level"""
        if level <= self.degradation_level:
            return False  # Already at this level
        
        self.degradation_level = level
        logger.warning(f"Applying degradation level {level} at {self.get_memory_gb():.2f}GB")
        
        if level >= self.LEVEL_WARNING:
            self._apply_warning_strategies()
        
        if level >= self.LEVEL_CRITICAL:
            self._apply_critical_strategies()
        
        if level >= self.LEVEL_EMERGENCY:
            self._apply_emergency_strategies()
        
        return True
    
    def _apply_warning_strategies(self):
        """Level 1: Conservative memory management"""
        logger.info("WARNING: Applying conservative memory strategies")
        
        # 1. Force garbage collection
        gc.collect()
        
        # 2. Clear Python caches
        import sys
        sys.intern.clear() if hasattr(sys.intern, 'clear') else None
        
        # 3. Reduce batch sizes
        os.environ["AUTORESOLVE_BATCH_SIZE"] = "small"
        
        self.strategies_applied.append("warning_gc")
        
    def _apply_critical_strategies(self):
        """Level 2: Aggressive memory reduction"""
        logger.warning("CRITICAL: Applying aggressive memory strategies")
        
        # 1. Clear ML framework caches
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
        
        # 2. Reduce model precision
        os.environ["AUTORESOLVE_PRECISION"] = "fp16"
        
        # 3. Disable caching
        os.environ["AUTORESOLVE_CACHE"] = "disabled"
        
        # 4. Clear disk caches (careful!)
        for cache_dir in self._cache_dirs:
            if cache_dir.exists():
                # Only clear old files (>1 day)
                import time
                now = time.time()
                for file in cache_dir.rglob("*"):
                    if file.is_file():
                        age = now - file.stat().st_mtime
                        if age > 86400:  # 1 day
                            try:
                                file.unlink()
                            except:
                                pass
        
        self.strategies_applied.append("critical_cache_clear")
        
    def _apply_emergency_strategies(self):
        """Level 3: Emergency measures to prevent OOM"""
        logger.error("EMERGENCY: Applying emergency memory strategies")
        
        # 1. Kill non-essential processes
        os.environ["AUTORESOLVE_MODE"] = "minimal"
        
        # 2. Swap to disk processing
        os.environ["AUTORESOLVE_PROCESSING"] = "disk"
        
        # 3. Reduce all dimensions
        os.environ["AUTORESOLVE_DIMENSIONS"] = "minimal"
        
        # 4. Force immediate cleanup
        gc.collect(2)  # Full collection
        
        self.strategies_applied.append("emergency_minimal")
    
    def cleanup(self):
        """Perform memory cleanup"""
        before_gb = self.get_memory_gb()
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except:
            pass
        
        after_gb = self.get_memory_gb()
        freed_gb = before_gb - after_gb
        
        if freed_gb > 0.1:
            logger.info(f"Freed {freed_gb:.2f}GB of memory")
        
        return freed_gb
    
    def get_status(self) -> Dict:
        """Get current memory status"""
        return {
            "current_gb": self.get_memory_gb(),
            "available_gb": self.get_available_gb(),
            "peak_gb": self.peak_memory_gb,
            "degradation_level": self.degradation_level,
            "strategies_applied": self.strategies_applied,
            "level_name": ["normal", "warning", "critical", "emergency"][self.degradation_level]
        }

# Global instance
_memory_manager = MemoryManager()

def memory_guard(threshold_gb: float = 14.0):
    """
    Decorator to guard functions with memory checks.
    Automatically applies degradation if needed.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check memory before
            level = _memory_manager.check_memory()
            if level > _memory_manager.LEVEL_NORMAL:
                _memory_manager.apply_degradation(level)
            
            # Check if we should proceed
            if _memory_manager.get_memory_gb() > threshold_gb:
                logger.error(f"Memory {_memory_manager.get_memory_gb():.2f}GB exceeds threshold {threshold_gb}GB")
                # Try cleanup
                _memory_manager.cleanup()
                
                # Check again
                if _memory_manager.get_memory_gb() > threshold_gb:
                    raise MemoryError(f"Cannot proceed: memory exceeds {threshold_gb}GB")
            
            try:
                # Run function
                result = func(*args, **kwargs)
                
                # Cleanup after if in degraded mode
                if _memory_manager.degradation_level > _memory_manager.LEVEL_NORMAL:
                    _memory_manager.cleanup()
                
                return result
                
            except Exception as e:
                # Emergency cleanup on error
                _memory_manager.cleanup()
                raise e
        
        return wrapper
    return decorator

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    return _memory_manager

def cleanup_memory():
    """Convenience function for memory cleanup"""
    return _memory_manager.cleanup()

def get_memory_status() -> Dict:
    """Get current memory status"""
    return _memory_manager.get_status()

# Adaptive processing functions based on memory
def get_batch_size() -> int:
    """Get adaptive batch size based on memory"""
    level = _memory_manager.degradation_level
    if level >= MemoryManager.LEVEL_CRITICAL:
        return 1
    elif level >= MemoryManager.LEVEL_WARNING:
        return 4
    else:
        return 16

def get_model_precision() -> str:
    """Get model precision based on memory"""
    level = _memory_manager.degradation_level
    if level >= MemoryManager.LEVEL_CRITICAL:
        return "int8"
    elif level >= MemoryManager.LEVEL_WARNING:
        return "fp16"
    else:
        return "fp32"

def get_cache_enabled() -> bool:
    """Check if caching should be enabled"""
    return _memory_manager.degradation_level < MemoryManager.LEVEL_CRITICAL

def get_frame_size() -> tuple:
    """Get adaptive frame size for processing"""
    level = _memory_manager.degradation_level
    if level >= MemoryManager.LEVEL_EMERGENCY:
        return (160, 120)
    elif level >= MemoryManager.LEVEL_CRITICAL:
        return (320, 240)
    elif level >= MemoryManager.LEVEL_WARNING:
        return (480, 360)
    else:
        return (640, 480)