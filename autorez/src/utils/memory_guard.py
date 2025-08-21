"""
Memory management and OOM protection for video processing.
Implements adaptive degradation and resource monitoring.
"""

import psutil
import torch
from contextlib import contextmanager
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class MemoryGuard:
    """Protects against OOM with adaptive quality degradation."""
    
    def __init__(self, max_gb: float = 16, safety_margin_gb: float = 2):
        """
        Initialize memory guard with limits and degradation strategy.
        
        Args:
            max_gb: Maximum memory allowed in GB
            safety_margin_gb: Safety buffer to prevent hard OOM
        """
        self.max_bytes = int((max_gb - safety_margin_gb) * 1024**3)
        self.safety_margin_bytes = int(safety_margin_gb * 1024**3)
        
        # Progressive degradation levels for memory reduction
        self.degradation_levels = [
            {'fps': 1.0, 'window': 16, 'crop': 256, 'batch_size': 8},
            {'fps': 0.5, 'window': 16, 'crop': 256, 'batch_size': 4},
            {'fps': 0.5, 'window': 8, 'crop': 224, 'batch_size': 2},
            {'fps': 0.25, 'window': 8, 'crop': 224, 'batch_size': 1},
            {'fps': 0.25, 'window': 4, 'crop': 112, 'batch_size': 1},  # Emergency level
        ]
        self.current_level = 0
        self.retry_count = 0
        self.max_retries = 3
    
    @contextmanager
    def protected_execution(self, operation_name: str = "operation"):
        """
        Context manager for memory-protected execution with automatic retry.
        
        Args:
            operation_name: Name of operation for logging
        """
        try:
            # Pre-check available memory
            if not self._check_memory_available():
                self._preemptive_degrade()
            
            yield self.get_current_params()
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                logger.warning(f"OOM during {operation_name}: {e}")
                
                # Clear caches
                self._clear_memory_caches()
                
                # Try degradation
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    params = self._degrade_and_get_params()
                    logger.info(f"Retrying with degraded params: {params}")
                    # Caller should retry with new params
                    raise MemoryRetryException(params)
                else:
                    raise MemoryError(f"OOM persists after {self.max_retries} retries")
            else:
                raise
        finally:
            self.retry_count = 0
    
    def _check_memory_available(self) -> bool:
        """Check if enough memory is available."""
        available = psutil.virtual_memory().available
        return available > self.safety_margin_bytes
    
    def _preemptive_degrade(self):
        """Degrade parameters preemptively when memory is low."""
        available_gb = psutil.virtual_memory().available / 1024**3
        logger.warning(f"Low memory detected: {available_gb:.2f}GB available")
        
        if self.current_level < len(self.degradation_levels) - 1:
            self.current_level += 1
            logger.info(f"Preemptively degrading to level {self.current_level}")
    
    def _clear_memory_caches(self):
        """Clear all possible memory caches."""
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _degrade_and_get_params(self) -> Dict:
        """Degrade quality parameters and return new settings."""
        if self.current_level < len(self.degradation_levels) - 1:
            self.current_level += 1
            return self.degradation_levels[self.current_level]
        raise MemoryError("Cannot degrade further, at minimum quality")
    
    def get_current_params(self) -> Dict:
        """Get current degradation parameters."""
        return self.degradation_levels[self.current_level].copy()
    
    def reset(self):
        """Reset to highest quality settings."""
        self.current_level = 0
        self.retry_count = 0
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics."""
        vm = psutil.virtual_memory()
        stats = {
            'total_gb': vm.total / 1024**3,
            'available_gb': vm.available / 1024**3,
            'used_gb': vm.used / 1024**3,
            'percent': vm.percent,
            'current_level': self.current_level,
            'max_allowed_gb': self.max_bytes / 1024**3
        }
        
        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
        
        return stats


class MemoryRetryException(Exception):
    """Exception to signal retry with degraded parameters."""
    def __init__(self, new_params: Dict):
        self.new_params = new_params
        super().__init__(f"Retry with params: {new_params}")