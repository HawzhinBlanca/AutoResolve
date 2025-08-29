"""
Memory management utilities for AutoResolve.
Prevents unbounded memory growth and manages resource constraints.
"""

import gc
import logging
import psutil
from typing import Optional

logger = logging.getLogger(__name__)

class MemoryError(Exception):
    """Raised when memory limits are exceeded."""
    pass

class MemoryManager:
    """Manages memory usage and enforces limits."""
    
    def __init__(self, max_memory_mb: int = 4000):
        """
        Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.warning_threshold = 0.8  # Warn at 80% usage
        self._initial_memory = self.get_current_memory_mb()
        logger.info(f"MemoryManager initialized with {max_memory_mb}MB limit")
    
    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def check_memory_usage(self) -> dict:
        """
        Check current memory usage against limits.
        
        Returns:
            Dict with memory statistics
        """
        current_mb = self.get_current_memory_mb()
        available_mb = self.get_available_memory_mb()
        usage_ratio = current_mb / self.max_memory_mb
        
        stats = {
            "current_mb": current_mb,
            "max_mb": self.max_memory_mb,
            "available_mb": available_mb,
            "usage_percent": usage_ratio * 100,
            "status": "ok"
        }
        
        if usage_ratio > 1.0:
            stats["status"] = "exceeded"
            logger.error(f"Memory limit exceeded: {current_mb:.1f}MB / {self.max_memory_mb}MB")
        elif usage_ratio > self.warning_threshold:
            stats["status"] = "warning"
            logger.warning(f"High memory usage: {current_mb:.1f}MB / {self.max_memory_mb}MB")
        
        return stats
    
    def enforce_limit(self):
        """
        Enforce memory limit, raising exception if exceeded.
        
        Raises:
            MemoryError: If memory limit is exceeded
        """
        stats = self.check_memory_usage()
        if stats["status"] == "exceeded":
            # Try to free memory first
            self.free_memory()
            
            # Check again
            stats = self.check_memory_usage()
            if stats["status"] == "exceeded":
                raise MemoryError(
                    f"Memory limit exceeded: {stats['current_mb']:.1f}MB / {self.max_memory_mb}MB"
                )
    
    def free_memory(self):
        """Attempt to free memory by running garbage collection."""
        logger.info("Running garbage collection to free memory")
        gc.collect()
        
        # For Python 3.9+, also run full collection
        if hasattr(gc, 'freeze'):
            gc.freeze()
            gc.collect()
            gc.unfreeze()

# Global instance
_memory_manager: Optional[MemoryManager] = None

def get_memory_manager(max_mb: int = 4000) -> MemoryManager:
    """Get or create global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(max_mb)
    return _memory_manager

def check_memory() -> dict:
    """Quick memory check."""
    return get_memory_manager().check_memory_usage()

def enforce_memory_limit():
    """Enforce global memory limit."""
    get_memory_manager().enforce_limit()
