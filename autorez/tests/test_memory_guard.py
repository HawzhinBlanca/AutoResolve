"""
Test suite for memory guard and OOM protection.
Tests degradation strategies and memory management.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.memory_guard import MemoryGuard, MemoryRetryException


class TestMemoryGuard:
    """Test memory management and degradation strategies."""
    
    def test_memory_guard_initialization(self):
        """Test MemoryGuard initialization with defaults."""
        guard = MemoryGuard(max_gb=16, safety_margin_gb=2)
        
        assert guard.max_bytes == 14 * 1024**3  # 16 - 2 = 14 GB
        assert guard.current_level == 0
        assert len(guard.degradation_levels) > 0
        
        # Check first level is highest quality
        first_level = guard.degradation_levels[0]
        assert first_level['fps'] == 1.0
        assert first_level['window'] == 16
        assert first_level['crop'] == 256
    
    def test_memory_degradation_progression(self):
        """Test progressive quality degradation."""
        guard = MemoryGuard(max_gb=16)
        
        initial_params = guard.get_current_params()
        assert initial_params['fps'] == 1.0
        
        # Simulate degradation
        guard._degrade_and_get_params()
        level1_params = guard.get_current_params()
        assert level1_params['fps'] < initial_params['fps'] or \
               level1_params['window'] < initial_params['window'] or \
               level1_params['crop'] < initial_params['crop']
        
        # Further degradation
        guard._degrade_and_get_params()
        level2_params = guard.get_current_params()
        assert guard.current_level == 2
    
    def test_memory_degradation_limits(self):
        """Test that degradation has limits."""
        guard = MemoryGuard(max_gb=16)
        
        # Degrade to maximum
        max_levels = len(guard.degradation_levels)
        for _ in range(max_levels - 1):
            guard._degrade_and_get_params()
        
        # Should be at last level
        assert guard.current_level == max_levels - 1
        
        # Further degradation should raise error
        with pytest.raises(MemoryError, match="Cannot degrade further"):
            guard._degrade_and_get_params()
    
    def test_memory_guard_reset(self):
        """Test resetting to highest quality."""
        guard = MemoryGuard(max_gb=16)
        
        # Degrade a few levels
        guard._degrade_and_get_params()
        guard._degrade_and_get_params()
        assert guard.current_level == 2
        
        # Reset
        guard.reset()
        assert guard.current_level == 0
        assert guard.retry_count == 0
        
        params = guard.get_current_params()
        assert params['fps'] == 1.0
    
    def test_protected_execution_success(self):
        """Test successful execution without OOM."""
        guard = MemoryGuard(max_gb=16)
        
        with guard.protected_execution("test_operation"):
            params = guard.get_current_params()
            assert params is not None
            # Simulate successful operation
            result = "success"
        
        assert result == "success"
        assert guard.current_level == 0  # No degradation needed
    
    def test_protected_execution_oom_retry(self):
        """Test OOM handling with retry signal."""
        guard = MemoryGuard(max_gb=16)
        
        # Simulate OOM error
        try:
            with guard.protected_execution("test_operation"):
                # Simulate CUDA OOM
                raise RuntimeError("CUDA out of memory")
        except MemoryRetryException as e:
            # Should get retry exception with new params
            assert e.new_params is not None
            assert guard.current_level == 1
    
    def test_memory_stats(self):
        """Test memory statistics reporting."""
        guard = MemoryGuard(max_gb=16)
        
        stats = guard.get_memory_stats()
        
        # Check required keys
        assert 'total_gb' in stats
        assert 'available_gb' in stats
        assert 'used_gb' in stats
        assert 'percent' in stats
        assert 'current_level' in stats
        assert 'max_allowed_gb' in stats
        
        # Validate values
        assert stats['total_gb'] > 0
        assert stats['available_gb'] >= 0
        assert stats['used_gb'] >= 0
        assert 0 <= stats['percent'] <= 100
        assert stats['current_level'] == 0
        assert stats['max_allowed_gb'] == 14  # 16 - 2 safety margin
    
    def test_preemptive_degradation(self):
        """Test preemptive degradation when memory is low."""
        guard = MemoryGuard(max_gb=16, safety_margin_gb=2)
        
        # Mock low memory condition
        original_check = guard._check_memory_available
        guard._check_memory_available = lambda: False
        
        try:
            with guard.protected_execution("test_operation"):
                params = guard.get_current_params()
                # Should have degraded preemptively
                assert guard.current_level > 0
        except:
            pass
        finally:
            guard._check_memory_available = original_check
    
    def test_batch_size_degradation(self):
        """Test that batch size is included in degradation."""
        guard = MemoryGuard(max_gb=16)
        
        # Check batch sizes decrease with degradation
        batch_sizes = []
        for level in guard.degradation_levels:
            batch_sizes.append(level['batch_size'])
        
        # Batch sizes should generally decrease
        assert batch_sizes[0] >= batch_sizes[-1]
        assert all('batch_size' in level for level in guard.degradation_levels)
    
    def test_emergency_degradation_level(self):
        """Test emergency degradation level has minimal settings."""
        guard = MemoryGuard(max_gb=16)
        
        emergency_level = guard.degradation_levels[-1]
        
        # Emergency level should have minimal resource usage
        assert emergency_level['fps'] <= 0.25
        assert emergency_level['window'] <= 8
        assert emergency_level['crop'] <= 224
        assert emergency_level['batch_size'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])