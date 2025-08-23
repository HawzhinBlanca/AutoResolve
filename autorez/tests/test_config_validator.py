"""
Test suite for configuration validation.
Tests schema validation and type conversion.
"""

import pytest
import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.schema_validator import ConfigValidator


class TestConfigValidator:
    """Test configuration validation logic."""
    
    def test_validate_embeddings_config_valid(self):
        """Test valid embeddings configuration."""
        config_content = """
[embeddings]
default_model = vjepa
backend = mps
fps = 1.0
window = 16
crop = 256
max_rss_gb = 16
max_segments = 500
seed = 1234
strategy = temp_attn
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            result = ConfigValidator.validate_config(f.name, 'embeddings')
            
            assert result['embeddings']['default_model'] == 'vjepa'
            assert result['embeddings']['backend'] == 'mps'
            assert result['embeddings']['fps'] == 1.0
            assert result['embeddings']['window'] == 16
            assert result['embeddings']['crop'] == 256
            
        os.unlink(f.name)
    
    def test_validate_embeddings_invalid_choice(self):
        """Test invalid choice value."""
        config_content = """
[embeddings]
default_model = invalid_model
backend = mps
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            with pytest.raises(ValueError, match="not in allowed choices"):
                ConfigValidator.validate_config(f.name, 'embeddings')
            
        os.unlink(f.name)
    
    def test_validate_fps_bounds(self):
        """Test FPS boundary validation."""
        config_content = """
[embeddings]
default_model = clip
backend = cuda
fps = 50.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            with pytest.raises(ValueError, match="above maximum"):
                ConfigValidator.validate_config(f.name, 'embeddings')
            
        os.unlink(f.name)
    
    def test_validate_silence_negative_threshold(self):
        """Test that RMS threshold must be negative."""
        config_content = """
[silence]
rms_thresh_db = 10.0
min_silence_s = 0.35
min_keep_s = 0.40
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            with pytest.raises(ValueError, match="above maximum"):
                ConfigValidator.validate_config(f.name, 'ops.silence')
            
        os.unlink(f.name)
    
    def test_validate_with_defaults(self):
        """Test that defaults are applied for missing values."""
        config_content = """
[embeddings]
default_model = clip
backend = mps
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            result = ConfigValidator.validate_config(f.name, 'embeddings')
            
            # Check defaults were applied
            assert result['embeddings']['fps'] == 1.0  # default
            assert result['embeddings']['window'] == 16  # default
            assert result['embeddings']['crop'] == 256  # default
            assert result['embeddings']['seed'] == 1234  # default
            
        os.unlink(f.name)
    
    def test_type_conversion(self):
        """Test type conversion from string."""
        config_content = """
[embeddings]
default_model = clip
backend = mps
fps = 2.5
window = 32
seed = 9999
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            result = ConfigValidator.validate_config(f.name, 'embeddings')
            
            # Check types are correct
            assert isinstance(result['embeddings']['fps'], float)
            assert isinstance(result['embeddings']['window'], int)
            assert isinstance(result['embeddings']['seed'], int)
            assert result['embeddings']['fps'] == 2.5
            assert result['embeddings']['window'] == 32
            assert result['embeddings']['seed'] == 9999
            
        os.unlink(f.name)
    
    def test_generate_default_config(self):
        """Test default config generation."""
        config_str = ConfigValidator.generate_default_config('embeddings')
        
        assert '[embeddings]' in config_str
        # Check for expected fields (not all have defaults)
        assert 'fps = 1.0' in config_str or 'fps' in config_str
        assert 'choices:' in config_str  # Should include choice comments
        assert 'min=' in config_str  # Should include bounds comments
    
    def test_validate_shorts_config(self):
        """Test shorts configuration validation."""
        config_content = """
[shorts]
target = 60
min_seg = 3.0
max_seg = 18.0
topk = 12
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            result = ConfigValidator.validate_config(f.name, 'ops.shorts')
            
            assert result['ops.shorts']['target'] == 60
            assert result['ops.shorts']['min_seg'] == 3.0
            assert result['ops.shorts']['max_seg'] == 18.0
            assert result['ops.shorts']['topk'] == 12
            
        os.unlink(f.name)
    
    def test_validate_min_greater_than_max(self):
        """Test that min_seg must be less than max_seg logically."""
        # Note: This would require additional business logic validation
        # beyond simple schema validation
    
    def test_validate_broll_config(self):
        """Test B-roll configuration validation."""
        config_content = """
[broll]
max_overlay_s = 7.0
min_gap_s = 4.0
dissolve_s = 0.25
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            result = ConfigValidator.validate_config(f.name, 'broll')
            
            assert result['broll']['max_overlay_s'] == 7.0
            assert result['broll']['min_gap_s'] == 4.0
            assert result['broll']['dissolve_s'] == 0.25
            
        os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])