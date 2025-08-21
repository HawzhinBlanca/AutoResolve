#!/usr/bin/env python
"""
Integration test for the full pipeline with all bug fixes.
Tests that all components work together correctly.
"""

import sys
import os
import tempfile
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.promotion import promote_vjepa
from src.utils.memory_guard import MemoryGuard
from src.utils.memory import set_seeds
from src.scoring.score_normalizer import ScoreNormalizer
from src.validators.segment_validator import SegmentValidator
from src.validators.duration_validator import DurationValidator
from src.config.schema_validator import ConfigValidator
from src.schemas.output_schemas import TRANSCRIPT_SCHEMA


def test_promotion_logic():
    """Test promotion logic integration."""
    print("Testing promotion logic...")
    
    # Test with realistic data
    results = {
        "top3": {
            "vjepa": 0.75,
            "clip": 0.60,
            "vjepa_ci": [0.70, 0.80],
            "clip_ci": [0.55, 0.65]
        },
        "mrr": {
            "vjepa": 0.68,
            "clip": 0.55,
            "vjepa_ci": [0.63, 0.73],
            "clip_ci": [0.50, 0.60]
        }
    }
    
    decision = promote_vjepa(results, 4.5)
    assert isinstance(decision, bool)
    print(f"  Promotion decision: {'PROMOTE' if decision else 'KEEP CLIP'}")
    print("  ✓ Promotion logic working")


def test_memory_management():
    """Test memory management integration."""
    print("Testing memory management...")
    
    # Test memory guard
    guard = MemoryGuard(max_gb=16, safety_margin_gb=2)
    
    # Test protected execution
    with guard.protected_execution("test_operation"):
        params = guard.get_current_params()
        assert params['fps'] == 1.0
        assert params['window'] == 16
    
    # Test memory stats
    stats = guard.get_memory_stats()
    assert 'available_gb' in stats
    assert 'current_level' in stats
    
    print(f"  Memory available: {stats['available_gb']:.2f}GB")
    print(f"  Current level: {stats['current_level']}")
    print("  ✓ Memory management working")


def test_score_normalization():
    """Test score normalization."""
    print("Testing score normalization...")
    
    normalizer = ScoreNormalizer()
    
    # Test with sample metrics
    metrics = {
        'content': 0.8,
        'narrative': 0.7,
        'tension': 0.6,
        'emphasis': 0.5,
        'continuity': 0.7,
        'rhythm_penalty': 0.2
    }
    
    score = normalizer.calculate_score(metrics)
    assert 0 <= score <= 1
    
    weights_info = normalizer.get_weights_info()
    assert 'content' in weights_info
    
    print(f"  Calculated score: {score:.3f}")
    print(f"  Weights: {weights_info}")
    print("  ✓ Score normalization working")


def test_validators():
    """Test input validators."""
    print("Testing validators...")
    
    # Test duration validator
    video_duration = 60.0
    min_seg = 3.0
    max_seg = 18.0
    
    adjusted_min, adjusted_max = DurationValidator.validate_segment_bounds(
        video_duration, min_seg, max_seg
    )
    
    assert adjusted_min >= min_seg or adjusted_min == video_duration * 0.2
    assert adjusted_max <= video_duration * 0.5
    
    print(f"  Video duration: {video_duration}s")
    print(f"  Adjusted bounds: {adjusted_min:.1f}s - {adjusted_max:.1f}s")
    
    # Test silence params
    adj_silence, adj_keep, adj_pad = DurationValidator.validate_silence_params(
        video_duration, 0.35, 0.40, 0.05
    )
    
    assert adj_silence <= video_duration * 0.1
    assert adj_keep <= video_duration * 0.3
    
    print(f"  Silence params validated")
    print("  ✓ Validators working")


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    # Create a test config
    config_content = """
[embeddings]
default_model = vjepa
backend = mps
fps = 1.0
window = 16
crop = 256
seed = 1234
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        f.flush()
        
        # Validate config
        result = ConfigValidator.validate_config(f.name, 'embeddings')
        
        assert result['embeddings']['default_model'] == 'vjepa'
        assert result['embeddings']['fps'] == 1.0
        assert 'strategy' in result['embeddings']  # Default applied
        
        print(f"  Config validated successfully")
        print(f"  Model: {result['embeddings']['default_model']}")
        print(f"  Backend: {result['embeddings']['backend']}")
        print("  ✓ Config validation working")
    
    os.unlink(f.name)


def test_thread_safety():
    """Test thread-safe operations."""
    print("Testing thread safety...")
    
    # Test thread-safe seed setting
    import threading
    
    def set_seeds_thread():
        set_seeds(1234)
    
    threads = []
    for i in range(5):
        t = threading.Thread(target=set_seeds_thread)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("  ✓ Thread-safe seed setting working")


def test_schema_validation():
    """Test JSON schema definitions."""
    print("Testing JSON schemas...")
    
    # Validate schema structure
    assert 'type' in TRANSCRIPT_SCHEMA
    assert 'properties' in TRANSCRIPT_SCHEMA
    assert 'segments' in TRANSCRIPT_SCHEMA['properties']
    
    print("  ✓ JSON schemas defined correctly")


def main():
    """Run all integration tests."""
    print("=" * 50)
    print("INTEGRATION TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_promotion_logic,
        test_memory_management,
        test_score_normalization,
        test_validators,
        test_config_validation,
        test_thread_safety,
        test_schema_validation
    ]
    
    failed = []
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(test.__name__)
    
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    if not failed:
        print("✓ All integration tests passed!")
        return 0
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())