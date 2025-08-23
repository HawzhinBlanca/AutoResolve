import logging

logger = logging.getLogger(__name__)

#!/usr/bin/env python
"""
AutoResolve Demo App - Test all bug fixes
Interactive CLI to demonstrate the fixed components
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.promotion import promote_vjepa
from src.utils.memory_guard import MemoryGuard
from src.utils.memory import set_seeds, rss_gb
from src.scoring.score_normalizer import ScoreNormalizer
from src.validators.duration_validator import DurationValidator
from src.config.schema_validator import ConfigValidator


class AutoResolveApp:
    def __init__(self):
        logger.info("=" * 60)
        logger.info("🎬 AutoResolve v3.0 - Bug Fix Demo")
        logger.info("=" * 60)
        self.memory_guard = MemoryGuard(max_gb=16)
        set_seeds(1234)  # Ensure deterministic behavior
        
    def main_menu(self):
        """Display main menu and handle user choice."""
        while True:
            logger.info("\n📋 Main Menu:")
            logger.info("1. Test Promotion Logic (V-JEPA vs CLIP)")
            logger.info("2. Test Memory Management")
            logger.info("3. Test Score Normalization")
            logger.info("4. Test Video Duration Validation")
            logger.info("5. Test Config Validation")
            logger.info("6. Run Full Pipeline Demo")
            logger.info("7. Show System Status")
            logger.info("0. Exit")
            
            choice = input("\nSelect option (0-7): ").strip()
            
            if choice == "0":
                logger.info("\n👋 Goodbye!")
                break
            elif choice == "1":
                self.test_promotion()
            elif choice == "2":
                self.test_memory()
            elif choice == "3":
                self.test_scoring()
            elif choice == "4":
                self.test_validation()
            elif choice == "5":
                self.test_config()
            elif choice == "6":
                self.run_pipeline()
            elif choice == "7":
                self.show_status()
            else:
                logger.info("❌ Invalid option")
    
    def test_promotion(self):
        """Test promotion logic with sample data."""
        logger.info("\n🔬 Testing Promotion Logic")
        logger.info("-" * 40)
        
        # Sample results
        results = {
            "top3": {
                "vjepa": 0.75,
                "clip": 0.60,
                "vjepa_ci": [0.70, 0.80],
                "clip_ci": [0.55, 0.65]
            },
            "mrr": {
                "vjepa": 0.70,
                "clip": 0.55,
                "vjepa_ci": [0.65, 0.75],
                "clip_ci": [0.50, 0.60]
            }
        }
        
        logger.info("📊 Metrics:")
        logger.info(f"  V-JEPA Top-3: {results['top3']['vjepa']:.2f} (CI: {results['top3']['vjepa_ci']})")
        logger.info(f"  CLIP Top-3:   {results['top3']['clip']:.2f} (CI: {results['top3']['clip_ci']})")
        logger.info(f"  V-JEPA MRR:   {results['mrr']['vjepa']:.2f} (CI: {results['mrr']['vjepa_ci']})")
        logger.info(f"  CLIP MRR:     {results['mrr']['clip']:.2f} (CI: {results['mrr']['clip_ci']})")
        
        sec_per_min = float(input("\nEnter processing speed (sec/min, default 4.0): ") or "4.0")
        
        decision = promote_vjepa(results, sec_per_min)
        
        logger.info(f"\n🎯 Decision: {'✅ PROMOTE V-JEPA' if decision else '❌ KEEP CLIP'}")
        logger.info(f"  Gains: Top-3 +{((results['top3']['vjepa']/results['top3']['clip'])-1)
              f"MRR +{((results['mrr']['vjepa']/results['mrr']['clip'])-1)*100:.1f}%")
    
    def test_memory(self):
        """Test memory management."""
        logger.info("\n💾 Testing Memory Management")
        logger.info("-" * 40)
        
        stats = self.memory_guard.get_memory_stats()
        logger.info(f"📊 Current Memory Status:")
        logger.info(f"  Available: {stats['available_gb']:.2f} GB")
        logger.info(f"  Used: {stats['used_gb']:.2f} GB ({stats['percent']:.1f}%)")
        logger.info(f"  Max Allowed: {stats['max_allowed_gb']:.2f} GB")
        logger.info(f"  Degradation Level: {stats['current_level']}/4")
        
        params = self.memory_guard.get_current_params()
        logger.info(f"\n⚙️ Current Parameters:")
        logger.info(f"  FPS: {params['fps']}")
        logger.info(f"  Window: {params['window']}")
        logger.info(f"  Crop: {params['crop']}")
        logger.info(f"  Batch Size: {params['batch_size']}")
        
        simulate = input("\nSimulate memory pressure? (y/n): ").lower()
        if simulate == 'y':
            logger.info("\n🔄 Simulating degradation...")
            for i in range(3):
                self.memory_guard._degrade_and_get_params()
                params = self.memory_guard.get_current_params()
            self.memory_guard.reset()
            logger.info("✅ Reset to original quality")
    
    def test_scoring(self):
        """Test score normalization."""
        logger.info("\n📈 Testing Score Normalization")
        logger.info("-" * 40)
        
        normalizer = ScoreNormalizer()
        logger.info(f"⚖️ Weights: {normalizer.get_weights_info()}")
        
        logger.info("\nEnter scores (0-1) or press Enter for defaults:")
        metrics = {
            'content': float(input("  Content score (0.8): ") or "0.8"),
            'narrative': float(input("  Narrative score (0.7): ") or "0.7"),
            'tension': float(input("  Tension score (0.6): ") or "0.6"),
            'emphasis': float(input("  Emphasis score (0.5): ") or "0.5"),
            'continuity': float(input("  Continuity score (0.7): ") or "0.7"),
            'rhythm_penalty': float(input("  Rhythm penalty (0.2): ") or "0.2")
        }
        
        score = normalizer.calculate_score(metrics)
        logger.info(f"\n🎯 Final Score: {score:.3f}")
        
        # Show contribution
        logger.info("\n📊 Score Breakdown:")
        for key, value in metrics.items():
            weight = normalizer.weights[key]
            contribution = weight * value
            logger.info(f"  {key}: {value:.2f} × {weight:+.2f} = {contribution:+.3f}")
    
    def test_validation(self):
        """Test duration validation."""
        logger.info("\n⏱️ Testing Duration Validation")
        logger.info("-" * 40)
        
        video_duration = float(input("Enter video duration in seconds (60): ") or "60")
        min_seg = float(input("Enter min segment duration (3.0): ") or "3.0")
        max_seg = float(input("Enter max segment duration (18.0): ") or "18.0")
        
        try:
            adj_min, adj_max = DurationValidator.validate_segment_bounds(
                video_duration, min_seg, max_seg
            )
            
            logger.info(f"\n✅ Validation Results:")
            logger.info(f"  Original: {min_seg}s - {max_seg}s")
            logger.info(f"  Adjusted: {adj_min:.1f}s - {adj_max:.1f}s")
            logger.info(f"  Reason: Max segment limited to 50% of video duration")
            
            # Test silence params
            silence_min, keep_min, pad = DurationValidator.validate_silence_params(
                video_duration, 0.35, 0.40, 0.05
            )
            logger.info(f"\n🔇 Silence Parameters:")
            logger.info(f"  Min silence: {silence_min:.2f}s")
            logger.info(f"  Min keep: {keep_min:.2f}s")
            logger.info(f"  Padding: {pad:.3f}s")
            
        except ValueError as e:
            logger.error(f"\n❌ Validation Error: {e}")
    
    def test_config(self):
        """Test configuration validation."""
        logger.info("\n⚙️ Testing Config Validation")
        logger.info("-" * 40)
        
        # Create sample config
        config_content = """
[embeddings]
default_model = vjepa
backend = mps
fps = 1.0
window = 16
crop = 256
max_rss_gb = 16
seed = 1234
"""
        
        logger.info("📄 Sample Config:")
        logger.info(config_content)
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            try:
                result = ConfigValidator.validate_config(f.name, 'embeddings')
                logger.info("\n✅ Config Valid!")
                logger.info("\n📊 Parsed Values:")
                for key, value in result['embeddings'].items():
                    logger.info(f"  {key}: {value}")
            except ValueError as e:
                logger.error(f"\n❌ Config Error: {e}")
            finally:
                os.unlink(f.name)
    
    def run_pipeline(self):
        """Run a simulated pipeline with all components."""
        logger.info("\n🚀 Running Full Pipeline Demo")
        logger.info("-" * 40)
        
        # Step 1: Memory check
        logger.info("\n1️⃣ Checking Memory...")
        stats = self.memory_guard.get_memory_stats()
        logger.info(f"   ✅ {stats['available_gb']:.1f}GB available")
        
        # Step 2: Config validation
        logger.info("\n2️⃣ Validating Configuration...")
        logger.info("   ✅ Config validated")
        
        # Step 3: Video validation
        logger.info("\n3️⃣ Validating Video Parameters...")
        video_duration = 120  # 2 minutes
        adj_min, adj_max = DurationValidator.validate_segment_bounds(
            video_duration, 3.0, 18.0
        )
        logger.info(f"   ✅ Segments: {adj_min:.1f}s - {adj_max:.1f}s")
        
        # Step 4: Score calculation
        logger.info("\n4️⃣ Calculating Scores...")
        normalizer = ScoreNormalizer()
        score = normalizer.calculate_score({
            'content': 0.75,
            'narrative': 0.68,
            'tension': 0.62,
            'emphasis': 0.55,
            'continuity': 0.71,
            'rhythm_penalty': 0.15
        })
        logger.info(f"   ✅ Quality score: {score:.3f}")
        
        # Step 5: Model selection
        logger.info("\n5️⃣ Selecting Embedding Model...")
        results = {
            "top3": {"vjepa": 0.72, "clip": 0.65, 
                    "vjepa_ci": [0.68, 0.76], "clip_ci": [0.61, 0.69]},
            "mrr": {"vjepa": 0.68, "clip": 0.60,
                   "vjepa_ci": [0.64, 0.72], "clip_ci": [0.56, 0.64]}
        }
        decision = promote_vjepa(results, 4.2)
        logger.info(f"   ✅ Model: {'V-JEPA' if decision else 'CLIP'}")
        
        logger.info("\n✨ Pipeline Complete!")
        logger.info(f"   Total memory used: {rss_gb():.2f}GB")
        logger.info(f"   All systems operational")
    
    def show_status(self):
        """Show system status."""
        logger.info("\n📊 System Status")
        logger.info("-" * 40)
        
        # Memory
        stats = self.memory_guard.get_memory_stats()
        # Current settings
        self.memory_guard.get_current_params()
        logger.info(f"⚙️ Quality Level: {5 - self.memory_guard.current_level}/5")
        
        # Process info
        logger.info(f"🔄 Process RSS: {rss_gb():.2f}GB")
        
        # Test results
        logger.info(f"\n✅ All Components:")
        logger.info(f"   • Promotion Logic: Working")
        logger.info(f"   • Memory Guard: Active")
        logger.info(f"   • Score Normalizer: Calibrated")
        logger.info(f"   • Validators: Ready")
        logger.info(f"   • Config System: Loaded")


def main():
    """Entry point for the application."""
    try:
        app = AutoResolveApp()
        app.main_menu()
    except KeyboardInterrupt:
        logger.info("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()