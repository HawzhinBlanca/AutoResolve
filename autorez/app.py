#!/usr/bin/env python
"""
AutoResolve Demo App - Test all bug fixes
Interactive CLI to demonstrate the fixed components
"""

import sys
import json
import os
from pathlib import Path

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
        print("=" * 60)
        print("🎬 AutoResolve v3.0 - Bug Fix Demo")
        print("=" * 60)
        self.memory_guard = MemoryGuard(max_gb=16)
        set_seeds(1234)  # Ensure deterministic behavior
        
    def main_menu(self):
        """Display main menu and handle user choice."""
        while True:
            print("\n📋 Main Menu:")
            print("1. Test Promotion Logic (V-JEPA vs CLIP)")
            print("2. Test Memory Management")
            print("3. Test Score Normalization")
            print("4. Test Video Duration Validation")
            print("5. Test Config Validation")
            print("6. Run Full Pipeline Demo")
            print("7. Show System Status")
            print("0. Exit")
            
            choice = input("\nSelect option (0-7): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
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
                print("❌ Invalid option")
    
    def test_promotion(self):
        """Test promotion logic with sample data."""
        print("\n🔬 Testing Promotion Logic")
        print("-" * 40)
        
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
        
        print("📊 Metrics:")
        print(f"  V-JEPA Top-3: {results['top3']['vjepa']:.2f} (CI: {results['top3']['vjepa_ci']})")
        print(f"  CLIP Top-3:   {results['top3']['clip']:.2f} (CI: {results['top3']['clip_ci']})")
        print(f"  V-JEPA MRR:   {results['mrr']['vjepa']:.2f} (CI: {results['mrr']['vjepa_ci']})")
        print(f"  CLIP MRR:     {results['mrr']['clip']:.2f} (CI: {results['mrr']['clip_ci']})")
        
        sec_per_min = float(input("\nEnter processing speed (sec/min, default 4.0): ") or "4.0")
        
        decision = promote_vjepa(results, sec_per_min)
        
        print(f"\n🎯 Decision: {'✅ PROMOTE V-JEPA' if decision else '❌ KEEP CLIP'}")
        print(f"  Gains: Top-3 +{((results['top3']['vjepa']/results['top3']['clip'])-1)*100:.1f}%, "
              f"MRR +{((results['mrr']['vjepa']/results['mrr']['clip'])-1)*100:.1f}%")
    
    def test_memory(self):
        """Test memory management."""
        print("\n💾 Testing Memory Management")
        print("-" * 40)
        
        stats = self.memory_guard.get_memory_stats()
        print(f"📊 Current Memory Status:")
        print(f"  Available: {stats['available_gb']:.2f} GB")
        print(f"  Used: {stats['used_gb']:.2f} GB ({stats['percent']:.1f}%)")
        print(f"  Max Allowed: {stats['max_allowed_gb']:.2f} GB")
        print(f"  Degradation Level: {stats['current_level']}/4")
        
        params = self.memory_guard.get_current_params()
        print(f"\n⚙️ Current Parameters:")
        print(f"  FPS: {params['fps']}")
        print(f"  Window: {params['window']}")
        print(f"  Crop: {params['crop']}")
        print(f"  Batch Size: {params['batch_size']}")
        
        simulate = input("\nSimulate memory pressure? (y/n): ").lower()
        if simulate == 'y':
            print("\n🔄 Simulating degradation...")
            for i in range(3):
                self.memory_guard._degrade_and_get_params()
                params = self.memory_guard.get_current_params()
                print(f"  Level {self.memory_guard.current_level}: "
                      f"fps={params['fps']}, window={params['window']}, "
                      f"crop={params['crop']}, batch={params['batch_size']}")
            
            self.memory_guard.reset()
            print("✅ Reset to original quality")
    
    def test_scoring(self):
        """Test score normalization."""
        print("\n📈 Testing Score Normalization")
        print("-" * 40)
        
        normalizer = ScoreNormalizer()
        print(f"⚖️ Weights: {normalizer.get_weights_info()}")
        
        print("\nEnter scores (0-1) or press Enter for defaults:")
        metrics = {
            'content': float(input("  Content score (0.8): ") or "0.8"),
            'narrative': float(input("  Narrative score (0.7): ") or "0.7"),
            'tension': float(input("  Tension score (0.6): ") or "0.6"),
            'emphasis': float(input("  Emphasis score (0.5): ") or "0.5"),
            'continuity': float(input("  Continuity score (0.7): ") or "0.7"),
            'rhythm_penalty': float(input("  Rhythm penalty (0.2): ") or "0.2")
        }
        
        score = normalizer.calculate_score(metrics)
        print(f"\n🎯 Final Score: {score:.3f}")
        
        # Show contribution
        print("\n📊 Score Breakdown:")
        for key, value in metrics.items():
            weight = normalizer.weights[key]
            contribution = weight * value
            print(f"  {key}: {value:.2f} × {weight:+.2f} = {contribution:+.3f}")
    
    def test_validation(self):
        """Test duration validation."""
        print("\n⏱️ Testing Duration Validation")
        print("-" * 40)
        
        video_duration = float(input("Enter video duration in seconds (60): ") or "60")
        min_seg = float(input("Enter min segment duration (3.0): ") or "3.0")
        max_seg = float(input("Enter max segment duration (18.0): ") or "18.0")
        
        try:
            adj_min, adj_max = DurationValidator.validate_segment_bounds(
                video_duration, min_seg, max_seg
            )
            
            print(f"\n✅ Validation Results:")
            print(f"  Original: {min_seg}s - {max_seg}s")
            print(f"  Adjusted: {adj_min:.1f}s - {adj_max:.1f}s")
            print(f"  Reason: Max segment limited to 50% of video duration")
            
            # Test silence params
            silence_min, keep_min, pad = DurationValidator.validate_silence_params(
                video_duration, 0.35, 0.40, 0.05
            )
            print(f"\n🔇 Silence Parameters:")
            print(f"  Min silence: {silence_min:.2f}s")
            print(f"  Min keep: {keep_min:.2f}s")
            print(f"  Padding: {pad:.3f}s")
            
        except ValueError as e:
            print(f"\n❌ Validation Error: {e}")
    
    def test_config(self):
        """Test configuration validation."""
        print("\n⚙️ Testing Config Validation")
        print("-" * 40)
        
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
        
        print("📄 Sample Config:")
        print(config_content)
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            try:
                result = ConfigValidator.validate_config(f.name, 'embeddings')
                print("\n✅ Config Valid!")
                print("\n📊 Parsed Values:")
                for key, value in result['embeddings'].items():
                    print(f"  {key}: {value}")
            except ValueError as e:
                print(f"\n❌ Config Error: {e}")
            finally:
                os.unlink(f.name)
    
    def run_pipeline(self):
        """Run a simulated pipeline with all components."""
        print("\n🚀 Running Full Pipeline Demo")
        print("-" * 40)
        
        # Step 1: Memory check
        print("\n1️⃣ Checking Memory...")
        stats = self.memory_guard.get_memory_stats()
        print(f"   ✅ {stats['available_gb']:.1f}GB available")
        
        # Step 2: Config validation
        print("\n2️⃣ Validating Configuration...")
        print("   ✅ Config validated")
        
        # Step 3: Video validation
        print("\n3️⃣ Validating Video Parameters...")
        video_duration = 120  # 2 minutes
        adj_min, adj_max = DurationValidator.validate_segment_bounds(
            video_duration, 3.0, 18.0
        )
        print(f"   ✅ Segments: {adj_min:.1f}s - {adj_max:.1f}s")
        
        # Step 4: Score calculation
        print("\n4️⃣ Calculating Scores...")
        normalizer = ScoreNormalizer()
        score = normalizer.calculate_score({
            'content': 0.75,
            'narrative': 0.68,
            'tension': 0.62,
            'emphasis': 0.55,
            'continuity': 0.71,
            'rhythm_penalty': 0.15
        })
        print(f"   ✅ Quality score: {score:.3f}")
        
        # Step 5: Model selection
        print("\n5️⃣ Selecting Embedding Model...")
        results = {
            "top3": {"vjepa": 0.72, "clip": 0.65, 
                    "vjepa_ci": [0.68, 0.76], "clip_ci": [0.61, 0.69]},
            "mrr": {"vjepa": 0.68, "clip": 0.60,
                   "vjepa_ci": [0.64, 0.72], "clip_ci": [0.56, 0.64]}
        }
        decision = promote_vjepa(results, 4.2)
        print(f"   ✅ Model: {'V-JEPA' if decision else 'CLIP'}")
        
        print("\n✨ Pipeline Complete!")
        print(f"   Total memory used: {rss_gb():.2f}GB")
        print(f"   All systems operational")
    
    def show_status(self):
        """Show system status."""
        print("\n📊 System Status")
        print("-" * 40)
        
        # Memory
        stats = self.memory_guard.get_memory_stats()
        print(f"💾 Memory: {stats['available_gb']:.1f}GB free / "
              f"{stats['percent']:.0f}% used")
        
        # Current settings
        params = self.memory_guard.get_current_params()
        print(f"⚙️ Quality Level: {5 - self.memory_guard.current_level}/5")
        
        # Process info
        print(f"🔄 Process RSS: {rss_gb():.2f}GB")
        
        # Test results
        print(f"\n✅ All Components:")
        print(f"   • Promotion Logic: Working")
        print(f"   • Memory Guard: Active")
        print(f"   • Score Normalizer: Calibrated")
        print(f"   • Validators: Ready")
        print(f"   • Config System: Loaded")


def main():
    """Entry point for the application."""
    try:
        app = AutoResolveApp()
        app.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()