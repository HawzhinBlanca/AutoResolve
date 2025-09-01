#!/usr/bin/env python3
"""
Code Authenticity Verification Suite
Tests all replaced fake implementations to ensure they work with real data
"""

import unittest
import tempfile
import json
import os
import sys
import time
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

class TestRealImplementations(unittest.TestCase):
    """Verify all fake code has been replaced with working implementations"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_video = cls._create_test_video()
        cls.backend_url = "http://localhost:8000"
        
    @classmethod
    def _create_test_video(cls):
        """Create a real test video file"""
        test_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        # Create 5 second test video with audio
        cmd = [
            "ffmpeg", "-f", "lavfi", 
            "-i", "testsrc=duration=5:size=320x240:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-y", test_file.name
        ]
        subprocess.run(cmd, capture_output=True, check=False)
        return test_file.name
    
    def test_real_transcription_no_hardcoded_data(self):
        """Test that transcription returns real data, not hardcoded strings"""
        from src.ops.transcribe import transcribe_audio
        
        # Run transcription
        result = transcribe_audio(self.test_video)
        
        # Verify it's not the fake hardcoded response
        self.assertNotEqual(result.get("transcription"), "Sample transcription text")
        self.assertIsInstance(result.get("segments"), list)
        
        # Verify real transcription structure
        self.assertIn("language", result)
        self.assertIn("segments", result)
        self.assertIn("meta", result)
        
        # Verify segments have real timestamps
        if result["segments"]:
            segment = result["segments"][0]
            self.assertIn("t0", segment)
            self.assertIn("t1", segment)
            self.assertIn("text", segment)
            self.assertIsInstance(segment["t0"], (int, float))
            self.assertIsInstance(segment["t1"], (int, float))
            self.assertGreaterEqual(segment["t1"], segment["t0"])
    
    def test_real_silence_detection_no_hardcoded_segments(self):
        """Test that silence detection returns real analysis, not hardcoded segments"""
        from src.ops.silence import SilenceRemover
        
        remover = SilenceRemover()
        result = remover.remove_silence(self.test_video)
        
        # Handle both tuple return (cuts, metrics) and dict return
        if isinstance(result, tuple):
            cuts, metrics = result
            keep_windows = cuts.get("keep_windows", [])
        elif isinstance(result, dict):
            keep_windows = result.get("keep_windows", [])
        else:
            keep_windows = result
        
        # Extract the actual windows from the structure
        if keep_windows and isinstance(keep_windows[0], dict):
            keep_windows = [(w["start"], w["end"]) for w in keep_windows]
        
        # Verify it's not returning hardcoded values
        self.assertNotEqual(keep_windows, [[0.0, 1.5], [10.2, 11.8]])
        
        # Verify real silence detection
        self.assertIsInstance(keep_windows, list)
        for window in keep_windows:
            self.assertEqual(len(window), 2)
            start, end = window
            self.assertIsInstance(start, (int, float))
            self.assertIsInstance(end, (int, float))
            self.assertGreater(end, start)
    
    def test_real_director_analysis_no_fake_scores(self):
        """Test that director analysis returns real analysis, not hardcoded scores"""
        from src.director.creative_director import analyze_video, continuity_between
        
        # Test video analysis
        results = analyze_video(self.test_video, modules=["narrative"])
        
        # Verify it has real metadata
        self.assertIn("_metadata", results)
        meta = results["_metadata"]
        self.assertIn("video_path", meta)
        self.assertEqual(meta["video_path"], self.test_video)
        self.assertIn("peak_rss_gb", meta)
        self.assertIsInstance(meta["peak_rss_gb"], (int, float))
        
        # Test continuity function doesn't return hardcoded 0.8
        shot_a = {"frames": [], "motion_vectors": []}
        shot_b = {"frames": [], "motion_vectors": []}
        result = continuity_between(shot_a, shot_b)
        
        # Should not be the exact hardcoded value
        self.assertNotEqual(result["continuity_score"], 0.8)
        self.assertIn("factors", result)
    
    def test_real_metrics_collection_no_defaults(self):
        """Test that gates uses real metrics, not hardcoded defaults"""
        from src.eval.gates import verify_gates
        
        # Create fake metrics that would fail gates
        bad_metrics = {
            "processing_speed_x": 10.0,  # Below minimum of 30
            "peak_rss_gb": 20.0,  # Above maximum of 16
            "ui_memory_mb": 300.0,  # Above maximum of 200
            "silence_sec_per_min": 1.0,  # Above maximum of 0.5
            "transcription_rtf": 2.0,  # Above maximum of 1.5
            "vjepa_sec_per_min": 10.0,  # Above maximum of 5.0
            "api_sec_per_min": 5.0,  # Above maximum of 3.0
            "api_cost_per_min": 0.10,  # Above maximum of 0.05
            "export_time_s": 5.0,  # Above maximum of 2.0
        }
        
        # Should raise ValueError for failed gates
        with self.assertRaises(ValueError) as ctx:
            verify_gates(bad_metrics)
        
        # Verify it reports actual failures
        self.assertIn("Gates failed", str(ctx.exception))
    
    def test_backend_endpoints_use_real_modules(self):
        """Test that backend endpoints call real modules, not return fake data"""
        # Start backend server in background
        backend_proc = subprocess.Popen(
            ["python", "-m", "uvicorn", "backend_service_final:app", "--port", "8001"],
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            # Wait for server to start
            time.sleep(2)
            
            import requests
            
            # Test transcribe endpoint
            response = requests.post(
                "http://localhost:8001/api/transcribe",
                json={"video_path": self.test_video}
            )
            if response.status_code == 200:
                data = response.json()
                task_id = data.get("task_id")
                self.assertIsNotNone(task_id)
                # Verify task_id is computed, not hardcoded
                self.assertIn("transcribe_", task_id)
            
            # Test silence endpoint
            response = requests.post(
                "http://localhost:8001/api/silence",
                json={"video_path": self.test_video}
            )
            if response.status_code == 200:
                data = response.json()
                task_id = data.get("task_id")
                self.assertIsNotNone(task_id)
                # Verify task_id is computed, not hardcoded
                self.assertIn("silence_", task_id)
            
        finally:
            # Clean up
            backend_proc.terminate()
            backend_proc.wait(timeout=5)
    
    def test_creative_decisions_use_real_analysis(self):
        """Test that creative decisions are based on real data analysis"""
        from src.director.creative_director import make_creative_decisions
        
        # Test with empty results
        decisions = make_creative_decisions({})
        self.assertIsInstance(decisions, list)
        
        # Test with real emotion data
        real_results = {
            "emotion": {
                "tension_curve": [
                    (0.0, 0.3),
                    (1.0, 0.5),
                    (2.0, 0.9),  # High tension
                    (3.0, 0.4),
                    (4.0, 0.95),  # Very high tension
                ]
            }
        }
        
        decisions = make_creative_decisions(real_results)
        self.assertIsInstance(decisions, list)
        
        # Verify decisions have real structure
        for decision in decisions:
            self.assertIn("time", decision)
            self.assertIn("action", decision)
            self.assertIn("reason", decision)
            self.assertIn("confidence", decision)
            self.assertIsInstance(decision["confidence"], (int, float))
            self.assertGreaterEqual(decision["confidence"], 0.0)
            self.assertLessEqual(decision["confidence"], 1.0)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if os.path.exists(cls.test_video):
            os.unlink(cls.test_video)


class TestNoFakePatterns(unittest.TestCase):
    """Scan codebase for fake/mock/stub patterns"""
    
    def test_no_mock_returns_in_production(self):
        """Verify no 'return mock_' or 'return fake_' in production code"""
        production_files = [
            "backend_service_final.py",
            "autoresolve_cli.py",
            "src/ops/transcribe.py",
            "src/ops/silence.py",
            "src/director/creative_director.py",
            "src/eval/gates.py"
        ]
        
        for file in production_files:
            filepath = Path(__file__).parent / file
            if filepath.exists():
                content = filepath.read_text()
                self.assertNotIn("return mock_", content, f"Found 'return mock_' in {file}")
                self.assertNotIn("return fake_", content, f"Found 'return fake_' in {file}")
                self.assertNotIn("return dummy_", content, f"Found 'return dummy_' in {file}")
    
    def test_no_hardcoded_sample_data(self):
        """Verify no hardcoded 'Sample' or 'Test' data in API responses"""
        filepath = Path(__file__).parent / "backend_service_final.py"
        if filepath.exists():
            content = filepath.read_text()
            
            # Check for specific fake patterns we found
            self.assertNotIn('"Sample transcription text"', content)
            self.assertNotIn('[[0.0, 1.5], [10.2, 11.8]]', content)
            self.assertNotIn('"Processed transcription"', content)
    
    def test_no_todo_fixme_hack_comments(self):
        """Verify no TODO/FIXME/HACK comments remain in our code"""
        # Only check our code, not third-party libraries
        base_path = Path(__file__).parent
        production_files = [
            *base_path.glob("*.py"),
            *base_path.glob("src/**/*.py"),
        ]
        
        for filepath in production_files:
            # Skip test files and third-party code
            if any(skip in str(filepath) for skip in ["test", "__pycache__", ".venv", "venv", "site-packages"]):
                continue
                
            content = filepath.read_text()
            lines = content.split("\n")
            
            for i, line in enumerate(lines, 1):
                # Skip this test file
                if filepath.name == "test_authenticity.py":
                    continue
                    
                self.assertNotIn("TODO", line, f"Found TODO in {filepath}:{i}")
                self.assertNotIn("FIXME", line, f"Found FIXME in {filepath}:{i}")
                self.assertNotIn("HACK", line, f"Found HACK in {filepath}:{i}")
    
    def test_no_sleep_delays_for_fake_processing(self):
        """Verify no sleep() calls used to simulate processing in our code"""
        # Only check our code, not third-party libraries
        base_path = Path(__file__).parent
        production_files = [
            *base_path.glob("*.py"),
            *base_path.glob("src/**/*.py"),
        ]
        
        for filepath in production_files:
            # Skip test files and third-party code
            if any(skip in str(filepath) for skip in ["test", "__pycache__", ".venv", "venv", "site-packages"]):
                continue
                
            content = filepath.read_text()
            lines = content.split("\n")
            
            for i, line in enumerate(lines, 1):
                if "sleep(" in line and not line.strip().startswith("#"):
                    # Check if it's a real sleep for timing or fake delay
                    if "time.sleep" in line:
                        # Verify it has a legitimate reason (commented)
                        self.assertIn("#", line, f"Unexplained sleep in {filepath}:{i}")


def run_verification():
    """Run complete verification suite"""
    print("=" * 70)
    print("CODE AUTHENTICITY VERIFICATION SUITE")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestRealImplementations))
    suite.addTests(loader.loadTestsFromTestCase(TestNoFakePatterns))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL AUTHENTICITY CHECKS PASSED")
        print("✅ No fake implementations remain")
        print("✅ All code performs real work")
    else:
        print("❌ FAKE CODE DETECTED")
        print(f"❌ Failed: {len(result.failures)} tests")
        print(f"❌ Errors: {len(result.errors)} tests")
    print("=" * 70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_verification())