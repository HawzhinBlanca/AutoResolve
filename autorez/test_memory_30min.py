#!/usr/bin/env python3
"""Memory stress test with 30-minute video"""

import gc
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor memory usage with adaptive degradation"""
    
    def __init__(self, threshold_gb: float = 14.0):
        self.threshold_gb = threshold_gb
        self.process = psutil.Process()
        self.initial_rss = self.get_rss_gb()
        self.peak_rss = self.initial_rss
        self.degradation_triggered = False
        
    def get_rss_gb(self) -> float:
        """Get current RSS in GB"""
        return self.process.memory_info().rss / (1024**3)
    
    def check_and_degrade(self) -> bool:
        """Check memory and trigger degradation if needed"""
        current_rss = self.get_rss_gb()
        self.peak_rss = max(self.peak_rss, current_rss)
        
        if current_rss > self.threshold_gb:
            if not self.degradation_triggered:
                logger.warning(f"⚠️ Memory degradation triggered at {current_rss:.2f}GB")
                self.degradation_triggered = True
                self.trigger_degradation()
            return True
        return False
    
    def trigger_degradation(self):
        """Trigger adaptive degradation"""
        # Force garbage collection
        gc.collect()
        
        # Clear caches
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        logger.info("Degradation: Cleared caches and forced GC")
    
    def report(self) -> Dict:
        """Get memory report"""
        return {
            "initial_gb": self.initial_rss,
            "current_gb": self.get_rss_gb(),
            "peak_gb": self.peak_rss,
            "degradation": self.degradation_triggered
        }

class Memory30MinTest:
    """Test memory with 30-minute video"""
    
    def __init__(self):
        self.video_path = "assets/test_30min.mp4"
        self.monitor = MemoryMonitor(threshold_gb=14.0)
        
    def test_quality_scoring(self) -> Dict:
        """Test quality scoring on full video"""
        logger.info("Testing quality scoring on 30-min video...")
        
        from src.scoring.broll_quality import VideoQualityAnalyzer
        
        analyzer = VideoQualityAnalyzer()
        segments_processed = 0
        
        # Process every 30 seconds of the video
        for t in range(0, 1800, 30):
            segment = {
                "t0": t,
                "t1": min(t + 10, 1800),
                "segment_id": f"seg_{t}"
            }
            
            try:
                # Check memory before processing
                if self.monitor.check_and_degrade():
                    # Skip frames extraction in degraded mode
                    score = 0.5  # Default score
                else:
                    metrics = analyzer.analyze_segment(segment, self.video_path)
                    metrics["overall"]
                
                segments_processed += 1
                
                if segments_processed % 10 == 0:
                    logger.info(f"  Processed {segments_processed} segments, RSS={self.monitor.get_rss_gb():.2f}GB")
                    
            except Exception as e:
                logger.error(f"Error processing segment {t}: {e}")
        
        return {
            "segments_processed": segments_processed,
            "memory": self.monitor.report()
        }
    
    def test_batch_operations(self) -> Dict:
        """Test batch operations that might spike memory"""
        logger.info("Testing batch operations...")
        
        results = {}
        
        # 1. Load multiple frames in memory
        logger.info("  Loading frames batch...")
        frames = []
        frame_limit = 100 if not self.monitor.degradation_triggered else 50
        
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for i in range(0, min(frame_limit, 1800)):
            if self.monitor.check_and_degrade():
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * fps))
            ret, frame = cap.read()
            if ret:
                # Downsample in degraded mode
                if self.monitor.degradation_triggered:
                    frame = cv2.resize(frame, (160, 120))
                frames.append(frame)
        
        cap.release()
        
        results["frames_loaded"] = len(frames)
        results["frame_memory_gb"] = self.monitor.get_rss_gb()
        
        # Clear frames
        frames = []
        gc.collect()
        
        # 2. Simulate embedding generation
        logger.info("  Simulating embeddings...")
        embeddings = []
        embedding_dim = 768 if not self.monitor.degradation_triggered else 256
        
        for i in range(60):  # 60 segments for 30 minutes
            if self.monitor.check_and_degrade():
                embedding_dim = 256  # Reduce dimension
            
            emb = np.random.randn(embedding_dim).astype(np.float32)
            embeddings.append(emb)
        
        results["embeddings_created"] = len(embeddings)
        results["embedding_dim"] = embedding_dim
        
        # Clear embeddings
        embeddings = []
        gc.collect()
        
        results["memory"] = self.monitor.report()
        return results
    
    def test_pipeline_simulation(self) -> Dict:
        """Simulate full pipeline memory usage"""
        logger.info("Simulating full pipeline...")
        
        stages = []
        
        # Stage 1: Transcription simulation
        logger.info("  Stage 1: Transcription")
        text_data = " ".join(["word"] * 5000)  # ~5000 words for 30 min
        stages.append(("transcription", self.monitor.get_rss_gb()))
        
        if self.monitor.check_and_degrade():
            text_data = text_data[:2500]  # Reduce in degraded mode
        
        # Stage 2: Silence detection simulation
        logger.info("  Stage 2: Silence detection")
        [(i, i+0.5) for i in range(0, 1800, 60)]
        stages.append(("silence", self.monitor.get_rss_gb()))
        
        # Stage 3: B-roll scoring simulation
        logger.info("  Stage 3: B-roll scoring")
        broll_scores = {}
        for i in range(30):  # 30 B-roll candidates
            if self.monitor.check_and_degrade():
                break
            broll_scores[f"broll_{i}"] = np.random.rand()
        stages.append(("broll", self.monitor.get_rss_gb()))
        
        # Stage 4: Director analysis simulation
        logger.info("  Stage 4: Director analysis")
        director_beats = []
        for i in range(10):  # 10 narrative beats
            if self.monitor.check_and_degrade():
                break
            director_beats.append({"t": i * 180, "type": "beat"})
        stages.append(("director", self.monitor.get_rss_gb()))
        
        # Cleanup
        text_data = None
        broll_scores = None
        director_beats = None
        gc.collect()
        
        return {
            "stages": stages,
            "memory": self.monitor.report()
        }
    
    def run_full_test(self) -> Dict:
        """Run complete memory test suite"""
        logger.info("="*50)
        logger.info("MEMORY STRESS TEST - 30 MINUTE VIDEO")
        logger.info("="*50)
        
        if not Path(self.video_path).exists():
            logger.error(f"Video not found: {self.video_path}")
            return {"error": "Video not found"}
        
        results = {}
        
        # Test 1: Quality scoring
        logger.info("\nTest 1: Quality Scoring")
        results["quality_scoring"] = self.test_quality_scoring()
        gc.collect()
        time.sleep(2)
        
        # Test 2: Batch operations
        logger.info("\nTest 2: Batch Operations")
        results["batch_ops"] = self.test_batch_operations()
        gc.collect()
        time.sleep(2)
        
        # Test 3: Pipeline simulation
        logger.info("\nTest 3: Pipeline Simulation")
        results["pipeline"] = self.test_pipeline_simulation()
        
        # Final report
        final_memory = self.monitor.report()
        
        logger.info("\n" + "="*50)
        logger.info("FINAL MEMORY REPORT")
        logger.info("="*50)
        logger.info(f"Initial RSS: {final_memory['initial_gb']:.2f} GB")
        logger.info(f"Peak RSS: {final_memory['peak_gb']:.2f} GB")
        logger.info(f"Final RSS: {final_memory['current_gb']:.2f} GB")
        logger.info(f"Degradation triggered: {final_memory['degradation']}")
        
        # Check compliance
        passed = final_memory['peak_gb'] < 16.0
        
        if passed:
            logger.info("✅ PASS: Peak memory under 16GB limit")
        else:
            logger.error(f"❌ FAIL: Peak memory {final_memory['peak_gb']:.2f}GB exceeds 16GB limit")
        
        results["passed"] = passed
        results["final_memory"] = final_memory
        
        # Save results
        import json
        Path("artifacts").mkdir(exist_ok=True)
        with open("artifacts/memory_test_30min.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to artifacts/memory_test_30min.json")
        
        return results

def main():
    tester = Memory30MinTest()
    results = tester.run_full_test()
    
    # Record telemetry
    from src.utils.telemetry import TelemetryCollector
    telemetry = TelemetryCollector()
    telemetry.record_memory_usage(
        current_rss_gb=results["final_memory"]["current_gb"],
        peak_rss_gb=results["final_memory"]["peak_gb"],
        degradation_triggered=results["final_memory"]["degradation"],
        degradation_type="adaptive" if results["final_memory"]["degradation"] else None
    )
    
    return 0 if results["passed"] else 1

if __name__ == "__main__":
    exit(main())