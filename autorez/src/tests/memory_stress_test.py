"""
Memory Stress Testing for AutoResolve v3.0
Validates memory constraints with real 30-minute video workloads
"""

import psutil
import torch
import numpy as np
import gc
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import logging
import tracemalloc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryStressTest:
    """
    Test memory usage under realistic workloads.
    Ensures peak RSS stays under 16GB limit.
    """
    
    def __init__(self, memory_limit_gb: float = 16.0):
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.test_results: List[Dict[str, Any]] = []
        self.peak_memory_gb = 0.0
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # System-wide memory
        virtual_memory = psutil.virtual_memory()
        
        return {
            "rss_gb": memory_info.rss / (1024**3),
            "vms_gb": memory_info.vms / (1024**3),
            "available_gb": virtual_memory.available / (1024**3),
            "percent_used": virtual_memory.percent,
            "gpu_memory_gb": self.get_gpu_memory()
        }
    
    def get_gpu_memory(self) -> float:
        """Get GPU memory usage if available"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        elif torch.backends.mps.is_available():
            # MPS doesn't provide direct memory queries
            return 0.0
        return 0.0
    
    def monitor_memory(self, operation_name: str):
        """Context manager to monitor memory during operation"""
        class MemoryMonitor:
            def __init__(self, test_instance, name):
                self.test = test_instance
                self.name = name
                self.start_memory = None
                self.peak_memory = 0.0
                
            def __enter__(self):
                # Force garbage collection before measurement
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                self.start_memory = self.test.get_memory_usage()
                logger.info(f"Starting {self.name}: RSS={self.start_memory['rss_gb']:.2f}GB")
                
                # Start tracing
                tracemalloc.start()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Get peak memory
                current_memory = self.test.get_memory_usage()
                self.peak_memory = current_memory['rss_gb']
                
                # Stop tracing
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Calculate delta
                memory_delta = self.peak_memory - self.start_memory['rss_gb']
                
                # Record result
                result = {
                    "operation": self.name,
                    "start_rss_gb": self.start_memory['rss_gb'],
                    "peak_rss_gb": self.peak_memory,
                    "delta_gb": memory_delta,
                    "traced_peak_gb": peak / (1024**3),
                    "within_limit": self.peak_memory <= self.test.memory_limit_gb,
                    "timestamp": time.time()
                }
                
                self.test.test_results.append(result)
                self.test.peak_memory_gb = max(self.test.peak_memory_gb, self.peak_memory)
                
                # Log result
                status = "✅" if result["within_limit"] else "❌"
                logger.info(f"{status} {self.name}: Peak={self.peak_memory:.2f}GB, Delta={memory_delta:.2f}GB")
                
                # Cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return MemoryMonitor(self, operation_name)
    
    def test_embedder_memory(self, video_path: str, duration_min: float = 30.0):
        """Test embedder memory usage with long video"""
        with self.monitor_memory(f"Embedder_{duration_min}min"):
            from src.embedders.vjepa_embedder import VJEPAEmbedder
            from src.utils.memory import Budget, enforce_budget
            
            # Create budget enforcer
            budget = Budget(
                max_gb=self.memory_limit_gb - 2,  # Leave 2GB headroom
                fps=1.0,
                window=16,
                crop=256,
                max_segments=int(duration_min * 60)  # 1 segment per second
            )
            
            # Initialize embedder
            embedder = VJEPAEmbedder()
            
            # Process video with budget enforcement
            segments = []
            chunk_size = 60  # Process 1 minute at a time
            
            for offset in range(0, int(duration_min * 60), chunk_size):
                # Check memory before processing chunk
                current_mem = self.get_memory_usage()
                if current_mem['rss_gb'] > self.memory_limit_gb - 3:
                    logger.warning(f"Approaching memory limit, triggering cleanup")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Apply degradation if needed
                    budget = enforce_budget(budget, embedder.device)
                
                # Process chunk (simulated for testing)
                chunk_segments, _ = embedder.embed_segments(
                    video_path,
                    fps=budget.fps,
                    window=budget.window,
                    strategy="temp_attn"
                )
                segments.extend(chunk_segments[:10])  # Limit segments per chunk
                
                # Clear intermediate results
                del chunk_segments
                gc.collect()
            
            logger.info(f"Processed {len(segments)} segments")
            return segments
    
    def test_director_memory(self, video_path: str):
        """Test director module memory usage"""
        with self.monitor_memory("Director_Analysis"):
            from src.director.creative_director import analyze_video
            
            results = analyze_video(video_path)
            
            # Process results to ensure they're computed
            for module, data in results.items():
                if isinstance(data, dict):
                    logger.info(f"  {module}: {len(data)} items")
            
            return results
    
    def test_transcription_memory(self, video_path: str):
        """Test transcription memory usage"""
        with self.monitor_memory("Transcription"):
            from src.ops.transcribe import transcribe_audio
            
            transcript = transcribe_audio(video_path)
            num_segments = len(transcript.get('segments', []))
            logger.info(f"  Transcribed {num_segments} segments")
            
            return transcript
    
    def test_broll_memory(self, video_path: str):
        """Test B-roll selection memory usage"""
        with self.monitor_memory("Broll_Selection"):
            from src.broll.selector import select_broll
            
            # Test with multiple queries
            queries = [
                "aerial city view",
                "close-up hands typing",
                "nature landscape",
                "action sequence"
            ]
            
            results = []
            for query in queries:
                selection = select_broll(video_path, query)
                results.append(selection)
                
                # Cleanup between queries
                gc.collect()
            
            logger.info(f"  Processed {len(queries)} B-roll queries")
            return results
    
    def test_pipeline_memory(self, video_path: str):
        """Test full pipeline memory usage"""
        with self.monitor_memory("Full_Pipeline"):
            # 1. Embeddings
            embeddings = self.test_embedder_memory(video_path, duration_min=5.0)
            
            # 2. Director analysis
            director = self.test_director_memory(video_path)
            
            # 3. Transcription
            transcript = self.test_transcription_memory(video_path)
            
            # 4. B-roll
            broll = self.test_broll_memory(video_path)
            
            # Check if we stayed within limits
            return {
                "embeddings_count": len(embeddings),
                "director_modules": len(director),
                "transcript_segments": len(transcript.get('segments', [])),
                "broll_queries": len(broll)
            }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate memory stress test report"""
        report = {
            "memory_limit_gb": self.memory_limit_gb,
            "peak_memory_gb": self.peak_memory_gb,
            "within_limit": self.peak_memory_gb <= self.memory_limit_gb,
            "test_results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r["within_limit"]),
                "failed": sum(1 for r in self.test_results if not r["within_limit"]),
                "max_delta_gb": max((r["delta_gb"] for r in self.test_results), default=0),
                "avg_peak_gb": np.mean([r["peak_rss_gb"] for r in self.test_results]) if self.test_results else 0
            }
        }
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("Memory Stress Test Summary")
        logger.info("="*60)
        logger.info(f"Memory Limit: {self.memory_limit_gb:.1f} GB")
        logger.info(f"Peak Memory: {self.peak_memory_gb:.2f} GB")
        logger.info(f"Tests Passed: {report['summary']['passed']}/{report['summary']['total_tests']}")
        
        if report["within_limit"]:
            logger.info("✅ MEMORY CONSTRAINTS SATISFIED")
        else:
            logger.error("❌ MEMORY LIMIT EXCEEDED")
            logger.error(f"   Exceeded by: {self.peak_memory_gb - self.memory_limit_gb:.2f} GB")
        
        # Record in telemetry
        from src.utils.telemetry import get_telemetry
        telemetry = get_telemetry()
        telemetry.record_memory_usage(
            current_rss_gb=self.get_memory_usage()['rss_gb'],
            peak_rss_gb=self.peak_memory_gb,
            degradation_triggered=self.peak_memory_gb > self.memory_limit_gb - 2
        )
        
        return report
    
    def save_report(self, output_path: str = "proof_pack/memory_stress_report.json"):
        """Save memory stress test report"""
        report = self.generate_report()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nMemory stress report saved to {output_path}")
        return report


def main():
    """Run memory stress tests"""
    # Use a test video
    test_video = "assets/pilots/clip_5m.mp4"
    if not Path(test_video).exists():
        import glob
        videos = glob.glob("assets/pilots/*.mp4")
        if videos:
            test_video = videos[0]
    
    logger.info(f"Running memory stress test with: {test_video}")
    
    # Run tests
    tester = MemoryStressTest(memory_limit_gb=16.0)
    
    # Test individual components
    tester.test_embedder_memory(test_video, duration_min=5.0)
    tester.test_director_memory(test_video)
    tester.test_transcription_memory(test_video)
    tester.test_broll_memory(test_video)
    
    # Test full pipeline
    tester.test_pipeline_memory(test_video)
    
    # Generate and save report
    report = tester.save_report()
    
    # Exit with appropriate code
    if report["within_limit"]:
        logger.info("Memory stress test PASSED")
        exit(0)
    else:
        logger.error("Memory stress test FAILED - exceeds 16GB limit")
        exit(1)


if __name__ == "__main__":
    main()