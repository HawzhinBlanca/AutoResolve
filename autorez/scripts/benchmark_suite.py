#!/usr/bin/env python3
"""
AutoResolve v3.2 - Performance Benchmark Suite
Comprehensive performance testing and validation
"""

import time
import json
import psutil
import statistics
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class BenchmarkSuite:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.baseline_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def _measure_duration(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure function execution time"""
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        return result, duration
    
    def benchmark_embedders(self) -> Dict[str, Any]:
        """Benchmark all embedder modules"""
        print("🧠 Benchmarking Embedders...")
        
        embedder_results = {}
        
        # V-JEPA Benchmark
        try:
            from src.embedders.vjepa_embedder import VJEPAEmbedder
            
            print("  Testing V-JEPA embedder...")
            vjepa = VJEPAEmbedder(memory_safe_mode=True)
            
            # Test with different video lengths
            test_videos = ['assets/test_30min.mp4'] if os.path.exists('assets/test_30min.mp4') else []
            
            vjepa_timings = []
            for video in test_videos:
                segments, duration = self._measure_duration(
                    vjepa.embed_segments, 
                    video, 
                    fps=1.0, 
                    window=16, 
                    strategy='temp_attn',
                    max_segments=20
                )
                
                video_duration = 1800.0  # 30 min test video
                processing_speed = video_duration / duration if duration > 0 else 0
                sec_per_min = (duration / (video_duration / 60)) if video_duration > 0 else 0
                
                vjepa_timings.append({
                    'video': video,
                    'segments': len(segments),
                    'duration': duration,
                    'processing_speed_x': processing_speed,
                    'sec_per_min': sec_per_min
                })
            
            embedder_results['vjepa'] = {
                'timings': vjepa_timings,
                'avg_speed_x': statistics.mean([t['processing_speed_x'] for t in vjepa_timings]) if vjepa_timings else 0,
                'avg_sec_per_min': statistics.mean([t['sec_per_min'] for t in vjepa_timings]) if vjepa_timings else 0
            }
            
        except Exception as e:
            logger.error(f"V-JEPA benchmark failed: {e}")
            embedder_results['vjepa'] = {'error': str(e)}
        
        # CLIP Benchmark
        try:
            from src.embedders.clip_embedder import CLIPEmbedder
            
            print("  Testing CLIP embedder...")
            clip = CLIPEmbedder()
            
            clip_timings = []
            for video in test_videos:
                segments, duration = self._measure_duration(
                    clip.embed_segments,
                    video,
                    fps=1.0,
                    window=16,
                    strategy='temp_attn',
                    max_segments=20
                )
                
                video_duration = 1800.0
                processing_speed = video_duration / duration if duration > 0 else 0
                
                clip_timings.append({
                    'video': video,
                    'segments': len(segments),
                    'duration': duration,
                    'processing_speed_x': processing_speed
                })
            
            embedder_results['clip'] = {
                'timings': clip_timings,
                'avg_speed_x': statistics.mean([t['processing_speed_x'] for t in clip_timings]) if clip_timings else 0
            }
            
        except Exception as e:
            logger.error(f"CLIP benchmark failed: {e}")
            embedder_results['clip'] = {'error': str(e)}
        
        return embedder_results
    
    def benchmark_pipeline_operations(self) -> Dict[str, Any]:
        """Benchmark individual pipeline operations"""
        print("⚙️  Benchmarking Pipeline Operations...")
        
        ops_results = {}
        test_video = 'assets/test_30min.mp4'
        
        if not os.path.exists(test_video):
            return {'error': 'Test video not found'}
        
        # Silence Detection Benchmark
        try:
            from src.ops.silence import SilenceRemover
            
            print("  Testing silence detection...")
            silence_remover = SilenceRemover()
            
            cuts_data, duration = self._measure_duration(
                silence_remover.remove_silence,
                test_video
            )
            
            video_duration = 1800.0  # 30 min
            sec_per_min = (duration / (video_duration / 60)) if video_duration > 0 else 0
            
            ops_results['silence'] = {
                'duration': duration,
                'sec_per_min': sec_per_min,
                'segments_found': len(cuts_data[0].get('keep_windows', [])),
                'passes_gate': sec_per_min <= 0.5
            }
            
        except Exception as e:
            logger.error(f"Silence benchmark failed: {e}")
            ops_results['silence'] = {'error': str(e)}
        
        # B-roll Selection Benchmark
        try:
            from src.broll.selector import BrollSelector
            
            print("  Testing B-roll selection...")
            broll_selector = BrollSelector()
            
            selection_data, duration = self._measure_duration(
                broll_selector.select_broll,
                test_video
            )
            
            ops_results['broll'] = {
                'duration': duration,
                'selections_count': len(selection_data[0].get('selections', [])) if isinstance(selection_data[0], dict) else 0,
                'match_rate': selection_data[0].get('match_rate', 0.0) if isinstance(selection_data[0], dict) else 0.0
            }
            
        except Exception as e:
            logger.error(f"B-roll benchmark failed: {e}")
            ops_results['broll'] = {'error': str(e)}
        
        return ops_results
    
    def benchmark_api_performance(self) -> Dict[str, Any]:
        """Benchmark API endpoint performance"""
        print("🌐 Benchmarking API Performance...")
        
        api_results = {}
        base_url = "http://localhost:8000"
        
        # Check if server is running
        try:
            import requests
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                return {'error': 'Backend server not responding'}
        except Exception:
            return {'error': 'Backend server not running'}
        
        # Benchmark key endpoints
        endpoints = [
            {"path": "/health", "method": "GET", "auth": False},
            {"path": "/api/projects", "method": "GET", "auth": True},
            {"path": "/api/telemetry/metrics", "method": "GET", "auth": True}
        ]
        
        for endpoint in endpoints:
            path = endpoint["path"]
            print(f"  Testing {path}...")
            
            timings = []
            errors = 0
            
            for _ in range(50):  # 50 samples for statistical significance
                start = time.time()
                try:
                    headers = {"x-api-key": "test"} if endpoint["auth"] else {}
                    response = requests.get(f"{base_url}{path}", headers=headers, timeout=5)
                    elapsed_ms = (time.time() - start) * 1000
                    
                    if response.status_code == 200:
                        timings.append(elapsed_ms)
                    else:
                        errors += 1
                        
                except Exception:
                    errors += 1
                    
                time.sleep(0.02)  # Small delay between requests
            
            if timings:
                p50 = statistics.quantiles(timings, n=2)[0]
                p99 = statistics.quantiles(timings, n=100)[98] if len(timings) >= 100 else max(timings)
                avg = statistics.mean(timings)
                
                api_results[path] = {
                    'avg_ms': round(avg, 2),
                    'p50_ms': round(p50, 2),
                    'p99_ms': round(p99, 2),
                    'samples': len(timings),
                    'errors': errors,
                    'passes_gate': p99 <= 200
                }
            else:
                api_results[path] = {
                    'error': f'No successful requests, {errors} errors'
                }
        
        return api_results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Monitor memory usage during operations"""
        print("🧮 Benchmarking Memory Usage...")
        
        initial_memory = self._get_memory_usage()
        peak_memory = initial_memory
        
        # Test memory during embedder initialization
        try:
            from src.embedders.vjepa_embedder import VJEPAEmbedder
            from src.embedders.clip_embedder import CLIPEmbedder
            
            # V-JEPA memory test
            vjepa = VJEPAEmbedder(memory_safe_mode=True)
            after_vjepa = self._get_memory_usage()
            peak_memory = max(peak_memory, after_vjepa)
            
            # CLIP memory test
            clip = CLIPEmbedder()
            after_clip = self._get_memory_usage()
            peak_memory = max(peak_memory, after_clip)
            
            del vjepa, clip  # Cleanup
            
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
        
        return {
            'initial_gb': round(initial_memory, 2),
            'peak_gb': round(peak_memory, 2),
            'delta_gb': round(peak_memory - initial_memory, 2),
            'passes_gate': peak_memory <= 16.0
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("🏁 AutoResolve v3.2 - Performance Benchmark Suite")
        print("=" * 60)
        
        # Collect system info
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'python_version': sys.version.split()[0]
        }
        
        benchmark_results = {
            'timestamp': time.time(),
            'system_info': system_info,
            'embedders': self.benchmark_embedders(),
            'pipeline_ops': self.benchmark_pipeline_operations(),
            'api_performance': self.benchmark_api_performance(),
            'memory_usage': self.benchmark_memory_usage()
        }
        
        # Calculate overall compliance
        gates_passing = []
        
        # Check each benchmark against gates
        if 'vjepa' in benchmark_results['embedders'] and 'avg_sec_per_min' in benchmark_results['embedders']['vjepa']:
            gates_passing.append(benchmark_results['embedders']['vjepa']['avg_sec_per_min'] <= 5.0)
        
        if 'silence' in benchmark_results['pipeline_ops'] and 'passes_gate' in benchmark_results['pipeline_ops']['silence']:
            gates_passing.append(benchmark_results['pipeline_ops']['silence']['passes_gate'])
        
        if benchmark_results['memory_usage']['passes_gate']:
            gates_passing.append(True)
        
        # Check API performance gates
        api_gates_pass = True
        for endpoint_data in benchmark_results['api_performance'].values():
            if isinstance(endpoint_data, dict) and 'passes_gate' in endpoint_data:
                api_gates_pass &= endpoint_data['passes_gate']
        gates_passing.append(api_gates_pass)
        
        overall_compliance = (sum(gates_passing) / len(gates_passing) * 100) if gates_passing else 0
        
        benchmark_results['compliance'] = {
            'overall_percent': round(overall_compliance, 1),
            'gates_passing': sum(gates_passing),
            'total_gates': len(gates_passing)
        }
        
        # Save results
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        # Print summary
        print("=" * 60)
        print(f"📊 Benchmark Summary:")
        print(f"   Overall Compliance: {overall_compliance:.1f}%")
        print(f"   Gates Passing: {sum(gates_passing)}/{len(gates_passing)}")
        print(f"   Peak Memory: {benchmark_results['memory_usage']['peak_gb']:.2f}GB")
        
        if 'vjepa' in benchmark_results['embedders']:
            vjepa_speed = benchmark_results['embedders']['vjepa'].get('avg_speed_x', 0)
            print(f"   V-JEPA Speed: {vjepa_speed:.1f}x realtime")
        
        print(f"   Results saved: artifacts/benchmark_results.json")
        print("=" * 60)
        
        return benchmark_results

def main():
    """Run benchmark suite from command line"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python scripts/benchmark_suite.py [--quick]")
        print("  --quick: Run quick benchmarks only")
        return
    
    quick_mode = "--quick" in sys.argv
    
    suite = BenchmarkSuite()
    
    if quick_mode:
        print("⚡ Quick Benchmark Mode")
        # Only run memory and basic API tests
        results = {
            'memory_usage': suite.benchmark_memory_usage(),
            'api_performance': suite.benchmark_api_performance()
        }
    else:
        results = suite.run_full_benchmark()
    
    # Return appropriate exit code
    compliance = results.get('compliance', {}).get('overall_percent', 0)
    if compliance >= 90:
        print("✅ Benchmarks PASSED")
        sys.exit(0)
    else:
        print("❌ Benchmarks FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()