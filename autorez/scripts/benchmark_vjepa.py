#!/usr/bin/env python
"""
Benchmark script for V-JEPA embedder performance.
Measures processing time, memory usage, and throughput.
"""

import time
import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embedders.vjepa_embedder import VJEPAEmbedder
from src.utils.memory import emit_metrics, rss_gb
from src.utils.memory_guard import MemoryGuard


def benchmark_vjepa(video_path: str, iterations: int = 3, use_memory_guard: bool = True):
    """
    Benchmark V-JEPA embedder with multiple iterations.
    
    Args:
        video_path: Path to video file for benchmarking
        iterations: Number of benchmark iterations
        use_memory_guard: Whether to use memory protection
        
    Returns:
        Dict containing benchmark results
    """
    embedder = VJEPAEmbedder()
    results = []
    
    # Optional memory guard
    memory_guard = MemoryGuard(max_gb=16) if use_memory_guard else None
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        
        # Get current memory params if using guard
        params = memory_guard.get_current_params() if memory_guard else {
            'fps': 1.0,
            'window': 16,
            'crop': 256
        }
        
        # Record start state
        start_time = time.time()
        start_memory = rss_gb()
        
        try:
            # Run embedding with params
            if memory_guard:
                with memory_guard.protected_execution(f"benchmark_iteration_{i}"):
                    segments, meta = embedder.embed_segments(
                        video_path, 
                        fps=params['fps'], 
                        window=params['window'],
                        crop=params['crop'],
                        strategy="temp_attn"
                    )
            else:
                segments, meta = embedder.embed_segments(
                    video_path, 
                    fps=1.0, 
                    window=16,
                    crop=256,
                    strategy="temp_attn"
                )
            
            # Record end state
            elapsed = time.time() - start_time
            peak_memory = rss_gb()
            memory_delta = peak_memory - start_memory
            
            # Calculate metrics
            video_duration = meta.get('video_duration', elapsed)
            sec_per_min = (elapsed / video_duration) * 60 if video_duration > 0 else 0
            
            iteration_result = {
                'iteration': i + 1,
                'segments': len(segments),
                'elapsed': elapsed,
                'sec_per_min': sec_per_min,
                'start_memory_gb': start_memory,
                'peak_memory_gb': peak_memory,
                'memory_delta_gb': memory_delta,
                'params': params,
                **meta
            }
            
            results.append(iteration_result)
            
            # Log to metrics
            emit_metrics('benchmark_vjepa', iteration_result)
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            results.append({
                'iteration': i + 1,
                'error': str(e),
                'params': params
            })
    
    # Calculate aggregate statistics
    successful_runs = [r for r in results if 'error' not in r]
    
    if successful_runs:
        avg_time = sum(r['elapsed'] for r in successful_runs) / len(successful_runs)
        avg_memory = sum(r['peak_memory_gb'] for r in successful_runs) / len(successful_runs)
        avg_sec_per_min = sum(r['sec_per_min'] for r in successful_runs) / len(successful_runs)
        
        summary = {
            'avg_elapsed': avg_time,
            'avg_memory_gb': avg_memory,
            'avg_sec_per_min': avg_sec_per_min,
            'successful_iterations': len(successful_runs),
            'total_iterations': iterations,
            'results': results
        }
    else:
        summary = {
            'error': 'All iterations failed',
            'results': results
        }
    
    # Final metrics emission
    emit_metrics('benchmark_vjepa_summary', summary)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Benchmark V-JEPA embedder')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    parser.add_argument('--no-memory-guard', action='store_true', 
                       help='Disable memory guard protection')
    parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print(f"Benchmarking V-JEPA with {args.iterations} iterations...")
    print(f"Video: {args.video}")
    print(f"Memory guard: {'disabled' if args.no_memory_guard else 'enabled'}")
    print("-" * 50)
    
    results = benchmark_vjepa(
        args.video, 
        args.iterations,
        use_memory_guard=not args.no_memory_guard
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    if 'error' not in results:
        print(f"Successful runs: {results['successful_iterations']}/{results['total_iterations']}")
        print(f"Average time: {results['avg_elapsed']:.2f}s")
        print(f"Average memory: {results['avg_memory_gb']:.2f}GB")
        print(f"Average sec/min: {results['avg_sec_per_min']:.2f}")
    else:
        print(f"Benchmark failed: {results['error']}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Return exit code based on success
    sys.exit(0 if 'error' not in results else 1)


if __name__ == "__main__":
    main()