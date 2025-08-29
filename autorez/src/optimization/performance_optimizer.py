"""
Performance Optimization Module for AutoResolve V3.0
Implements caching, batching, and parallel processing optimizations
"""

import asyncio
import concurrent.futures
import functools
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from multiprocessing import Pool, cpu_count
import torch

logger = logging.getLogger(__name__)

class CacheManager:
    """Intelligent caching system for processed data"""
    
    def __init__(self, cache_dir="/tmp/autoresolve_cache", max_size_gb=5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.cache_index = {}
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            with open(index_file) as f:
                self.cache_index = json.load(f)
    
    def _save_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f)
    
    def get_cache_key(self, operation: str, params: Dict) -> str:
        """Generate cache key from operation and parameters"""
        key_str = f"{operation}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached result"""
        if key not in self.cache_index:
            return None
        
        cache_file = self.cache_dir / f"{key}.pkl"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.cache_index[key]["last_accessed"] = time.time()
                return data
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Store result in cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.cache_index[key] = {
                "created": time.time(),
                "last_accessed": time.time(),
                "ttl": ttl_seconds,
                "size": cache_file.stat().st_size
            }
            self._save_index()
            self._cleanup_if_needed()
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def _cleanup_if_needed(self):
        """Remove old entries if cache size exceeds limit"""
        total_size = sum(entry["size"] for entry in self.cache_index.values())
        
        if total_size > self.max_size_bytes:
            # Sort by last accessed time (LRU)
            sorted_keys = sorted(
                self.cache_index.keys(),
                key=lambda k: self.cache_index[k]["last_accessed"]
            )
            
            # Remove oldest entries
            while total_size > self.max_size_bytes * 0.8:  # Keep 80% capacity
                if not sorted_keys:
                    break
                    
                old_key = sorted_keys.pop(0)
                cache_file = self.cache_dir / f"{old_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                
                total_size -= self.cache_index[old_key]["size"]
                del self.cache_index[old_key]
            
            self._save_index()


class ParallelProcessor:
    """Parallel processing optimizations"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or cpu_count()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
    
    async def process_batch_async(self, func, items: List, batch_size: int = 10):
        """Process items in parallel batches asynchronously"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            batch_futures = [
                asyncio.get_event_loop().run_in_executor(self.executor, func, item)
                for item in batch
            ]
            batch_results = await asyncio.gather(*batch_futures)
            results.extend(batch_results)
        
        return results
    
    def process_batch_sync(self, func, items: List, batch_size: int = 10):
        """Process items in parallel batches synchronously"""
        with Pool(processes=self.max_workers) as pool:
            return pool.map(func, items)


class GPUAccelerator:
    """GPU acceleration for neural operations"""
    
    def __init__(self):
        self.device = self._get_best_device()
        logger.info(f"GPU Accelerator using device: {self.device}")
    
    def _get_best_device(self) -> torch.device:
        """Select best available device (MPS > CUDA > CPU)"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def batch_embeddings(self, embedder, items: List, batch_size: int = 32) -> np.ndarray:
        """Compute embeddings in optimized batches on GPU"""
        embeddings = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            
            # Move to GPU
            if hasattr(embedder, 'to'):
                embedder = embedder.to(self.device)
            
            # Process batch
            with torch.no_grad():
                batch_embeddings = embedder.encode(batch)
                
                # Move back to CPU for storage
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def optimize_memory(self):
        """Free GPU memory"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            # MPS doesn't have explicit cache clearing
            pass


class VideoStreamProcessor:
    """Optimized video stream processing"""
    
    def __init__(self):
        self.frame_cache = {}
        self.cache_size = 1000  # Maximum cached frames
    
    def process_video_chunks(self, video_path: str, chunk_duration: float = 10.0):
        """Process video in efficient chunks"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        chunk_frames = int(chunk_duration * fps)
        
        chunks_processed = 0
        
        while True:
            frames = []
            for _ in range(chunk_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if not frames:
                break
            
            # Process chunk
            yield np.array(frames), chunks_processed * chunk_duration
            chunks_processed += 1
            
            # Clear old cache entries
            if len(self.frame_cache) > self.cache_size:
                # Remove oldest 10%
                keys_to_remove = list(self.frame_cache.keys())[:self.cache_size // 10]
                for key in keys_to_remove:
                    del self.frame_cache[key]
        
        cap.release()


class OptimizedPipeline:
    """Main optimized pipeline with all performance enhancements"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.parallel = ParallelProcessor()
        self.gpu = GPUAccelerator()
        self.video_processor = VideoStreamProcessor()
    
    async def process_video_optimized(self, video_path: str) -> Dict:
        """Process video with all optimizations"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self.cache.get_cache_key("pipeline", {"video": video_path})
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info("Using cached pipeline result")
            return cached_result
        
        result = {}
        
        # Process video in chunks
        chunks = []
        for chunk, timestamp in self.video_processor.process_video_chunks(video_path):
            chunks.append((chunk, timestamp))
        
        # Parallel processing of chunks
        async def process_chunk(chunk_data):
            chunk, timestamp = chunk_data
            # Simulate processing (replace with actual processing)
            return {
                "timestamp": timestamp,
                "features": np.random.randn(512)  # Placeholder
            }
        
        chunk_results = await self.parallel.process_batch_async(
            process_chunk, 
            chunks,
            batch_size=4
        )
        
        result["chunks"] = chunk_results
        result["processing_time"] = time.time() - start_time
        
        # Cache the result
        self.cache.set(cache_key, result, ttl_seconds=3600)
        
        # Clean up GPU memory
        self.gpu.optimize_memory()
        
        return result


class PerformanceMonitor:
    """Monitor and report performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "processing_times": [],
            "memory_usage": [],
            "gpu_usage": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def record_processing_time(self, operation: str, duration: float):
        """Record operation timing"""
        self.metrics["processing_times"].append({
            "operation": operation,
            "duration": duration,
            "timestamp": time.time()
        })
    
    def record_memory_usage(self):
        """Record current memory usage"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics["memory_usage"].append({
            "memory_mb": memory_mb,
            "timestamp": time.time()
        })
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        if not self.metrics["processing_times"]:
            return {"error": "No metrics collected"}
        
        avg_time = np.mean([m["duration"] for m in self.metrics["processing_times"]])
        max_memory = max([m["memory_mb"] for m in self.metrics["memory_usage"]]) if self.metrics["memory_usage"] else 0
        
        cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        cache_hit_rate = self.metrics["cache_hits"] / cache_total if cache_total > 0 else 0
        
        return {
            "average_processing_time": avg_time,
            "max_memory_mb": max_memory,
            "cache_hit_rate": cache_hit_rate,
            "total_operations": len(self.metrics["processing_times"])
        }


# Singleton instances
cache_manager = CacheManager()
performance_monitor = PerformanceMonitor()
gpu_accelerator = GPUAccelerator()

def optimized_decorator(cache_ttl: int = 3600):
    """Decorator to add caching to any function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_manager.get_cache_key(
                func.__name__,
                {"args": str(args), "kwargs": str(kwargs)}
            )
            
            # Check cache
            cached = cache_manager.get(cache_key)
            if cached is not None:
                performance_monitor.metrics["cache_hits"] += 1
                return cached
            
            performance_monitor.metrics["cache_misses"] += 1
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record metrics
            performance_monitor.record_processing_time(func.__name__, duration)
            performance_monitor.record_memory_usage()
            
            # Cache result
            cache_manager.set(cache_key, result, ttl_seconds=cache_ttl)
            
            return result
        return wrapper
    return decorator