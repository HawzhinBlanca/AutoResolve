#!/usr/bin/env python3
"""Test basic operations on 30-minute video"""

import os
import time
import psutil
import subprocess
from src.scoring.broll_quality import VideoQualityAnalyzer, calculate_broll_quality_score

def test_30min_video():
    """Validate 30-minute test video for Day 2 testing"""
    video_path = "assets/test_30min.mp4"
    
    if not os.path.exists(video_path):
        return False
    print("="*50)
    
    # 1. Verify video properties
    result = subprocess.run([
        "ffprobe", "-v", "error", 
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json", video_path
    ], capture_output=True, text=True)
    
    import json
    props = json.loads(result.stdout)["format"]
    duration = float(props["duration"])
    size_mb = int(props["size"]) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")
    # Validate duration is ~30 minutes
    if 29 <= duration/60 <= 31:
    else:
        return False
    
    # 2. Test quality scoring on a segment
    analyzer = VideoQualityAnalyzer()
    
    # Test at different points in the video
    test_segments = [
        {"t0": 60, "t1": 70, "name": "1 minute mark"},
        {"t0": 900, "t1": 910, "name": "15 minute mark"},  
        {"t0": 1700, "t1": 1710, "name": "28 minute mark"}
    ]
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024**3)
    
    for segment in test_segments:
        start_time = time.time()
        
        try:
            analyzer.analyze_segment(segment, video_path)
            score = calculate_broll_quality_score(segment, video_path)
            
            elapsed = time.time() - start_time
            current_memory = process.memory_info().rss / (1024**3)
            print(f"    Quality score: {score:.3f}")
            print(f"    Memory usage: {current_memory:.2f} GB")
            
            # Validate score is in range
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"
            
        except Exception as e:
            return False
    
    # 3. Check memory didn't explode
    final_memory = process.memory_info().rss / (1024**3)
    memory_increase = final_memory - initial_memory
    print(f"  Initial: {initial_memory:.2f} GB")
    print(f"  Increase: {memory_increase:.2f} GB")
    
    if final_memory < 2.0:  # Less than 2GB for basic ops
    else:
    # 4. Test codec compatibility
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height",
        "-of", "json", video_path
    ], capture_output=True, text=True)
    
    stream = json.loads(result.stdout)["streams"][0]
    print(f"  Resolution: {stream['width']}x{stream['height']}")
    
    if stream['codec_name'] == 'h264':
    else:
    print("\n" + "="*50)
    return True

if __name__ == "__main__":
    success = test_30min_video()
    exit(0 if success else 1)