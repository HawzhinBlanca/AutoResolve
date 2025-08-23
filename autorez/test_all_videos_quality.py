#!/usr/bin/env python3
"""Test quality scoring on all pilot videos"""

import glob
import json
from pathlib import Path
from src.scoring.broll_quality import VideoQualityAnalyzer, calculate_broll_quality_score

def test_all_videos():
    """Test quality analyzer on all pilot videos"""
    analyzer = VideoQualityAnalyzer()
    videos = glob.glob("assets/pilots/*.mp4")
    
    # Skip very large videos for now
    skip_videos = ["nature_documentary.mp4", "sintel_animation.mp4"]
    videos = [v for v in videos if not any(skip in v for skip in skip_videos)]
    
    results = []
    for video_path in sorted(videos):
        video_name = Path(video_path).name
        # Test a segment from middle of video
        segment = {
            "t0": 10.0,
            "t1": 20.0,
            "video": video_name
        }
        
        try:
            # Get detailed metrics
            metrics = analyzer.analyze_segment(segment, video_path)
            
            # Also test the main function
            overall_score = calculate_broll_quality_score(segment, video_path)
            
            # Validate ranges
            for key, value in metrics.items():
                assert 0.0 <= value <= 1.0, f"Score {key}={value} out of range for {video_name}"
            
            results.append({
                "video": video_name,
                "metrics": metrics,
                "overall": overall_score
            })
            print(f"    - Sharpness: {metrics['sharpness']:.3f}")
            print(f"    - Composition: {metrics['composition']:.3f}")
            print(f"    - Exposure: {metrics['exposure']:.3f}")
            
        except Exception as e:
            results.append({
                "video": video_name,
                "error": str(e)
            })
    
    # Save results
    output_path = "artifacts/quality_test_results.json"
    Path("artifacts").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print(f"SUMMARY: Tested {len(results)} videos")
    successful = [r for r in results if "error" not in r]
    print(f"  ✗ Failed: {len(results) - len(successful)}")
    
    if successful:
        avg_score = sum(r["overall"] for r in successful) / len(successful)
        # Check all scores are in valid range
        all_valid = all(
            all(0.0 <= v <= 1.0 for v in r["metrics"].values())
            for r in successful
        )
        if all_valid:
        else:
    print(f"Results saved to: {output_path}")
    return results

if __name__ == "__main__":
    test_all_videos()