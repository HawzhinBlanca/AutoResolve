import logging
import time
import json
import os
import argparse
import av
import numpy as np
from pathlib import Path
from src.director.narrative import detect_story_beats
from src.director.emotion import analyze_tension
from src.director.rhythm import detect_pace
from src.director.continuity import detect_shot_boundaries
from src.director.emphasis import track_saliency
from src.utils.memory import rss_gb, emit_metrics
from src.ops.openrouter import get_client

logger = logging.getLogger(__name__)

# Blueprint3 Director Module - Creative Director
# Orchestration of all director modules

def analyze_video(video_path: str, modules=None, fps=None, window=None, config=None):
    """
    Orchestrate all director modules for comprehensive video analysis
    
    Args:
        video_path: Path to video file
        modules: List of modules to run (default: all)
        fps: Frames per second for analysis
        window: Window size for temporal segments
        config: Configuration object (optional, for OpenRouter)
        
    Returns:
        Dictionary with results from all requested modules
    """
    if modules is None:
        modules = ["narrative", "emotion", "rhythm", "continuity", "emphasis"]
    
    results = {}
    timings = {}
    
    # Get video duration for performance calculation
    try:
        with av.open(video_path) as container:
            duration_s = container.duration / 1000000.0 if container.duration else 60.0
    except Exception:
        duration_s = 60.0  # Default to 1 minute
    
    duration_min = duration_s / 60.0
    
    # Run each requested module
    if "narrative" in modules:
        start = time.time()
        try:
            results["narrative"] = detect_story_beats(video_path, fps=fps, window=window)
            elapsed = time.time() - start
            timings["narrative"] = {
                "elapsed_s": elapsed,
                "sec_per_min": elapsed / duration_min
            }
        except Exception as e:
            results["narrative"] = {"error": str(e)}
            timings["narrative"] = {"error": True}
    
    if "emotion" in modules:
        start = time.time()
        try:
            results["emotion"] = analyze_tension(video_path, fps=fps, window=window)
            elapsed = time.time() - start
            timings["emotion"] = {
                "elapsed_s": elapsed,
                "sec_per_min": elapsed / duration_min
            }
        except Exception as e:
            results["emotion"] = {"error": str(e)}
            timings["emotion"] = {"error": True}
    
    if "rhythm" in modules:
        start = time.time()
        try:
            results["rhythm"] = detect_pace(video_path, fps=fps, window=window)
            elapsed = time.time() - start
            timings["rhythm"] = {
                "elapsed_s": elapsed,
                "sec_per_min": elapsed / duration_min
            }
        except Exception as e:
            results["rhythm"] = {"error": str(e)}
            timings["rhythm"] = {"error": True}
    
    if "continuity" in modules:
        start = time.time()
        try:
            results["continuity"] = detect_shot_boundaries(video_path, fps=fps, window=window)
            elapsed = time.time() - start
            timings["continuity"] = {
                "elapsed_s": elapsed,
                "sec_per_min": elapsed / duration_min
            }
        except Exception as e:
            results["continuity"] = {"error": str(e)}
            timings["continuity"] = {"error": True}
    
    if "emphasis" in modules:
        start = time.time()
        try:
            results["emphasis"] = track_saliency(video_path, fps=fps, window=window)
            elapsed = time.time() - start
            timings["emphasis"] = {
                "elapsed_s": elapsed,
                "sec_per_min": elapsed / duration_min
            }
        except Exception as e:
            results["emphasis"] = {"error": str(e)}
            timings["emphasis"] = {"error": True}
    
    # Calculate aggregate performance
    total_elapsed = sum(t.get("elapsed_s", 0) for t in timings.values() if "error" not in t)
    total_sec_per_min = total_elapsed / duration_min if duration_min > 0 else 0
    
    # Add metadata
    results["_metadata"] = {
        "video_path": video_path,
        "modules_run": modules,
        "timings": timings,
        "total_elapsed_s": total_elapsed,
        "total_sec_per_min": total_sec_per_min,
        "peak_rss_gb": rss_gb(),
        "fps": fps,
        "window": window
    }
    
    # Emit telemetry
    emit_metrics("director_analysis", {
        "modules": len(modules),
        "total_elapsed_s": total_elapsed,
        "sec_per_min": total_sec_per_min,
        "peak_rss_gb": rss_gb()
    })
    
    # NEW: OpenRouter augmentation (if enabled)
    if config and config.get('openrouter', 'enabled', fallback='false').lower() == 'true':
        orc = get_client(config)
        
        # Prepare compact summary for augmentation
        summary = {
            'duration_s': duration_s,
            'modules_run': modules
        }
        
        # Add key data from each module
        if "narrative" in results and "beats" in results["narrative"]:
            summary['narrative_beats'] = len(results["narrative"]["beats"])
        if "emotion" in results and "tension_peaks" in results["emotion"]:
            summary['tension_peaks'] = results["emotion"]["tension_peaks"][:10]
        if "rhythm" in results and "peaks" in results["rhythm"]:
            summary['rhythm_peaks'] = results["rhythm"]["peaks"][:10]
        
        # Get narrative beats using OpenRouter
        enhanced_beats = orc.json_chat(
            model=orc.narrative_model,
            system="Label narrative beats using three-act structure",
            user=json.dumps(summary)
        )
        
        if not enhanced_beats.get('_skipped'):
            results['narrative_beats'] = enhanced_beats.get('beats', [])
            results['openrouter_enhanced'] = True
    
    # Save results to artifacts
    os.makedirs("artifacts", exist_ok=True)
    results_file = "artifacts/director_results.json"
    with open(results_file, "w") as f:
        # Convert dataclass to dict for JSON serialization
        json_results = {}
        for key, value in results.items():
            if hasattr(value, "__dict__"):
                json_results[key] = value.__dict__
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2, default=str)
    
    return results

# Alias for compatibility with hybrid_eval.py
analyze_footage = analyze_video

def make_creative_decisions(results):
    """Make creative decisions based on real analysis results"""
    import numpy as np
    
    decisions = []
    
    # Process real emotion data with statistical analysis
    if "emotion" in results and isinstance(results["emotion"], dict):
        # Extract tension data - could be peaks or curve
        tension_peaks = results["emotion"].get("tension_peaks", [])
        tension_curve = results["emotion"].get("tension_curve", [])
        
        if tension_curve:
            # Analyze the full tension curve statistically
            times = [t[0] for t in tension_curve]
            values = np.array([t[1] for t in tension_curve])
            
            if len(values) > 0:
                mean_tension = np.mean(values)
                std_tension = np.std(values)
                
                # Find statistically significant peaks
                for i, (time, value) in enumerate(tension_curve):
                    z_score = (value - mean_tension) / std_tension if std_tension > 0 else 0
                    
                    if z_score > 2.0:  # Very high tension (2+ std deviations)
                        decisions.append({
                            "time": float(time),
                            "action": "slow_motion",
                            "duration": min(2.0, 0.5 * z_score),
                            "reason": "extreme_tension_peak",
                            "confidence": min(1.0, z_score / 3.0),
                            "z_score": float(z_score)
                        })
                    elif z_score > 1.5:  # High tension
                        decisions.append({
                            "time": float(time),
                            "action": "emphasis",
                            "intensity": "high",
                            "reason": "tension_peak",
                            "confidence": (z_score - 1.5) / 0.5,
                            "z_score": float(z_score)
                        })
        elif tension_peaks:
            # Process discrete peaks
            for peak in tension_peaks:
                if isinstance(peak, (list, tuple)) and len(peak) >= 2:
                    time, intensity = peak[0], peak[1]
                    decisions.append({
                        "time": float(time),
                        "action": "emphasis",
                        "intensity": float(intensity),
                        "reason": "tension_peak",
                        "confidence": min(1.0, float(intensity))
                    })
    
    # Process real rhythm data with beat pattern analysis
    if "rhythm" in results and isinstance(results["rhythm"], dict):
        peaks = results["rhythm"].get("peaks", [])
        beat_times = results["rhythm"].get("beat_times", [])
        tempo = results["rhythm"].get("tempo", 0)
        
        if beat_times and len(beat_times) > 1:
            # Analyze beat intervals for rhythm patterns
            intervals = np.diff(beat_times)
            
            for i in range(1, len(beat_times)):
                interval = beat_times[i] - beat_times[i-1]
                
                # Fast cutting for rapid beats
                if interval < 0.5 and tempo > 120:
                    decisions.append({
                        "time": float(beat_times[i]),
                        "action": "quick_cut",
                        "interval": float(interval),
                        "reason": "fast_rhythm",
                        "tempo": float(tempo),
                        "confidence": 0.8
                    })
                # Hold shots for slow beats
                elif interval > 2.0 and tempo < 80:
                    decisions.append({
                        "time": float(beat_times[i]),
                        "action": "hold_shot",
                        "duration": float(interval),
                        "reason": "slow_rhythm",
                        "tempo": float(tempo),
                        "confidence": 0.7
                    })
                # Sync cuts to beat
                elif 0.5 <= interval <= 2.0:
                    decisions.append({
                        "time": float(beat_times[i]),
                        "action": "beat_cut",
                        "reason": "rhythm_sync",
                        "tempo": float(tempo),
                        "confidence": 0.9
                    })
    
    # Process continuity data for scene transitions
    if "continuity" in results and isinstance(results["continuity"], dict):
        cuts = results["continuity"].get("cuts", [])
        transitions = results["continuity"].get("transitions", [])
        
        for cut_time in cuts:
            decisions.append({
                "time": float(cut_time),
                "action": "scene_transition",
                "type": "cut",
                "reason": "shot_boundary",
                "confidence": 1.0
            })
    
    # Process emphasis/saliency data
    if "emphasis" in results and isinstance(results["emphasis"], dict):
        saliency_peaks = results["emphasis"].get("peaks", [])
        
        for peak in saliency_peaks:
            if isinstance(peak, (list, tuple)) and len(peak) >= 2:
                time, saliency = peak[0], peak[1]
                if saliency > 0.7:  # High saliency threshold
                    decisions.append({
                        "time": float(time),
                        "action": "zoom_in",
                        "saliency": float(saliency),
                        "reason": "high_visual_interest",
                        "confidence": float(saliency)
                    })
    
    # Sort decisions by time and remove duplicates
    decisions.sort(key=lambda x: x["time"])
    
    # Merge nearby decisions (within 0.5 seconds)
    if decisions:
        merged = [decisions[0]]
        for decision in decisions[1:]:
            if decision["time"] - merged[-1]["time"] > 0.5:
                merged.append(decision)
            elif decision["confidence"] > merged[-1]["confidence"]:
                merged[-1] = decision  # Replace with higher confidence decision
        decisions = merged
    
    return decisions

def evaluate_director_quality(results, annotations=None):
    """
    Evaluate Director module quality against annotations
    
    Args:
        results: Director analysis results
        annotations: Ground truth annotations (optional)
        
    Returns:
        Quality metrics including F1@IoU scores
    """
    metrics = {}
    
    # Check performance gates
    metadata = results.get("_metadata", {})
    total_sec_per_min = metadata.get("total_sec_per_min", 999)
    peak_rss = metadata.get("peak_rss_gb", 999)
    
    metrics["performance"] = {
        "sec_per_min": total_sec_per_min,
        "target": "≤7.5",
        "pass": total_sec_per_min <= 7.5
    }
    
    metrics["memory"] = {
        "peak_rss_gb": peak_rss,
        "target": "<16",
        "pass": peak_rss < 16
    }
    
    # If we have annotations, calculate F1@IoU
    if annotations:
        # Compute F1 at IoU≥0.5 between predicted events and annotated intervals
        def iou(a, b):
            a0, a1 = float(a[0]), float(a[1])
            b0, b1 = float(b[0]), float(b[1])
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0)
            return (inter / union) if union > 0 else 0.0

        # Build predicted intervals from available module outputs
        predicted = []
        # Use shot boundaries if available
        if "continuity" in results and isinstance(results["continuity"], dict):
            cuts = results["continuity"].get("cuts") or []
            # Treat cuts as zero-length events expanded to small intervals
            for c in cuts:
                t = float(c)
                predicted.append([max(0.0, t - 0.25), t + 0.25])
        # Fallback to narrative beats if present
        if not predicted and "narrative" in results and isinstance(results["narrative"], dict):
            beats = results["narrative"].get("beats") or []
            for b in beats:
                t = float(b)
                predicted.append([max(0.0, t - 0.25), t + 0.25])

        gold = []
        ann_intervals = annotations.get("intervals") if isinstance(annotations, dict) else annotations
        if isinstance(ann_intervals, list):
            for it in ann_intervals:
                try:
                    gold.append([float(it[0]), float(it[1])])
                except Exception:
                    continue

        tp = 0
        matched = set()
        for p in predicted:
            for gi, g in enumerate(gold):
                if gi in matched:
                    continue
                if iou(p, g) >= 0.5:
                    tp += 1
                    matched.add(gi)
                    break
        fp = max(0, len(predicted) - tp)
        fn = max(0, len(gold) - tp)
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics["f1_iou_0.5"] = {
            "value": round(f1, 3),
            "target": "≥0.60",
            "pass": f1 >= 0.60
        }
    
    # Check module completeness
    modules_run = metadata.get("modules_run", [])
    expected_modules = ["narrative", "emotion", "rhythm", "continuity", "emphasis"]
    
    metrics["completeness"] = {
        "modules_run": len(modules_run),
        "expected": len(expected_modules),
        "pass": set(modules_run) == set(expected_modules)
    }
    
    # Overall pass/fail
    all_gates = [
        metrics["performance"]["pass"],
        metrics["memory"]["pass"],
        metrics.get("f1_iou_0.5", {}).get("pass", False),
        metrics.get("pr_auc", {}).get("pass", False)
    ]
    
    metrics["overall"] = {
        "gates_passed": sum(all_gates),
        "total_gates": len(all_gates),
        "pass": all(all_gates)
    }
    
    return metrics

# Blueprint line 231 specifies analyze_footage as the function name
analyze_footage = analyze_video

def continuity_between(shotA, shotB):
    """
    Check real continuity between two shots using actual visual analysis
    Blueprint line 232
    """
    import numpy as np
    
    def extract_shot_features(shot_data):
        """Extract real visual features from shot"""
        features = {}
        
        # Process frame data if available
        if isinstance(shot_data, dict):
            # Color histogram analysis
            if "color_histogram" in shot_data:
                features["color_hist"] = np.array(shot_data["color_histogram"])
            elif "dominant_colors" in shot_data:
                # Convert dominant colors to histogram
                colors = shot_data["dominant_colors"]
                hist = np.zeros(96)  # 32 bins per channel
                for color in colors:
                    if isinstance(color, (list, tuple)) and len(color) >= 3:
                        r_bin = int(color[0] * 32 / 256)
                        g_bin = int(color[1] * 32 / 256) 
                        b_bin = int(color[2] * 32 / 256)
                        hist[r_bin] += 1
                        hist[32 + g_bin] += 1
                        hist[64 + b_bin] += 1
                features["color_hist"] = hist / (np.sum(hist) + 1e-10)
            
            # Motion analysis
            if "motion_vectors" in shot_data:
                vectors = shot_data["motion_vectors"]
                if isinstance(vectors, (list, np.ndarray)) and len(vectors) > 0:
                    features["motion"] = np.mean(np.abs(vectors))
                else:
                    features["motion"] = 0.0
            elif "motion_intensity" in shot_data:
                features["motion"] = float(shot_data["motion_intensity"])
            
            # Object tracking
            if "objects" in shot_data:
                features["objects"] = set(shot_data["objects"])
            elif "detected_items" in shot_data:
                features["objects"] = set(shot_data["detected_items"])
            
            # Lighting analysis
            if "brightness" in shot_data:
                features["brightness"] = float(shot_data["brightness"])
            if "contrast" in shot_data:
                features["contrast"] = float(shot_data["contrast"])
        
        return features
    
    # Extract features from both shots
    features_a = extract_shot_features(shotA)
    features_b = extract_shot_features(shotB)
    
    # Calculate continuity score based on multiple factors
    score = 0.0
    factors = {}
    weights = {"color": 0.35, "motion": 0.25, "objects": 0.25, "lighting": 0.15}
    
    # Color continuity (histogram correlation)
    if "color_hist" in features_a and "color_hist" in features_b:
        hist_a = features_a["color_hist"]
        hist_b = features_b["color_hist"]
        
        # Compute correlation coefficient
        if len(hist_a) == len(hist_b) and len(hist_a) > 0:
            mean_a = np.mean(hist_a)
            mean_b = np.mean(hist_b)
            std_a = np.std(hist_a)
            std_b = np.std(hist_b)
            
            if std_a > 0 and std_b > 0:
                correlation = np.sum((hist_a - mean_a) * (hist_b - mean_b)) / (len(hist_a) * std_a * std_b)
                factors["color"] = float(np.clip(correlation, -1, 1))
                score += factors["color"] * weights["color"]
            else:
                factors["color"] = 0.5  # Neutral if no variation
                score += 0.5 * weights["color"]
    
    # Motion continuity
    if "motion" in features_a and "motion" in features_b:
        motion_a = features_a["motion"]
        motion_b = features_b["motion"]
        
        # Calculate motion similarity (inverse of normalized difference)
        max_motion = max(motion_a, motion_b, 1.0)
        motion_diff = abs(motion_a - motion_b) / max_motion
        motion_score = 1.0 - min(1.0, motion_diff)
        
        factors["motion"] = float(motion_score)
        score += motion_score * weights["motion"]
    
    # Object continuity
    if "objects" in features_a and "objects" in features_b:
        objects_a = features_a["objects"]
        objects_b = features_b["objects"]
        
        if objects_a or objects_b:
            # Jaccard similarity
            intersection = len(objects_a & objects_b)
            union = len(objects_a | objects_b)
            object_score = intersection / union if union > 0 else 0.0
            factors["objects"] = float(object_score)
            score += object_score * weights["objects"]
    
    # Lighting continuity
    lighting_score = 0.0
    lighting_factors = 0
    
    if "brightness" in features_a and "brightness" in features_b:
        brightness_diff = abs(features_a["brightness"] - features_b["brightness"])
        brightness_score = 1.0 - min(1.0, brightness_diff)
        lighting_score += brightness_score
        lighting_factors += 1
    
    if "contrast" in features_a and "contrast" in features_b:
        contrast_diff = abs(features_a["contrast"] - features_b["contrast"])
        contrast_score = 1.0 - min(1.0, contrast_diff)
        lighting_score += contrast_score
        lighting_factors += 1
    
    if lighting_factors > 0:
        factors["lighting"] = float(lighting_score / lighting_factors)
        score += factors["lighting"] * weights["lighting"]
    
    # Normalize score if not all factors were available
    total_weight = sum(weights[k] for k in factors.keys() if k in weights)
    if total_weight > 0:
        score = score / total_weight
    
    return {
        "continuity_score": float(np.clip(score, 0, 1)),
        "match": score > 0.6,
        "factors": factors,
        "confidence": float(len(factors) / 4.0)  # Confidence based on available features
    }


def main():
    """Main entry point for creative director"""
    parser = argparse.ArgumentParser(description='Creative Director Analysis')
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--out', required=True, help='Output JSON path')
    args = parser.parse_args()
    
    # Analyze video
    analysis = analyze_video(args.video)
    decisions = make_creative_decisions(analysis)
    print("\nCreative Decisions:")
    for decision in decisions:
        print(f"- {decision}")
    
    # Save results
    with open(args.out, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"✓ Analysis saved to {args.out}")
    return 0

if __name__ == "__main__":
    """CLI entry point for director-analyze target"""
    parser = argparse.ArgumentParser(description="Analyze video with director modules")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--modules", nargs="+", help="Modules to run")
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_footage(args.video, modules=args.modules)
    
    # Save results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Convert dataclasses to dicts for JSON serialization
    def make_serializable(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (float, np.floating)) and np.isnan(obj):
            return None
        return obj
    
    serializable_results = make_serializable(results)
    
    with open(args.out, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Director analysis saved to {args.out}")
