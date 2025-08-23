#!/usr/bin/env python3
"""Test silence detection accuracy on 30-minute video"""

import time
import json
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SilenceDetectionTest:
    """Test silence detection on 30-minute video"""
    
    def __init__(self):
        self.video_path = "assets/test_30min.mp4"
        self.audio_path = "artifacts/test_30min_audio.wav"
        
    def detect_silence_ffmpeg(self) -> List[Tuple[float, float]]:
        """Detect silence using ffmpeg silencedetect filter"""
        logger.info("Detecting silence with ffmpeg...")
        
        cmd = [
            "ffmpeg", "-i", self.video_path,
            "-af", "silencedetect=noise=-30dB:d=0.5",
            "-f", "null", "-"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse silence regions from stderr
        silence_regions = []
        lines = result.stderr.split('\n')
        
        silence_start = None
        for line in lines:
            if "silence_start:" in line:
                silence_start = float(line.split("silence_start:")[1].strip())
            elif "silence_end:" in line and silence_start is not None:
                silence_end = float(line.split("silence_end:")[1].split()[0])
                silence_regions.append((silence_start, silence_end))
                silence_start = None
        
        return silence_regions
    
    def detect_silence_custom(self, rms_thresh_db: float = -34, min_silence_s: float = 0.35) -> Dict:
        """Custom silence detection using RMS threshold"""
        logger.info(f"Detecting silence with RMS threshold {rms_thresh_db}dB...")
        
        # Extract audio if not exists
        if not Path(self.audio_path).exists():
            cmd = [
                "ffmpeg", "-i", self.video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-y", self.audio_path
            ]
            subprocess.run(cmd, capture_output=True)
        
        # Load audio
        import wave
        with wave.open(self.audio_path, 'rb') as wav:
            frames = wav.readframes(wav.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            sample_rate = wav.getframerate()
        
        # Normalize audio
        audio = audio / 32768.0
        
        # Calculate RMS in windows
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = window_size // 2
        
        rms_values = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_db = 20 * np.log10(rms + 1e-10)
            rms_values.append((i / sample_rate, rms_db))
        
        # Find silence regions
        silence_regions = []
        in_silence = False
        silence_start = None
        
        for t, rms_db in rms_values:
            if rms_db < rms_thresh_db:
                if not in_silence:
                    silence_start = t
                    in_silence = True
            else:
                if in_silence:
                    silence_duration = t - silence_start
                    if silence_duration >= min_silence_s:
                        silence_regions.append((silence_start, t))
                    in_silence = False
        
        # Calculate statistics
        total_silence = sum(end - start for start, end in silence_regions)
        total_duration = len(audio) / sample_rate
        
        return {
            "silence_regions": silence_regions,
            "total_silence_s": total_silence,
            "total_duration_s": total_duration,
            "silence_percentage": (total_silence / total_duration) * 100,
            "num_silence_regions": len(silence_regions)
        }
    
    def test_false_cut_rate(self) -> Dict:
        """Test for false cuts in speech"""
        logger.info("Testing false cut rate...")
        
        # Get silence regions
        result = self.detect_silence_custom()
        silence_regions = result["silence_regions"]
        
        # Simulate speech segments (inverse of silence)
        speech_segments = []
        last_end = 0
        
        for start, end in silence_regions:
            if start > last_end:
                speech_segments.append((last_end, start))
            last_end = end
        
        # Add final segment if needed
        if last_end < result["total_duration_s"]:
            speech_segments.append((last_end, result["total_duration_s"]))
        
        # Check for false cuts (very short speech segments)
        false_cuts = 0
        min_speech_duration = 1.0  # Minimum meaningful speech duration
        
        for start, end in speech_segments:
            duration = end - start
            if 0 < duration < min_speech_duration:
                false_cuts += 1
        
        false_cut_rate = false_cuts / len(speech_segments) if speech_segments else 0
        
        return {
            "total_segments": len(speech_segments),
            "false_cuts": false_cuts,
            "false_cut_rate": false_cut_rate,
            "passes_gate": false_cut_rate <= 0.05
        }
    
    def run_full_test(self) -> Dict:
        """Run complete silence detection test"""
        logger.info("="*50)
        logger.info("SILENCE DETECTION TEST - 30 MINUTE VIDEO")
        logger.info("="*50)
        
        if not Path(self.video_path).exists():
            logger.error(f"Video not found: {self.video_path}")
            return {"error": "Video not found"}
        
        # Test 1: FFmpeg baseline
        logger.info("\nTest 1: FFmpeg silencedetect")
        start_time = time.time()
        ffmpeg_regions = self.detect_silence_ffmpeg()
        ffmpeg_time = time.time() - start_time
        
        logger.info(f"  Found {len(ffmpeg_regions)} silence regions")
        logger.info(f"  Processing time: {ffmpeg_time:.2f}s")
        
        # Test 2: Custom RMS-based detection
        logger.info("\nTest 2: Custom RMS detection")
        start_time = time.time()
        custom_result = self.detect_silence_custom()
        custom_time = time.time() - start_time
        
        logger.info(f"  Found {custom_result['num_silence_regions']} silence regions")
        logger.info(f"  Total silence: {custom_result['total_silence_s']:.1f}s ({custom_result['silence_percentage']:.1f}%)")
        logger.info(f"  Processing time: {custom_time:.2f}s")
        
        # Test 3: False cut rate
        logger.info("\nTest 3: False cut rate analysis")
        false_cut_result = self.test_false_cut_rate()
        
        logger.info(f"  Total segments: {false_cut_result['total_segments']}")
        logger.info(f"  False cuts: {false_cut_result['false_cuts']}")
        logger.info(f"  False cut rate: {false_cut_result['false_cut_rate']:.3f}")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("SILENCE DETECTION SUMMARY")
        logger.info("="*50)
        
        passed = false_cut_result["passes_gate"]
        
        if passed:
            logger.info(f"✅ PASS: False cut rate {false_cut_result['false_cut_rate']:.3f} <= 0.05")
        else:
            logger.error(f"❌ FAIL: False cut rate {false_cut_result['false_cut_rate']:.3f} > 0.05")
        
        # Compile results
        results = {
            "ffmpeg": {
                "regions_found": len(ffmpeg_regions),
                "processing_time_s": ffmpeg_time
            },
            "custom_rms": custom_result,
            "false_cut_analysis": false_cut_result,
            "passed": passed
        }
        
        # Save results
        Path("artifacts").mkdir(exist_ok=True)
        with open("artifacts/silence_detection_test.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to artifacts/silence_detection_test.json")
        
        # Record telemetry
        from src.utils.telemetry import TelemetryCollector
        telemetry = TelemetryCollector()
        telemetry.record_silence_detection(
            total_segments=false_cut_result["total_segments"],
            kept_segments=false_cut_result["total_segments"] - false_cut_result["false_cuts"],
            removed_segments=custom_result["num_silence_regions"],
            false_cuts=false_cut_result["false_cuts"]
        )
        
        return results

def main():
    tester = SilenceDetectionTest()
    results = tester.run_full_test()
    return 0 if results.get("passed", False) else 1

if __name__ == "__main__":
    exit(main())