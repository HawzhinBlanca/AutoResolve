#!/usr/bin/env python3
"""
REAL Video Analysis Pipeline
Actually analyzes video content, not random cuts
"""

import os
import sys
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from scipy.signal import find_peaks
from collections import defaultdict

class RealVideoAnalyzer:
    def __init__(self):
        self.base_dir = Path("/Users/hawzhin/AutoResolve")
        self.source_video = self.base_dir / "test_video_43min.mp4"
        self.output_dir = self.base_dir / "real_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_video_scenes(self):
        """Actually analyze video for scene changes"""
        print("🎬 Analyzing real video scenes...")
        
        cap = cv2.VideoCapture(str(self.source_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"  Video: {fps:.1f} fps, {total_frames} frames, {duration:.1f}s")
        
        # Analyze frames for scene changes
        prev_frame = None
        scene_changes = []
        frame_diffs = []
        
        # Sample every 30 frames (1 second at 30fps)
        sample_rate = int(fps)
        
        for i in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (64, 48))  # Reduce for faster comparison
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray_small)
                diff_score = np.mean(diff)
                frame_diffs.append((i / fps, diff_score))
                
                # Detect scene change (threshold based on statistics)
                if diff_score > 30:  # Significant change
                    scene_changes.append({
                        "time": i / fps,
                        "frame": i,
                        "score": float(diff_score)
                    })
                    print(f"  Scene change at {i/fps:.1f}s (score: {diff_score:.1f})")
            
            prev_frame = gray_small
            
            # Show progress
            if i % (sample_rate * 30) == 0:  # Every 30 seconds
                progress = (i / total_frames) * 100
                print(f"  Progress: {progress:.1f}%")
        
        cap.release()
        
        # Save scene analysis
        scenes_data = {
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
            "scene_changes": scene_changes,
            "total_scenes": len(scene_changes)
        }
        
        scenes_file = self.output_dir / "scene_analysis.json"
        with open(scenes_file, 'w') as f:
            json.dump(scenes_data, f, indent=2)
        
        print(f"✅ Found {len(scene_changes)} real scene changes")
        return scenes_data
        
    def analyze_audio_activity(self):
        """Analyze actual audio for speech and silence"""
        print("🎤 Analyzing real audio activity...")
        
        # Extract audio
        audio_file = self.output_dir / "audio_analysis.wav"
        subprocess.run([
            "ffmpeg", "-i", str(self.source_video),
            "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
            str(audio_file), "-y"
        ], capture_output=True)
        
        # Load audio
        audio, sr = librosa.load(str(audio_file), sr=22050)
        
        # Compute energy
        hop_length = 512
        frame_length = 2048
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to time
        times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
        
        # Find silence (low energy)
        silence_threshold = np.percentile(energy, 20)  # Bottom 20% is silence
        silence_regions = []
        speech_regions = []
        
        in_silence = False
        silence_start = 0
        
        for i, (t, e) in enumerate(zip(times, energy)):
            is_silent = e < silence_threshold
            
            if is_silent and not in_silence:
                # Start of silence
                silence_start = t
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence
                if t - silence_start > 0.5:  # Only count silence > 0.5s
                    silence_regions.append({
                        "start": float(silence_start),
                        "end": float(t),
                        "duration": float(t - silence_start)
                    })
                in_silence = False
            elif not is_silent:
                # Speech/audio activity
                if i > 0 and energy[i-1] < silence_threshold:
                    speech_start = t
                if i < len(energy) - 1 and energy[i+1] < silence_threshold:
                    speech_regions.append({
                        "start": float(speech_start) if 'speech_start' in locals() else float(t),
                        "end": float(t),
                        "energy": float(e)
                    })
        
        # Find audio peaks (loud moments)
        peaks, properties = find_peaks(energy, height=np.percentile(energy, 80), distance=sr)
        audio_peaks = []
        for peak_idx in peaks:
            audio_peaks.append({
                "time": float(times[peak_idx]),
                "energy": float(energy[peak_idx])
            })
        
        audio_data = {
            "duration": float(times[-1]),
            "silence_regions": silence_regions[:50],  # Limit to 50
            "total_silence": sum(s["duration"] for s in silence_regions),
            "audio_peaks": audio_peaks[:30],  # Top 30 peaks
            "speech_percentage": (1 - sum(s["duration"] for s in silence_regions) / times[-1]) * 100
        }
        
        audio_file = self.output_dir / "audio_analysis.json"
        with open(audio_file, 'w') as f:
            json.dump(audio_data, f, indent=2)
        
        print(f"✅ Audio analysis complete:")
        print(f"   - {len(silence_regions)} silence regions")
        print(f"   - {len(audio_peaks)} audio peaks")
        print(f"   - {audio_data['speech_percentage']:.1f}% speech activity")
        
        return audio_data
        
    def detect_faces_and_objects(self):
        """Detect faces and important objects in video"""
        print("👤 Detecting faces and objects...")
        
        # Load OpenCV face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        cap = cv2.VideoCapture(str(self.source_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        face_moments = []
        
        # Sample every 60 frames (2 seconds at 30fps)
        sample_rate = int(fps * 2)
        
        for i in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                face_moments.append({
                    "time": i / fps,
                    "faces": len(faces),
                    "frame": i
                })
                
            # Progress
            if i % (sample_rate * 30) == 0:
                print(f"  Processed {i/fps:.1f}s")
        
        cap.release()
        
        detection_data = {
            "face_moments": face_moments[:50],  # Top 50 moments with faces
            "total_face_detections": len(face_moments)
        }
        
        detection_file = self.output_dir / "detection_analysis.json"
        with open(detection_file, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        print(f"✅ Found {len(face_moments)} moments with faces")
        return detection_data
        
    def analyze_motion_intensity(self):
        """Analyze motion intensity throughout video"""
        print("🏃 Analyzing motion intensity...")
        
        cap = cv2.VideoCapture(str(self.source_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        motion_data = []
        prev_gray = None
        
        # Sample every 15 frames (0.5 seconds at 30fps)
        sample_rate = int(fps / 2)
        
        for i in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_gray is not None:
                # Calculate motion
                frame_delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_score = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
                
                motion_data.append({
                    "time": i / fps,
                    "motion": float(motion_score)
                })
                
                # High motion moment
                if motion_score > 0.1:
                    print(f"  High motion at {i/fps:.1f}s: {motion_score:.3f}")
            
            prev_gray = gray
        
        cap.release()
        
        # Find peak motion moments
        motion_values = [m["motion"] for m in motion_data]
        motion_threshold = np.percentile(motion_values, 75)
        high_motion = [m for m in motion_data if m["motion"] > motion_threshold]
        
        motion_analysis = {
            "motion_samples": len(motion_data),
            "high_motion_moments": high_motion[:30],  # Top 30
            "average_motion": float(np.mean(motion_values))
        }
        
        motion_file = self.output_dir / "motion_analysis.json"
        with open(motion_file, 'w') as f:
            json.dump(motion_analysis, f, indent=2)
        
        print(f"✅ Found {len(high_motion)} high motion moments")
        return motion_analysis
        
    def generate_smart_highlights(self, scenes, audio, faces, motion):
        """Generate highlights based on REAL analysis"""
        print("✨ Generating smart highlights from real analysis...")
        
        highlights = []
        
        # 1. Add scene changes with high motion
        for scene in scenes["scene_changes"][:10]:
            # Check if this scene has high motion nearby
            motion_nearby = any(
                abs(m["time"] - scene["time"]) < 2 and m["motion"] > 0.1
                for m in motion["high_motion_moments"]
            )
            if motion_nearby:
                highlights.append({
                    "start": scene["time"],
                    "end": min(scene["time"] + 5, scenes["duration"]),
                    "type": "scene_change_motion",
                    "score": scene["score"]
                })
        
        # 2. Add moments with faces and audio peaks
        for face_moment in faces["face_moments"][:10]:
            # Check for audio peak nearby
            audio_peak_nearby = any(
                abs(p["time"] - face_moment["time"]) < 2
                for p in audio["audio_peaks"]
            )
            if audio_peak_nearby:
                highlights.append({
                    "start": face_moment["time"],
                    "end": min(face_moment["time"] + 4, scenes["duration"]),
                    "type": "face_with_speech",
                    "score": 0.9
                })
        
        # 3. Add high motion with audio
        for motion_moment in motion["high_motion_moments"][:10]:
            if motion_moment["motion"] > 0.15:
                highlights.append({
                    "start": motion_moment["time"],
                    "end": min(motion_moment["time"] + 3, scenes["duration"]),
                    "type": "high_action",
                    "score": motion_moment["motion"]
                })
        
        # Sort and merge overlapping
        highlights.sort(key=lambda x: x["start"])
        
        merged = []
        for h in highlights:
            if merged and h["start"] < merged[-1]["end"]:
                # Merge overlapping
                merged[-1]["end"] = max(merged[-1]["end"], h["end"])
                merged[-1]["score"] = max(merged[-1]["score"], h["score"])
            else:
                merged.append(h)
        
        # Take best highlights
        merged.sort(key=lambda x: -x["score"])
        final_highlights = merged[:15]  # Top 15
        final_highlights.sort(key=lambda x: x["start"])
        
        highlights_data = {
            "clips": final_highlights,
            "total_clips": len(final_highlights),
            "total_duration": sum(h["end"] - h["start"] for h in final_highlights)
        }
        
        highlights_file = self.output_dir / "smart_highlights.json"
        with open(highlights_file, 'w') as f:
            json.dump(highlights_data, f, indent=2)
        
        print(f"✅ Generated {len(final_highlights)} smart highlights")
        print(f"   Total duration: {highlights_data['total_duration']:.1f}s")
        
        return highlights_data
        
    def create_highlights_video(self, highlights):
        """Create the actual highlights video"""
        print("🎬 Creating highlights video...")
        
        # Extract clips
        clips = []
        for i, clip in enumerate(highlights["clips"]):
            clip_file = self.output_dir / f"clip_{i:02d}.mp4"
            
            subprocess.run([
                "ffmpeg", "-i", str(self.source_video),
                "-ss", str(clip["start"]),
                "-t", str(clip["end"] - clip["start"]),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                str(clip_file), "-y"
            ], capture_output=True)
            
            clips.append(clip_file)
            print(f"  Extracted: {clip['type']} at {clip['start']:.1f}s")
        
        # Concatenate
        concat_file = self.output_dir / "concat.txt"
        with open(concat_file, 'w') as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")
        
        final = self.output_dir / "REAL_HIGHLIGHTS.mp4"
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264", "-preset", "slow", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            str(final), "-y"
        ], capture_output=True)
        
        print(f"✅ Final highlights: {final}")
        return final
        
    def run_pipeline(self):
        """Run the real analysis pipeline"""
        print("="*60)
        print("🚀 REAL VIDEO ANALYSIS PIPELINE")
        print("="*60)
        
        try:
            # 1. Analyze scenes
            scenes = self.analyze_video_scenes()
            
            # 2. Analyze audio
            audio = self.analyze_audio_activity()
            
            # 3. Detect faces
            faces = self.detect_faces_and_objects()
            
            # 4. Analyze motion
            motion = self.analyze_motion_intensity()
            
            # 5. Generate smart highlights
            highlights = self.generate_smart_highlights(scenes, audio, faces, motion)
            
            # 6. Create video
            final = self.create_highlights_video(highlights)
            
            print("\n" + "="*60)
            print("✅ REAL ANALYSIS COMPLETE!")
            print("="*60)
            print(f"\n📊 Analysis Summary:")
            print(f"  • {scenes['total_scenes']} scene changes detected")
            print(f"  • {len(audio['audio_peaks'])} audio peaks found")
            print(f"  • {faces['total_face_detections']} face detections")
            print(f"  • {len(motion['high_motion_moments'])} high motion moments")
            print(f"  • {highlights['total_clips']} smart highlights generated")
            print(f"  • {highlights['total_duration']:.1f}s total highlights duration")
            
            # Open video
            subprocess.run(["open", str(final)])
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # Check for dependencies
    try:
        import cv2
        import librosa
        import numpy as np
    except ImportError as e:
        print(f"Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python", "librosa", "soundfile", "scipy", "-q"])
        print("Please run the script again.")
        sys.exit(1)
    
    analyzer = RealVideoAnalyzer()
    analyzer.run_pipeline()