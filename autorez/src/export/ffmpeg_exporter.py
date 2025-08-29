"""
FFmpeg-based video exporter for AutoResolve
Handles real MP4 export from timeline data
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, List
import tempfile
import shutil

logger = logging.getLogger(__name__)

class FFmpegExporter:
    def __init__(self):
        self.ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_path = shutil.which("ffprobe") or "ffprobe"
        
    def export_timeline_to_mp4(
        self,
        clips: List[Dict],
        output_path: str,
        resolution: str = "1920x1080",
        fps: int = 30,
        preset: str = "medium",
        crf: int = 23,
        audio_bitrate: str = "192k"
    ) -> Dict:
        """
        Export timeline clips to MP4 using FFmpeg
        
        Args:
            clips: List of timeline clips with source paths and timing
            output_path: Output MP4 path
            resolution: Video resolution (WxH)
            fps: Frame rate
            preset: FFmpeg preset (ultrafast, fast, medium, slow, veryslow)
            crf: Quality (0-51, lower is better)
            audio_bitrate: Audio bitrate
            
        Returns:
            Export result with status and metadata
        """
        try:
            # Validate clips have required fields
            valid_clips = []
            for clip in clips:
                if "source_path" in clip and os.path.exists(clip["source_path"]):
                    valid_clips.append(clip)
                elif "path" in clip and os.path.exists(clip["path"]):
                    clip["source_path"] = clip["path"]
                    valid_clips.append(clip)
            
            if not valid_clips:
                return {
                    "status": "error",
                    "error": "No valid clips with source paths found"
                }
            
            # Sort clips by start time
            valid_clips.sort(key=lambda x: x.get("start_time", 0))
            
            # Create concat list file for FFmpeg
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as concat_file:
                concat_path = concat_file.name
                
                for clip in valid_clips:
                    source = clip["source_path"]
                    start = clip.get("start_time", 0)
                    duration = clip.get("duration", 10)
                    
                    # Write file directive with trim parameters
                    concat_file.write(f"file '{source}'\n")
                    concat_file.write(f"inpoint {start}\n")
                    concat_file.write(f"outpoint {start + duration}\n")
            
            # Build FFmpeg command
            width, height = resolution.split('x')
            cmd = [
                self.ffmpeg_path,
                "-f", "concat",
                "-safe", "0",
                "-i", concat_path,
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                "-r", str(fps),
                "-c:a", "aac",
                "-b:a", audio_bitrate,
                "-movflags", "+faststart",
                "-y",  # Overwrite output
                output_path
            ]
            
            logger.info(f"Starting FFmpeg export: {' '.join(cmd)}")
            
            # Execute FFmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            # Clean up temp file
            try:
                os.unlink(concat_path)
            except:
                pass
            
            if process.returncode != 0:
                logger.error(f"FFmpeg failed: {stderr}")
                return {
                    "status": "error",
                    "error": f"FFmpeg export failed: {stderr[:500]}"
                }
            
            # Verify output exists
            if not os.path.exists(output_path):
                return {
                    "status": "error",
                    "error": "Output file was not created"
                }
            
            # Get output file metadata
            output_size = os.path.getsize(output_path)
            
            return {
                "status": "success",
                "output_path": output_path,
                "output_size": output_size,
                "clips_exported": len(valid_clips),
                "resolution": resolution,
                "fps": fps,
                "preset": preset,
                "crf": crf
            }
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def export_with_transitions(
        self,
        clips: List[Dict],
        output_path: str,
        transition_duration: float = 0.5,
        **kwargs
    ) -> Dict:
        """
        Export with crossfade transitions between clips
        """
        try:
            if len(clips) < 2:
                # No transitions needed for single clip
                return self.export_timeline_to_mp4(clips, output_path, **kwargs)
            
            # Create complex filter for transitions
            filter_complex = []
            inputs = []
            
            for i, clip in enumerate(clips):
                source = clip.get("source_path", clip.get("path"))
                if not source or not os.path.exists(source):
                    continue
                    
                start = clip.get("start_time", 0)
                duration = clip.get("duration", 10)
                
                # Add input with trim
                inputs.extend([
                    "-ss", str(start),
                    "-t", str(duration),
                    "-i", source
                ])
                
                if i == 0:
                    filter_complex.append("[0:v][0:a]")
                elif i < len(clips) - 1:
                    # Add crossfade
                    filter_complex.append(
                        f"[{i-1}v][{i}:v]xfade=transition=fade:duration={transition_duration}:offset={sum(c.get('duration', 10) for c in clips[:i])-transition_duration}[{i}v];"
                    )
                    filter_complex.append(
                        f"[{i-1}a][{i}:a]acrossfade=duration={transition_duration}[{i}a];"
                    )
            
            # Build final filter
            filter_str = "".join(filter_complex)
            
            resolution = kwargs.get("resolution", "1920x1080")
            fps = kwargs.get("fps", 30)
            preset = kwargs.get("preset", "medium")
            crf = kwargs.get("crf", 23)
            
            cmd = [
                self.ffmpeg_path,
                *inputs,
                "-filter_complex", filter_str,
                "-map", f"[{len(clips)-1}v]",
                "-map", f"[{len(clips)-1}a]",
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-r", str(fps),
                "-c:a", "aac",
                "-b:a", kwargs.get("audio_bitrate", "192k"),
                "-movflags", "+faststart",
                "-y",
                output_path
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"FFmpeg transitions failed: {process.stderr}")
                # Fall back to simple export
                return self.export_timeline_to_mp4(clips, output_path, **kwargs)
            
            return {
                "status": "success",
                "output_path": output_path,
                "clips_exported": len(clips),
                "transitions": True
            }
            
        except Exception as e:
            logger.error(f"Transition export failed, using simple export: {e}")
            return self.export_timeline_to_mp4(clips, output_path, **kwargs)
    
    def create_proxy(self, source_path: str, proxy_dir: str = "/tmp/proxies") -> str:
        """Create low-res proxy for editing"""
        try:
            Path(proxy_dir).mkdir(parents=True, exist_ok=True)
            
            source_name = Path(source_path).stem
            proxy_path = f"{proxy_dir}/{source_name}_proxy.mp4"
            
            cmd = [
                self.ffmpeg_path,
                "-i", source_path,
                "-vf", "scale=640:-2",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                "-c:a", "aac",
                "-b:a", "96k",
                "-y",
                proxy_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return proxy_path
            
        except Exception as e:
            logger.error(f"Proxy creation failed: {e}")
            return source_path  # Fall back to original