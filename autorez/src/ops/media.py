import logging

logger = logging.getLogger(__name__)

#!/usr/bin/env python3
"""
Media I/O operations module
Handles audio extraction, concatenation, re-muxing
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
import time

def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    format: str = 'wav'
) -> Dict:
    """
    Extract audio from video file
    
    Args:
        video_path: Path to input video
        output_path: Output audio path
        sample_rate: Target sample rate
        channels: Number of channels (1=mono, 2=stereo)
        format: Output format (wav, mp3, aac)
        
    Returns:
        Result dictionary
    """
    if not output_path:
        output_path = Path(video_path).with_suffix(f'.{format}')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn',  # No video
        '-ar', str(sample_rate),
        '-ac', str(channels)
    ]
    
    # Format-specific options
    if format == 'wav':
        cmd.extend(['-acodec', 'pcm_s16le'])
    elif format == 'mp3':
        cmd.extend(['-acodec', 'libmp3lame', '-b:a', '128k'])
    elif format == 'aac':
        cmd.extend(['-acodec', 'aac', '-b:a', '128k'])
    
    cmd.extend([str(output_path), '-y'])
    
    # Execute
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return {
            "success": False,
            "error": result.stderr,
            "path": str(output_path)
        }
    
    # Get duration
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json', str(output_path)
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    
    duration = 0
    if probe_result.returncode == 0:
        probe_data = json.loads(probe_result.stdout)
        duration = float(probe_data.get('format', {}).get('duration', 0))
    
    # Emit telemetry
    try:
        from src.utils.memory import emit_metrics
        emit_metrics("media_extract_audio", {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "format": format,
            "processing_time": time.time() - start_time
        })
    except ImportError:
        pass
    
    return {
        "success": True,
        "path": str(output_path),
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "format": format
    }

# Aliases for Blueprint requirements
extract_segment = lambda *args, **kwargs: extract_audio(*args, **kwargs)
concat_segments = lambda *args, **kwargs: concatenate_clips(*args, **kwargs)

def concatenate_clips(
    clip_paths: List[str],
    output_path: str,
    codec: str = 'copy'
) -> Dict:
    """
    Concatenate multiple video clips
    
    Args:
        clip_paths: List of input clip paths
        output_path: Output video path
        codec: Video codec ('copy' for stream copy, 'libx264' for re-encode)
        
    Returns:
        Result dictionary
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create concat file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for clip in clip_paths:
            f.write(f"file '{Path(clip).absolute()}'\n")
        concat_file = f.name
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file
    ]
    
    if codec == 'copy':
        cmd.extend(['-c', 'copy'])
    else:
        cmd.extend([
            '-c:v', codec,
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k'
        ])
    
    cmd.extend([str(output_path), '-y'])
    
    # Execute
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up concat file
    Path(concat_file).unlink()
    
    if result.returncode != 0:
        return {
            "success": False,
            "error": result.stderr,
            "path": str(output_path)
        }
    
    # Get output info
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration,size',
        '-of', 'json', str(output_path)
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    
    duration = 0
    size = 0
    if probe_result.returncode == 0:
        probe_data = json.loads(probe_result.stdout)
        format_data = probe_data.get('format', {})
        duration = float(format_data.get('duration', 0))
        size = int(format_data.get('size', 0))
    
    # Emit telemetry
    try:
        from src.utils.memory import emit_metrics
        emit_metrics("media_concatenate", {
            "clips": len(clip_paths),
            "duration": duration,
            "size_mb": size / (1024 * 1024),
            "processing_time": time.time() - start_time
        })
    except ImportError:
        pass
    
    return {
        "success": True,
        "path": str(output_path),
        "clips": len(clip_paths),
        "duration": duration,
        "size": size
    }

def remux_media(
    video_path: str,
    audio_path: str,
    output_path: str,
    offset: float = 0.0
) -> Dict:
    """
    Re-mux video with new audio track
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Output path
        offset: Audio offset in seconds
        
    Returns:
        Result dictionary
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-itsoffset', str(offset),
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-map', '0:v:0',
        '-map', '1:a:0',
        str(output_path), '-y'
    ]
    
    # Execute
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return {
            "success": False,
            "error": result.stderr,
            "path": str(output_path)
        }
    
    # Emit telemetry
    try:
        from src.utils.memory import emit_metrics
        emit_metrics("media_remux", {
            "offset": offset,
            "processing_time": time.time() - start_time
        })
    except ImportError:
        pass
    
    return {
        "success": True,
        "path": str(output_path),
        "offset": offset
    }

def get_media_info(media_path: str) -> Dict:
    """
    Get detailed media information
    
    Args:
        media_path: Path to media file
        
    Returns:
        Media information dictionary
    """
    # Run ffprobe
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'stream:format',
        '-of', 'json',
        str(media_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return {
            "success": False,
            "error": result.stderr
        }
    
    try:
        data = json.loads(result.stdout)
        
        # Extract relevant info
        format_info = data.get('format', {})
        streams = data.get('streams', [])
        
        video_stream = next((s for s in streams if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in streams if s['codec_type'] == 'audio'), None)
        
        info = {
            "success": True,
            "path": media_path,
            "duration": float(format_info.get('duration', 0)),
            "size": int(format_info.get('size', 0)),
            "bitrate": int(format_info.get('bit_rate', 0))
        }
        
        if video_stream:
            info["video"] = {
                "codec": video_stream.get('codec_name'),
                "width": video_stream.get('width'),
                "height": video_stream.get('height'),
                "fps": eval(video_stream.get('r_frame_rate', '0/1'))
            }
        
        if audio_stream:
            info["audio"] = {
                "codec": audio_stream.get('codec_name'),
                "channels": audio_stream.get('channels'),
                "sample_rate": int(audio_stream.get('sample_rate', 0))
            }
        
        return info
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def safe_codec_defaults() -> Dict[str, str]:
    """Get safe codec defaults for maximum compatibility"""
    return {
        "video": "libx264",
        "audio": "aac",
        "preset": "medium",
        "crf": "23",
        "audio_bitrate": "128k"
    }


def main():
    """CLI entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Media I/O operations")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract audio
    extract_parser = subparsers.add_parser('extract', help='Extract audio')
    extract_parser.add_argument('input', help='Input video')
    extract_parser.add_argument('--output', help='Output audio')
    extract_parser.add_argument('--rate', type=int, default=16000, help='Sample rate')
    extract_parser.add_argument('--channels', type=int, default=1, help='Channels')
    extract_parser.add_argument('--format', default='wav', help='Format')
    
    # Concatenate
    concat_parser = subparsers.add_parser('concat', help='Concatenate clips')
    concat_parser.add_argument('clips', nargs='+', help='Input clips')
    concat_parser.add_argument('--output', required=True, help='Output video')
    concat_parser.add_argument('--codec', default='copy', help='Codec')
    
    # Remux
    remux_parser = subparsers.add_parser('remux', help='Remux video with audio')
    remux_parser.add_argument('video', help='Input video')
    remux_parser.add_argument('audio', help='Input audio')
    remux_parser.add_argument('--output', required=True, help='Output video')
    remux_parser.add_argument('--offset', type=float, default=0, help='Audio offset')
    
    # Info
    info_parser = subparsers.add_parser('info', help='Get media info')
    info_parser.add_argument('input', help='Input media')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        result = extract_audio(
            args.input,
            args.output,
            args.rate,
            args.channels,
            args.format
        )
    elif args.command == 'concat':
        result = concatenate_clips(args.clips, args.output, args.codec)
    elif args.command == 'remux':
        result = remux_media(args.video, args.audio, args.output, args.offset)
    elif args.command == 'info':
        result = get_media_info(args.input)
        logger.info(json.dumps(result, indent=2))
        return 0 if result.get('success') else 1
    else:
        parser.print_help()
        return 1
    
    if result.get('success'):
        logger.info(f"✓ Success: {result.get('path')}")
        return 0
    else:
        logger.error(f"✗ Failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())