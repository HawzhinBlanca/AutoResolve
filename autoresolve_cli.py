#!/usr/bin/env python3
"""
AutoResolve V3.0 - Complete CLI Interface
Professional video editing pipeline with AI-powered content analysis
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

# Add autorez to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'autorez'))

from autoresolve_complete import AutoResolve

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('autoresolve.log')
        ]
    )

def print_banner():
    """Print AutoResolve banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                         AUTORESOLVE V3.0                            ║
║                  Professional AI Video Editor                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  • Real-time silence detection & removal                            ║
║  • AI-powered scene change detection                                ║
║  • V-JEPA visual embeddings for content analysis                    ║
║  • Creative Director for story beats & emphasis                     ║
║  • Timeline generation with professional cuts                       ║
║  • Export to FCPXML, EDL, and other formats                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def validate_input(video_path: str) -> bool:
    """Validate input video file"""
    if not Path(video_path).exists():
        print(f"❌ Error: Video file '{video_path}' not found")
        return False
    
    # Check file size
    size_mb = Path(video_path).stat().st_size / (1024 * 1024)
    if size_mb > 5000:  # 5GB limit
        print(f"⚠️  Warning: Large video file ({size_mb:.1f}MB). Processing may take time.")
    
    return True

def process_video(args):
    """Process a single video"""
    print_banner()
    
    if not validate_input(args.video):
        return False
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create output directory
    output_dir = Path(args.output) if args.output else Path("autoresolve_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📹 Input: {args.video}")
    print(f"📁 Output: {output_dir}")
    print(f"⚙️  Options: silence_threshold={args.silence_threshold}dB")
    if args.broll:
        print(f"🎬 B-roll: {args.broll}")
    print()
    
    # Initialize AutoResolve
    try:
        autoresolve = AutoResolve()
        
        # Process video
        result = autoresolve.process_video(
            args.video, 
            output_dir=str(output_dir)
        )
        
        if result:
            print_results(result, output_dir)
            return True
        else:
            print("❌ Processing failed!")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

def print_results(result: dict, output_dir: Path):
    """Print processing results"""
    print("\n" + "="*70)
    print("                    PROCESSING COMPLETE")
    print("="*70)
    
    # Performance metrics
    processing_time = result.get('processing_time', 0)
    timeline_duration = result.get('timeline', {}).get('duration', 0)
    rtf = timeline_duration / processing_time if processing_time > 0 else 0
    
    print(f"\n📊 Performance:")
    print(f"  • Processing time: {processing_time:.1f}s")
    print(f"  • Realtime factor: {rtf:.0f}x")
    print(f"  • Timeline duration: {timeline_duration/60:.1f}m")
    
    # Analysis results
    results = result.get('results', {})
    print(f"\n🔍 Analysis:")
    print(f"  • Silence regions: {results.get('silence_regions', 0)}")
    print(f"  • Scene changes: {results.get('scene_changes', 0)}")
    print(f"  • Story beats: {results.get('story_beats', 0)}")
    print(f"  • Timeline clips: {results.get('timeline_clips', 0)}")
    
    # Output files
    exports = result.get('exports', {})
    print(f"\n📁 Output files:")
    for format_name, filepath in exports.items():
        if filepath and Path(filepath).exists():
            size_kb = Path(filepath).stat().st_size / 1024
            print(f"  • {format_name.upper()}: {filepath} ({size_kb:.1f}KB)")
    
    # Project file
    project_file = output_dir / "project.json"
    if project_file.exists():
        size_kb = project_file.stat().st_size / 1024
        print(f"  • PROJECT: {project_file} ({size_kb:.1f}KB)")
    
    print(f"\n✅ Ready for import into professional video editors!")
    print("="*70)

def batch_process(args):
    """Process multiple videos"""
    print_banner()
    print(f"🎬 Batch processing videos from: {args.input_dir}")
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory '{args.input_dir}' not found")
        return False
    
    # Find video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.mxf', '.prores']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"❌ No video files found in '{args.input_dir}'")
        return False
    
    print(f"📹 Found {len(video_files)} video files")
    
    # Process each video
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        
        # Create individual output directory
        output_dir = Path(args.output) / video_file.stem
        
        # Update args for this video
        args.video = str(video_file)
        args.output = str(output_dir)
        
        if process_video(args):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n📊 Batch processing complete:")
    print(f"  • Successful: {successful}")
    print(f"  • Failed: {failed}")
    print(f"  • Total: {len(video_files)}")
    
    return failed == 0

def create_project(args):
    """Create a new AutoResolve project"""
    print("🆕 Creating new AutoResolve project...")
    
    project_dir = Path(args.name)
    project_dir.mkdir(exist_ok=True)
    
    # Create project structure
    (project_dir / "media").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)
    (project_dir / "broll").mkdir(exist_ok=True)
    
    # Create project file
    project_config = {
        "name": args.name,
        "created_at": datetime.now().isoformat(),
        "version": "3.0",
        "settings": {
            "silence_threshold": -30,
            "scene_threshold": 0.3,
            "enable_transcription": True,
            "output_format": ["fcpxml", "edl"]
        },
        "directories": {
            "media": "media/",
            "output": "output/",
            "broll": "broll/"
        }
    }
    
    project_file = project_dir / "autoresolve_project.json"
    with open(project_file, 'w') as f:
        json.dump(project_config, f, indent=2)
    
    print(f"✅ Project '{args.name}' created at: {project_dir}")
    print(f"  • Configuration: {project_file}")
    print(f"  • Media folder: {project_dir / 'media'}")
    print(f"  • Output folder: {project_dir / 'output'}")
    print(f"  • B-roll folder: {project_dir / 'broll'}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AutoResolve V3.0 - Professional AI Video Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python autoresolve_cli.py process video.mp4
  
  # Process with custom settings
  python autoresolve_cli.py process video.mp4 -s -35 --transcribe -v
  
  # Batch process directory
  python autoresolve_cli.py batch /path/to/videos -o /path/to/output
  
  # Create new project
  python autoresolve_cli.py create my_project
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a video file')
    process_parser.add_argument('video', help='Input video file')
    process_parser.add_argument('-o', '--output', help='Output directory')
    process_parser.add_argument('-s', '--silence-threshold', type=float, default=-30,
                               help='Silence threshold in dB (default: -30)')
    process_parser.add_argument('--scene-threshold', type=float, default=0.3,
                               help='Scene change threshold (default: 0.3)')
    process_parser.add_argument('-b', '--broll', help='B-roll directory')
    process_parser.add_argument('-t', '--transcribe', action='store_true',
                               help='Enable transcription')
    process_parser.add_argument('--fast', action='store_true',
                               help='Fast mode (skip detailed analysis)')
    process_parser.add_argument('-v', '--verbose', action='store_true',
                               help='Verbose logging')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple videos')
    batch_parser.add_argument('input_dir', help='Input directory with videos')
    batch_parser.add_argument('-o', '--output', default='batch_output',
                             help='Output directory (default: batch_output)')
    batch_parser.add_argument('-s', '--silence-threshold', type=float, default=-30,
                             help='Silence threshold in dB (default: -30)')
    batch_parser.add_argument('--scene-threshold', type=float, default=0.3,
                             help='Scene change threshold (default: 0.3)')
    batch_parser.add_argument('-b', '--broll', help='B-roll directory')
    batch_parser.add_argument('-t', '--transcribe', action='store_true',
                             help='Enable transcription')
    batch_parser.add_argument('--fast', action='store_true',
                             help='Fast mode (skip detailed analysis)')
    batch_parser.add_argument('-v', '--verbose', action='store_true',
                             help='Verbose logging')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new project')
    create_parser.add_argument('name', help='Project name')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        if args.command == 'process':
            success = process_video(args)
        elif args.command == 'batch':
            success = batch_process(args)
        elif args.command == 'create':
            create_project(args)
            success = True
        else:
            parser.print_help()
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())