#!/usr/bin/env python3
"""
AutoResolve v3.2 - Command Line Interface
Production CLI for video processing pipeline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def process_video(video_path: str, output_path: str = None, config: dict = None):
    """Process video through AutoResolve pipeline"""
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return False
    
    print(f"🎬 Processing video: {os.path.basename(video_path)}")
    start_time = time.time()
    
    try:
        # Import pipeline components
        from src.ops.silence import SilenceRemover
        from src.director.creative_director import analyze_video
        from src.broll.selector import BrollSelector
        
        results = {
            "video_path": video_path,
            "timestamp": time.time(),
            "pipeline_version": "3.2.0"
        }
        
        # Stage 1: Silence Detection
        print("🔇 Stage 1: Silence detection...")
        silence_remover = SilenceRemover()
        cuts_data, silence_metrics = silence_remover.remove_silence(video_path)
        results["silence"] = {
            "cuts": cuts_data,
            "metrics": silence_metrics
        }
        print(f"   Found {len(cuts_data.get('keep_windows', []))} speech segments")
        
        # Stage 2: Creative Director Analysis
        print("🎭 Stage 2: Creative analysis...")
        try:
            director_analysis = analyze_video(video_path)
            results["director"] = director_analysis
            print(f"   Analyzed {len(director_analysis.get('scenes', []))} scenes")
        except Exception as e:
            print(f"   ⚠️  Director analysis failed: {e}")
            results["director"] = {"error": str(e)}
        
        # Stage 3: B-roll Selection
        print("🎥 Stage 3: B-roll selection...")
        try:
            broll_selector = BrollSelector()
            selection_data, broll_metrics = broll_selector.select_broll(
                video_path,
                transcript_data=None,
                output_path=None
            )
            results["broll"] = {
                "selection": selection_data,
                "metrics": broll_metrics
            }
            selections_count = len(selection_data.get('selections', [])) if isinstance(selection_data, dict) else 0
            print(f"   Generated {selections_count} B-roll suggestions")
        except Exception as e:
            print(f"   ⚠️  B-roll selection failed: {e}")
            results["broll"] = {"error": str(e)}
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        
        # Estimate video duration for speed calculation
        try:
            import subprocess
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                video_duration = float(data.get('format', {}).get('duration', 0.0) or 0.0)
                processing_speed = video_duration / processing_time if processing_time > 0 else 0
            else:
                processing_speed = 0
        except:
            processing_speed = 0
        
        results["performance"] = {
            "processing_time_s": round(processing_time, 2),
            "processing_speed_x": round(processing_speed, 1),
            "video_duration_s": video_duration if 'video_duration' in locals() else 0
        }
        
        # Save results
        output_dir = Path(output_path) if output_path else Path("artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"autoresolve_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Processing complete in {processing_time:.2f}s ({processing_speed:.1f}x realtime)")
        print(f"📄 Results saved: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AutoResolve v3.2 - Video Processing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autoresolve_cli.py process video.mp4
  python autoresolve_cli.py process video.mp4 --output ./results/
  python autoresolve_cli.py health
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process video file')
    process_parser.add_argument('video_path', help='Path to video file')
    process_parser.add_argument('--output', '-o', help='Output directory', default='artifacts')
    process_parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check system health')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'process':
        config = {}
        if args.config and os.path.exists(args.config):
            with open(args.config) as f:
                config = json.load(f)
        
        success = process_video(args.video_path, args.output, config)
        return 0 if success else 1
        
    elif args.command == 'health':
        try:
            from scripts.startup_checks import run_comprehensive_checks
            success = run_comprehensive_checks()
            return 0 if success else 1
        except ImportError:
            print("❌ Health checks not available")
            return 1
            
    elif args.command == 'status':
        try:
            import requests
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"🟢 Backend: {data['status']}")
                print(f"💾 Memory: {data['memory_usage_gb']}GB")
                print(f"⚙️  Active tasks: {data['active_tasks']}")
            else:
                print("🔴 Backend: not responding")
        except:
            print("🔴 Backend: offline")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())