#!/usr/bin/env python3
"""
AutoResolve CLI - Command line interface for video processing
"""

import argparse
import sys
import requests
import json
from pathlib import Path

def process_video(video_path: str, backend_url: str = "http://localhost:8000"):
    """Process video through AutoResolve pipeline"""
    
    # Check if file exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Call backend API
    try:
        response = requests.post(
            f"{backend_url}/api/process",
            json={"video_path": video_path}
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"Processing complete: {result['task_id']}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to process video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AutoResolve CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a video")
    process_parser.add_argument("video", help="Path to video file")
    process_parser.add_argument("--backend", default="http://localhost:8000", 
                                help="Backend URL")
    
    args = parser.parse_args()
    
    if args.command == "process":
        success = process_video(args.video, args.backend)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()