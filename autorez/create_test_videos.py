#!/usr/bin/env python3
"""
Create test videos for V-JEPA vs CLIP evaluation
"""

import os
import subprocess

videos = [
    ("city_aerial", "testsrc2=size=640x480:rate=30:duration=180", "Urban cityscape test"),
    ("nature_forest", "mandelbrot=size=640x480:rate=30", "Nature scene test"),
    ("office_work", "testsrc=size=640x480:rate=30:duration=180", "Office environment test"),
    ("beach_sunset", "gradients=size=640x480:rate=30:duration=180", "Beach/sunset test"),
    ("mountain_climb", "cellauto=size=640x480:rate=30", "Mountain scene test")
]

output_dir = "assets/pilots"
os.makedirs(output_dir, exist_ok=True)

for name, filter_expr, description in videos:
    output_path = f"{output_dir}/{name}.mp4"
    
    if os.path.exists(output_path):
        print(f"✓ {name}.mp4 already exists")
        continue
    
    print(f"Creating {name}.mp4 - {description}...")
    
    # Handle duration properly
    if "duration" in filter_expr:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", filter_expr,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
    else:
        # For filters without duration, add -t flag
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", filter_expr,
            "-t", "180",  # 3 minutes
            "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Created {name}.mp4")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create {name}.mp4: {e}")

print("\n✅ Test videos ready for evaluation")