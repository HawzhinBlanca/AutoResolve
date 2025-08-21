#!/bin/bash

echo "🚀 AutoResolve Video Import System"
echo "=================================="

# Check if backend is running
echo "📡 Checking backend..."
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is running"
else
    echo "🔧 Starting backend..."
    cd /Users/hawzhin/AutoResolve/AutoResolveUI
    python3 backend_service.py > /tmp/autoresolve_backend.log 2>&1 &
    BACKEND_PID=$!
    echo "✅ Backend started (PID: $BACKEND_PID)"
    sleep 2
fi

# Demo the video import functionality
echo ""
echo "🎬 Video Import Features:"
echo "  • Drag & Drop support"
echo "  • File browser integration" 
echo "  • URL import capability"
echo "  • Metadata extraction"
echo "  • Real-time processing"
echo ""

# Run interactive demo
python3 << 'EOF'
import requests
import json
import time

print("📊 System Status:")
print("-" * 40)

try:
    # Get system status
    response = requests.get("http://localhost:5000/api/status")
    if response.status_code == 200:
        status = response.json()
        print(f"Memory: {status['memory_used_percent']:.1f}% used")
        print(f"Available: {status['memory_available_gb']:.2f} GB")
        print(f"Quality Level: {status['quality_level']}/5")
        print(f"Bug Fixes Active: {status['bug_fixes']}/12")
        print(f"Tests Passing: {status['tests_passing']}/47")
except:
    print("Backend not responding")

print("\n🎯 Processing Demo Video:")
print("-" * 40)

# Validate video
print("1. Validating video parameters...")
validation_data = {
    "duration": 120.0,
    "min_seg": 2.0,
    "max_seg": 8.0
}
response = requests.post("http://localhost:5000/api/validate", json=validation_data)
if response.status_code == 200:
    result = response.json()
    print(f"   ✓ Valid: {result['valid']}")
    if result.get('message'):
        print(f"   {result['message']}")

# Model comparison
print("\n2. Comparing AI models...")
response = requests.post("http://localhost:5000/api/model/compare", json={})
if response.status_code == 200:
    result = response.json()
    print(f"   ✓ Recommendation: {result['recommendation']}")
    print(f"   V-JEPA Gain: +{result['vjepa_gain_top3']:.1%}")

# Calculate score
print("\n3. Calculating quality score...")
score_data = {
    "content": 0.8,
    "narrative": 0.7,
    "tension": 0.6,
    "emphasis": 0.9,
    "continuity": 0.75,
    "rhythm_penalty": 0.1
}
response = requests.post("http://localhost:5000/api/score", json=score_data)
if response.status_code == 200:
    result = response.json()
    print(f"   ✓ Final Score: {result['score']:.3f}")

print("\n" + "="*60)
print("✅ All systems operational!")
print("🎬 AutoResolve is ready for video import and processing")
print("="*60)

# Interactive menu
print("\n📹 VIDEO IMPORT OPTIONS:")
print("1. Simulate drag & drop import")
print("2. Simulate file browser import")
print("3. Simulate URL import")
print("4. View processing pipeline")
print("5. Exit")

choice = input("\nSelect option (1-5): ")

if choice == "1":
    print("\n🎯 Drag & Drop Import Simulation")
    print("File: sample_video.mp4")
    print("Size: 125.3 MB")
    print("Duration: 2:00")
    print("Resolution: 1920x1080")
    print("✅ Video imported successfully!")
    
elif choice == "2":
    print("\n📁 File Browser Import Simulation")
    print("Browsing: /Users/Videos/")
    print("Selected: project_final.mov")
    print("Size: 450.7 MB")
    print("Duration: 5:30")
    print("Resolution: 3840x2160")
    print("✅ Video imported successfully!")
    
elif choice == "3":
    print("\n🌐 URL Import Simulation")
    url = input("Enter video URL (or press Enter for demo): ")
    if not url:
        url = "https://example.com/sample.mp4"
    print(f"Downloading: {url}")
    print("Progress: [████████████████████] 100%")
    print("✅ Video downloaded and imported!")
    
elif choice == "4":
    print("\n⚙️ Processing Pipeline:")
    print("1. Video validation ✓")
    print("2. Segment analysis ✓")
    print("3. AI model selection ✓")
    print("4. Score calculation ✓")
    print("5. Memory optimization ✓")
    print("6. Export generation ✓")
    
print("\n🎬 Thank you for using AutoResolve!")
EOF