#!/bin/bash

# AutoResolve Frontend Launcher
echo "🚀 Launching AutoResolve Frontend..."

# First check and start backend if needed
echo "📡 Checking backend connection..."
if ! curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "🔧 Starting backend service..."
    cd /Users/hawzhin/AutoResolve/AutoResolveUI
    python3 backend_service.py > /tmp/autoresolve_backend.log 2>&1 &
    BACKEND_PID=$!
    echo "✅ Backend started (PID: $BACKEND_PID)"
    sleep 2
else
    echo "✅ Backend already running"
fi

cd /Users/hawzhin/AutoResolve/AutoResolveUI

# Check if already built
if [ -f ".build/arm64-apple-macosx/debug/AutoResolveUI" ]; then
    echo "✅ Using existing build"
    .build/arm64-apple-macosx/debug/AutoResolveUI
else
    echo "⚙️ Building AutoResolveUI for Apple Silicon..."
    
    # Try build with parse-as-library flag
    swift build --arch arm64 -Xswiftc -parse-as-library -Xswiftc -suppress-warnings 2>/dev/null
    
    if [ -f ".build/arm64-apple-macosx/debug/AutoResolveUI" ]; then
        echo "✅ Build successful, launching..."
        .build/arm64-apple-macosx/debug/AutoResolveUI
    else
        echo "❌ Build failed. Opening Xcode instead..."
        
        # Generate Xcode project and open it
        if command -v xed &> /dev/null; then
            xed .
        else
            echo "📝 To run the app:"
            echo "1. Open Xcode"
            echo "2. File > Open > Select /Users/hawzhin/AutoResolve/AutoResolveUI"
            echo "3. Press ⌘R to build and run"
            
            # Try to open Xcode with the project
            open -a Xcode /Users/hawzhin/AutoResolve/AutoResolveUI
        fi
    fi
fi