#!/bin/bash

# AutoResolve Frontend Launcher
# Builds and launches the native macOS SwiftUI application

echo "🚀 AutoResolve Frontend Launcher"
echo "================================"

# Check if we're in the right directory
if [ ! -d "AutoResolveUI" ]; then
    echo "❌ Error: AutoResolveUI directory not found"
    echo "Please run this script from the AutoResolve root directory"
    exit 1
fi

# Check for Swift
if ! command -v swift &> /dev/null; then
    echo "❌ Error: Swift not found"
    echo "Please install Xcode and Swift toolchain"
    exit 1
fi

echo "📦 Building AutoResolve UI..."
cd AutoResolveUI

# Build in release mode for best performance
if swift build --configuration release; then
    echo "✅ Build successful!"
    
    # Check if backend is running
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Backend is running"
    else
        echo "⚠️  Warning: Backend not detected at http://localhost:8080"
        echo "   The app will run but won't receive live data"
        echo "   Start the backend with: cd autorez && python3 -m src.backend.server"
    fi
    
    echo ""
    echo "🎯 Launching AutoResolve..."
    echo "================================"
    
    # Launch the app
    .build/release/AutoResolveUI &
    
    # Store the PID
    APP_PID=$!
    echo "✅ AutoResolve launched (PID: $APP_PID)"
    echo ""
    echo "📌 Tips:"
    echo "   - Use Cmd+Q to quit the app"
    echo "   - Check Menu Bar for status indicator"
    echo "   - Open Settings with Cmd+,"
    echo "   - View logs in Console.app"
    echo ""
    echo "🎉 AutoResolve is running!"
    
else
    echo "❌ Build failed!"
    echo "Please check the error messages above"
    exit 1
fi