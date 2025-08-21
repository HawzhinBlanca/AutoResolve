#!/bin/bash

# Launch script for AutoResolve
# This will build and run the minimal working version

echo "🚀 Building AutoResolve..."

cd /Users/hawzhin/AutoResolve/AutoResolveSimple

# Build the app
if swift build; then
    echo "✅ Build successful!"
    echo "🎬 Launching AutoResolve..."
    
    # Run the app
    ./.build/debug/AutoResolveSimple
else
    echo "❌ Build failed"
    exit 1
fi