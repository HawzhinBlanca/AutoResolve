#!/bin/bash

echo "🚀 AutoResolve Standalone Runner"
echo "================================"

cd /Users/hawzhin/AutoResolve/AutoResolveUI

# Check if backend is running
echo "📡 Checking backend..."
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is running"
else
    echo "🔧 Starting backend..."
    python3 backend_service.py > /tmp/backend.log 2>&1 &
    sleep 2
fi

echo "🔨 Building AutoResolveUI..."

# Build using xcodebuild
xcodebuild -scheme AutoResolveUI \
           -configuration Debug \
           -destination 'platform=macOS' \
           -derivedDataPath ./DerivedData \
           build 2>&1 | grep -E "BUILD|Error|Warning" || true

# Find the built app
APP_PATH=$(find ./DerivedData -name "AutoResolveUI" -type f -perm +111 2>/dev/null | head -1)

if [ -z "$APP_PATH" ]; then
    # Try alternative paths
    APP_PATH=$(find .build -name "AutoResolveUI" -type f -perm +111 2>/dev/null | head -1)
fi

if [ -n "$APP_PATH" ]; then
    echo "✅ Found app at: $APP_PATH"
    echo "🚀 Launching AutoResolve..."
    "$APP_PATH"
else
    echo "❌ Could not find built executable"
    echo ""
    echo "Alternative: Using osascript to run in Xcode..."
    
    # Open and run in Xcode using AppleScript
    osascript << 'EOF'
    tell application "Xcode"
        activate
        open "/Users/hawzhin/AutoResolve/AutoResolveUI"
        delay 2
        tell application "System Events"
            keystroke "r" using command down
        end tell
    end tell
EOF
    
    echo "✅ AutoResolve is running in Xcode!"
fi