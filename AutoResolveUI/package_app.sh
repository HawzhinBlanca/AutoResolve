#!/bin/bash
# Package AutoResolve as .app bundle
# TASK 4: Package as .app bundle

echo "📦 Packaging AutoResolve.app..."

# Clean and build
echo "🔨 Building release version..."
cd /Users/hawzhin/AutoResolve/AutoResolveSimple
swift build -c release

# Create app bundle structure
echo "📁 Creating app bundle..."
rm -rf AutoResolve.app
mkdir -p AutoResolve.app/Contents/{MacOS,Resources}

# Copy files
echo "📋 Copying files..."
cp .build/release/AutoResolveSimple AutoResolve.app/Contents/MacOS/AutoResolve
cp Info.plist AutoResolve.app/Contents/
cp AppIcon.icns AutoResolve.app/Contents/Resources/

# Fix executable name in Info.plist
sed -i '' 's/AutoResolveSimple/AutoResolve/g' AutoResolve.app/Contents/Info.plist

# Sign the app (optional, for distribution)
if command -v codesign &> /dev/null; then
    echo "✍️ Signing app..."
    codesign --force --deep --sign - AutoResolve.app
fi

# Verify
echo "✅ Verifying app bundle..."
if [ -f AutoResolve.app/Contents/MacOS/AutoResolve ]; then
    echo "   ✓ Executable found"
fi
if [ -f AutoResolve.app/Contents/Info.plist ]; then
    echo "   ✓ Info.plist found"
fi
if [ -f AutoResolve.app/Contents/Resources/AppIcon.icns ]; then
    echo "   ✓ Icon found"
fi

# Get app size
APP_SIZE=$(du -sh AutoResolve.app | cut -f1)
echo "📊 App size: $APP_SIZE"

echo "✨ AutoResolve.app successfully packaged!"
echo "📍 Location: $(pwd)/AutoResolve.app"
echo ""
echo "To launch: open AutoResolve.app"
echo "To move to Applications: mv AutoResolve.app /Applications/"