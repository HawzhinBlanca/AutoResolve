#!/usr/bin/env python3
"""
AutoResolve Launcher
Builds and runs the Swift app without Xcode
"""

import subprocess
import os
import sys
import time

def main():
    print("🚀 AutoResolve Launcher")
    print("=" * 50)
    
    # Change to project directory
    os.chdir("/Users/hawzhin/AutoResolve/AutoResolveUI")
    
    # Check if backend is running
    print("📡 Checking backend...")
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:5000/api/health"],
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            print("✅ Backend is running")
        else:
            print("🔧 Starting backend...")
            subprocess.Popen(
                ["python3", "backend_service.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)
    except:
        print("⚠️  Backend check failed")
    
    print("\n🔨 Building AutoResolveUI...")
    
    # Create a simple Swift executable wrapper
    wrapper_code = '''
import Cocoa
import SwiftUI

// Import all the views
#if canImport(AutoResolveUI)
import AutoResolveUI
#endif

// Create the app delegate
class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create the main view
        let contentView = PolishedMainView()
            .environmentObject(BackendConnection.shared)
            .environmentObject(ConnectedUnifiedStore())
            .environmentObject(VideoImportManager.shared)
        
        // Create the window
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.center()
        window.setFrameAutosaveName("AutoResolve")
        window.contentView = NSHostingView(rootView: contentView)
        window.title = "AutoResolve"
        window.makeKeyAndOrderFront(nil)
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// Create and run the app
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
'''
    
    # Write the wrapper
    with open("Sources/AutoResolveUI/AppLauncher.swift", "w") as f:
        f.write(wrapper_code)
    
    # Build using swiftc directly
    print("🔧 Compiling...")
    
    compile_cmd = [
        "swiftc",
        "-o", "AutoResolveApp",
        "-framework", "SwiftUI",
        "-framework", "AVKit",
        "-framework", "AppKit",
        "-parse-as-library",
        "-target", "arm64-apple-macosx14.0",
        "Sources/AutoResolveUI/*.swift"
    ]
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Try alternative build approach
            print("⚠️  Direct compilation failed, trying SPM build...")
            
            # Use swift build with specific configuration
            build_cmd = ["swift", "build", "--configuration", "release", "-Xswiftc", "-parse-as-library"]
            subprocess.run(build_cmd, check=False)
            
            # Look for built executable
            possible_paths = [
                ".build/release/AutoResolveUI",
                ".build/debug/AutoResolveUI",
                ".build/arm64-apple-macosx/release/AutoResolveUI",
                ".build/arm64-apple-macosx/debug/AutoResolveUI"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"✅ Found executable at {path}")
                    print("🚀 Launching AutoResolve...")
                    subprocess.run([path])
                    return
            
            print("❌ Could not find built executable")
            print("\nℹ️  Opening in Xcode instead...")
            subprocess.run(["open", "-a", "Xcode", "/Users/hawzhin/AutoResolve/AutoResolveUI"])
            print("\n📝 In Xcode:")
            print("1. Press ⌘+R to run the app")
            print("2. The app will launch with full video import functionality")
        else:
            # Run the compiled app
            if os.path.exists("AutoResolveApp"):
                print("✅ Build successful!")
                print("🚀 Launching AutoResolve...")
                subprocess.run(["./AutoResolveApp"])
            else:
                print("❌ Build completed but executable not found")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Alternative: Run from Xcode")
        print("1. Open Xcode (it should already be open)")
        print("2. Press ⌘+R to run")

if __name__ == "__main__":
    main()