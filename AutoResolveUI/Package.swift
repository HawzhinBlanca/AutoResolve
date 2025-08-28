// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AutoResolveUI",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "AutoResolveUI",
            targets: ["AutoResolveUI"]
        ),
    ],
    targets: [
        .executableTarget(
            name: "AutoResolveUI",
            path: "Sources/AutoResolveUI",
            exclude: [
                "main.swift.disabled",
                "aggressive_fix.sh",
                "AI",
                "Audio",
                "AudioProcessing",
                "Authentication",
                "Cache",
                "ColorGrading",
                "Compliance",
                "Import",
                "MediaPool",
                "MotionGraphics",
                "MultiCam",
                "Performance",
                "Plugins",
                "ResolveIntegration",
                "Tests",
                "scripts",
                "Timeline/VirtualScrollingTimeline.swift",
                "Timeline/TimelineSnappingSystem.swift"
            ],
            swiftSettings: [
                .unsafeFlags(["-suppress-warnings"])
            ]
        ),
    ]
)