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
        .executable(
            name: "MainApp",
            targets: ["MainApp"]
        ),
        .library(
            name: "BackendCore",
            targets: ["BackendCore"]
        ),
        .executable(
            name: "BackendSmoke",
            targets: ["BackendSmoke"]
        ),
        .executable(
            name: "UISmoke",
            targets: ["UISmoke"]
        ),
    ],
    targets: [
        .executableTarget(
            name: "AutoResolveUI",
            path: "Sources/AutoResolveUI",
            exclude: [
                "Backend/NetworkRetry.swift",
                "Backend/CircuitBreaker.swift",
                "Backend/ConnectionManager.swift",
                "main.swift.disabled", 
                "aggressive_fix.sh",
                "AI",
                "Audio",
                "AudioProcessing",
                "Authentication",
                "Cache",
                "BRoll",
                "ColorGrading",
                "Compliance",
                "Security/KeychainHelper.swift",
                "Inspector",
                "Inspectors",
                "Import",
                "MediaPool",
                "MotionGraphics",
                "MultiCam",
                "Performance",
                "Plugins",
                "ResolveIntegration",
                "Tests",
                "scripts"
            ],
            swiftSettings: [
                .unsafeFlags(["-suppress-warnings"])
            ]
        ),
        .target(
            name: "BackendCore",
            path: "Sources/AutoResolveUI",
            sources: [
                "Backend/NetworkRetry.swift",
                "Backend/CircuitBreaker.swift",
                "Backend/ConnectionManager.swift"
            ],
            swiftSettings: [
                .unsafeFlags(["-suppress-warnings"])
            ]
        ),
        .executableTarget(
            name: "BackendSmoke",
            path: "Tools/BackendSmoke"
        ),
        .executableTarget(
            name: "UISmoke",
            path: "Tools/UISmoke"
        ),
        .executableTarget(
            name: "MainApp",
            path: "Sources/MainApp"
        ),
    ]
)