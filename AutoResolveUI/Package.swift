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
            dependencies: [],
            sources: ["SimpleEnhanced.swift"],
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        ),
    ]
)