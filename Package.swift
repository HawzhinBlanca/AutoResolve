// swift-tools-version: 5.10
// AutoResolve + AIDirector - Blueprint v7.5

import PackageDescription

let package = Package(
    name: "AutoResolve",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "AutoResolveCore", targets: ["AutoResolveCore"]),
        .library(name: "AIDirector", targets: ["AIDirector"]),
        .executable(name: "AutoResolveUI", targets: ["AutoResolveUI"])
    ],
    dependencies: [
        .package(url: "https://github.com/stephencelis/SQLite.swift.git", from: "0.14.0")
    ],
    targets: [
        // Core package - Foundation types
        .target(
            name: "AutoResolveCore",
            dependencies: [
                .product(name: "SQLite", package: "SQLite.swift")
            ],
            path: "AutoResolveCore/Sources/AutoResolveCore"
        ),
        
        // AI Director - Smart editing
        .target(
            name: "AIDirector",
            dependencies: ["AutoResolveCore"],
            path: "AIDirector/Sources/AIDirector"
        ),
        
        // Main UI Application
        .executableTarget(
            name: "AutoResolveUI",
            dependencies: ["AutoResolveCore", "AIDirector"]
        ),
        
        // Tests
        .testTarget(
            name: "AutoResolveUITests",
            dependencies: ["AutoResolveCore", "AutoResolveUI"],
            path: "Tests/AutoResolveUITests"
        ),
        .testTarget(
            name: "AIDirectorTests",
            dependencies: ["AIDirector"],
            path: "Tests/AIDirectorTests"
        )
    ]
)