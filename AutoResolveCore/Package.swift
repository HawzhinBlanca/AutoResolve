// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "AutoResolveCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "AutoResolveCore",
            targets: ["AutoResolveCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/stephencelis/SQLite.swift.git", from: "0.14.0")
    ],
    targets: [
        .target(
            name: "AutoResolveCore",
            dependencies: [
                .product(name: "SQLite", package: "SQLite.swift")
            ]),
        .testTarget(
            name: "AutoResolveCoreTests",
            dependencies: ["AutoResolveCore"]),
    ]
)