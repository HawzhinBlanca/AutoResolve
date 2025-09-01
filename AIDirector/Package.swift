// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "AIDirector",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AIDirector", targets: ["AIDirector"])
    ],
    dependencies: [
        .package(path: "../AutoResolveCore")
    ],
    targets: [
        .target(
            name: "AIDirector",
            dependencies: [
                .product(name: "AutoResolveCore", package: "AutoResolveCore")
            ]
        )
    ]
)