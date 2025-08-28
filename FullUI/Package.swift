// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FullUI",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "FullUI", targets: ["FullUI"])
    ],
    targets: [
        .executableTarget(
            name: "FullUI",
            dependencies: []
        )
    ]
)