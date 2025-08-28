import Combine
import SwiftUI
import AVFoundation
import CoreImage

// MARK: - Plugin Architecture

@MainActor
public class PluginManager: ObservableObject {
    @Published public var loadedPlugins: [AutoResolvePlugin] = []
    @Published public var availablePlugins: [PluginInfo] = []
    @Published public var isLoadingPlugins = false
    @Published public var pluginErrors: [PluginError] = []
    
    private let pluginQueue = DispatchQueue(label: "plugin.manager", qos: .userInitiated)
    private var pluginRegistry: [String: PluginRegistration] = [:]
    private var cancellables = Set<AnyCancellable>()
    private let logger = Logger.shared
    
    public static let shared = PluginManager()
    
    private init() {
        loadBuiltInPlugins()
        scanForPlugins()
    }
    
    // MARK: - Plugin Protocol
    
    public protocol AutoResolvePlugin: AnyObject {
        var info: PluginInfo { get }
        var capabilities: PluginCapability { get }
        var configuration: PluginConfiguration { get set }
        
        func initialize() async throws
        func shutdown() async
        func process(_ input: PluginInput) async throws -> PluginOutput
        func createUI() -> AnyView?
    }
    
    // MARK: - Plugin Info
    
    public struct PluginInfo: Identifiable, Codable {
        public let id: UUID
        public let name: String
        public let version: String
        public let author: String
        public let description: String
        public let category: PluginCategory
        public let iconName: String?
        public let website: URL?
        public let minimumVersion: String
        public let maximumVersion: String?
        
        public enum PluginCategory: String, Codable, CaseIterable {
            case effect = "Effects"
            case transition = "Transitions"
            case generator = "Generators"
            case title = "Titles"
            case colorGrading = "Color Grading"
            case audioEffect = "Audio Effects"
            case analysis = "Analysis"
            case export = "Export"
            case `import` = "Import"
            case workflow = "Workflow"
            case utility = "Utility"
        }
        
        public init(
            name: String,
            version: String = "1.0.0",
            author: String,
            description: String,
            category: PluginCategory
        ) {
            self.id = UUID()
            self.name = name
            self.version = version
            self.author = author
            self.description = description
            self.category = category
            self.iconName = nil
            self.website = nil
            self.minimumVersion = "3.0.0"
            self.maximumVersion = nil
        }
    }
    
    // MARK: - Plugin Capabilities
    
    public struct PluginCapability: OptionSet {
        public let rawValue: Int
        
        public init(rawValue: Int) {
            self.rawValue = rawValue
        }
        
        public static let videoProcessing = PluginCapability(rawValue: 1 << 0)
        public static let audioProcessing = PluginCapability(rawValue: 1 << 1)
        public static let realtimePreview = PluginCapability(rawValue: 1 << 2)
        public static let gpuAccelerated = PluginCapability(rawValue: 1 << 3)
        public static let multiThreaded = PluginCapability(rawValue: 1 << 4)
        public static let batchProcessing = PluginCapability(rawValue: 1 << 5)
        public static let customUI = PluginCapability(rawValue: 1 << 6)
        public static let keyframeAnimation = PluginCapability(rawValue: 1 << 7)
        public static let colorManagement = PluginCapability(rawValue: 1 << 8)
        public static let metadata = PluginCapability(rawValue: 1 << 9)
        
        public static let all: PluginCapability = [
            .videoProcessing, .audioProcessing, .realtimePreview,
            .gpuAccelerated, .multiThreaded, .batchProcessing,
            .customUI, .keyframeAnimation, .colorManagement, .metadata
        ]
    }
    
    // MARK: - Plugin Configuration
    
    public struct PluginConfiguration: Codable, Sendable {
        public var parameters: [PluginParameter]
        public var presets: [PluginPreset]
        public var currentPreset: String?
        
        public struct PluginParameter: Codable, Sendable, Identifiable {
            public let id: String
            public let name: String
            public let type: ParameterType
            public var value: ParameterValue
            public let defaultValue: ParameterValue
            public let range: ParameterRange?
            public let options: [String]?
            public let group: String?
            public let isAnimatable: Bool
            
            public init(
                id: String,
                name: String,
                type: ParameterType,
                value: ParameterValue,
                defaultValue: ParameterValue,
                range: ParameterRange? = nil,
                options: [String]? = nil,
                group: String? = nil,
                isAnimatable: Bool = false
            ) {
                self.id = id
                self.name = name
                self.type = type
                self.value = value
                self.defaultValue = defaultValue
                self.range = range
                self.options = options
                self.group = group
                self.isAnimatable = isAnimatable
            }
            
            public enum ParameterType: String, Codable, Sendable {
                case float
                case integer
                case boolean
                case string
                case color
                case point
                case size
                case rect
                case transform
                case dropdown
                case slider
                case curve
            }
            
            public enum ParameterValue: Codable, Sendable {
                case float(Float)
                case integer(Int)
                case boolean(Bool)
                case string(String)
                case color(r: Float, g: Float, b: Float, a: Float)
                case point(x: Float, y: Float)
                case size(width: Float, height: Float)
                case rect(x: Float, y: Float, width: Float, height: Float)
                case transform(TransformValue)
                
                public struct TransformValue: Codable, Sendable {
                    public var position: CGPoint
                    public var scale: CGSize
                    public var rotation: Float
                    public var anchor: CGPoint
                }
            }
            
            public struct ParameterRange: Codable, Sendable {
                public let min: Float
                public let max: Float
                public let step: Float?
            }
        }
        
        public struct PluginPreset: Codable, Sendable, Identifiable {
            public let id: String
            public let name: String
            public let description: String?
            public let parameters: [PluginParameter]
            public let isDefault: Bool
            public let isUserCreated: Bool
        }
        
        public init() {
            self.parameters = []
            self.presets = []
            self.currentPreset = nil
        }
    }
    
    // MARK: - Plugin I/O
    
    public struct PluginInput {
        public let video: CVPixelBuffer?
        public let audio: AVAudioPCMBuffer?
        public let timecode: CMTime
        public let duration: CMTime
        public let metadata: [String: Any]
        public let projectSettings: ProjectSettings
        
        public struct ProjectSettings {
            public let resolution: CGSize
            public let frameRate: Double
            public let colorSpace: String
            public let audioSampleRate: Double
        }
    }
    
    public struct PluginOutput {
        public let video: CVPixelBuffer?
        public let audio: AVAudioPCMBuffer?
        public let metadata: [String: Any]
        public let processingTime: TimeInterval
        
        public init(
            video: CVPixelBuffer? = nil,
            audio: AVAudioPCMBuffer? = nil,
            metadata: [String: Any] = [:],
            processingTime: TimeInterval = 0
        ) {
            self.video = video
            self.audio = audio
            self.metadata = metadata
            self.processingTime = processingTime
        }
    }
    
    // MARK: - Plugin Registration
    
    private struct PluginRegistration {
        let plugin: AutoResolvePlugin
        let bundle: Bundle?
        let isBuiltIn: Bool
        let loadDate: Date
    }
    
    // MARK: - Plugin Loading
    
    public func loadPlugin(at url: URL) async throws {
        logger.info("Loading plugin from: \(url.path)")
        
        await MainActor.run {
            isLoadingPlugins = true
        }
        
        defer {
            Task { @MainActor in
                isLoadingPlugins = false
            }
        }
        
        guard url.pathExtension == "arplugin" else {
            throw PluginError.invalidPluginFormat
        }
        
        let bundle = Bundle(url: url)
        guard let bundle = bundle else {
            throw PluginError.bundleLoadFailed
        }
        
        guard let principalClass = bundle.principalClass as? NSObject.Type else {
            throw PluginError.missingPrincipalClass
        }
        
        guard let plugin = principalClass.init() as? AutoResolvePlugin else {
            throw PluginError.invalidPrincipalClass
        }
        
        try await plugin.initialize()
        
        let registration = PluginRegistration(
            plugin: plugin,
            bundle: bundle,
            isBuiltIn: false,
            loadDate: Date()
        )
        
        await MainActor.run {
            pluginRegistry[plugin.info.id.uuidString] = registration
            loadedPlugins.append(plugin)
        }
        
        logger.info("Successfully loaded plugin: \(plugin.info.name)")
    }
    
    public func unloadPlugin(_ pluginId: UUID) async {
        guard let registration = pluginRegistry[pluginId.uuidString] else { return }
        
        await registration.plugin.shutdown()
        
        await MainActor.run {
            pluginRegistry.removeValue(forKey: pluginId.uuidString)
            loadedPlugins.removeAll { $0.info.id == pluginId }
        }
        
        logger.info("Unloaded plugin: \(registration.plugin.info.name)")
    }
    
    // MARK: - Plugin Discovery
    
    private func scanForPlugins() {
        Task {
            await MainActor.run {
                isLoadingPlugins = true
            }
            
            let pluginURLs = findPluginBundles()
            
            for url in pluginURLs {
                do {
                    try await loadPlugin(at: url)
                } catch {
                    await MainActor.run {
                        pluginErrors.append(PluginError.loadFailed(url: url, error: error))
                    }
                }
            }
            
            await MainActor.run {
                isLoadingPlugins = false
            }
        }
    }
    
    private func findPluginBundles() -> [URL] {
        var urls: [URL] = []
        
        let libraryURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AutoResolve")
            .appendingPathComponent("Plugins")
        
        if let libraryURL = libraryURL {
            do {
                let contents = try FileManager.default.contentsOfDirectory(
                    at: libraryURL,
                    includingPropertiesForKeys: nil
                )
                urls.append(contentsOf: contents.filter { $0.pathExtension == "arplugin" })
            } catch {
                logger.error("Failed to scan plugin directory: \(error)")
            }
        }
        
        if let bundlePluginsURL = Bundle.main.url(forResource: "Plugins", withExtension: nil) {
            do {
                let contents = try FileManager.default.contentsOfDirectory(
                    at: bundlePluginsURL,
                    includingPropertiesForKeys: nil
                )
                urls.append(contentsOf: contents.filter { $0.pathExtension == "arplugin" })
            } catch {
                logger.error("Failed to scan bundle plugins: \(error)")
            }
        }
        
        return urls
    }
    
    // MARK: - Built-in Plugins
    
    private func loadBuiltInPlugins() {
        let builtInPlugins: [AutoResolvePlugin] = [
            GlowEffectPlugin(),
            ChromaticAberrationPlugin(),
            FilmGrainPlugin(),
            LensFlaresPlugin(),
            ParticleSystemPlugin(),
            TextAnimatorPlugin(),
            AudioVisualizerPlugin(),
            ColorMatchPlugin(),
            StabilizerPlugin(),
            DenoiserPlugin()
        ]
        
        for plugin in builtInPlugins {
            Task {
                do {
                    try await plugin.initialize()
                    
                    let registration = PluginRegistration(
                        plugin: plugin,
                        bundle: nil,
                        isBuiltIn: true,
                        loadDate: Date()
                    )
                    
                    await MainActor.run {
                        pluginRegistry[plugin.info.id.uuidString] = registration
                        loadedPlugins.append(plugin)
                    }
                } catch {
                    logger.error("Failed to initialize built-in plugin: \(plugin.info.name)")
                }
            }
        }
    }
    
    // MARK: - Plugin Execution
    
    public func executePlugin(
        _ pluginId: UUID,
        with input: PluginInput
    ) async throws -> PluginOutput {
        guard let registration = pluginRegistry[pluginId.uuidString] else {
            throw PluginError.pluginNotFound
        }
        
        return try await registration.plugin.process(input)
    }
    
    // MARK: - Plugin Chain
    
    public func executePluginChain(
        _ pluginIds: [UUID],
        with input: PluginInput
    ) async throws -> PluginOutput {
        var currentInput = input
        var finalOutput: PluginOutput?
        
        for pluginId in pluginIds {
            let output = try await executePlugin(pluginId, with: currentInput)
            
            currentInput = PluginInput(
                video: output.video ?? currentInput.video,
                audio: output.audio ?? currentInput.audio,
                timecode: currentInput.timecode,
                duration: currentInput.duration,
                metadata: output.metadata.merging(currentInput.metadata) { new, _ in new },
                projectSettings: currentInput.projectSettings
            )
            
            finalOutput = output
        }
        
        guard let output = finalOutput else {
            throw PluginError.chainExecutionFailed
        }
        
        return output
    }
}

// MARK: - Plugin Errors

public enum PluginError: LocalizedError {
    case invalidPluginFormat
    case bundleLoadFailed
    case missingPrincipalClass
    case invalidPrincipalClass
    case initializationFailed
    case pluginNotFound
    case chainExecutionFailed
    case loadFailed(url: URL, error: Error)
    
    public var errorDescription: String? {
        switch self {
        case .invalidPluginFormat:
            return "Invalid plugin format"
        case .bundleLoadFailed:
            return "Failed to load plugin bundle"
        case .missingPrincipalClass:
            return "Plugin bundle missing principal class"
        case .invalidPrincipalClass:
            return "Invalid plugin principal class"
        case .initializationFailed:
            return "Plugin initialization failed"
        case .pluginNotFound:
            return "Plugin not found"
        case .chainExecutionFailed:
            return "Plugin chain execution failed"
        case .loadFailed(let url, let error):
            return "Failed to load plugin at \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }
}

// MARK: - Base Plugin Class

open class BasePlugin: PluginManager.AutoResolvePlugin {
    public let info: PluginManager.PluginInfo
    public let capabilities: PluginManager.PluginCapability
    public var configuration: PluginManager.PluginConfiguration
    
    public init(
        info: PluginManager.PluginInfo,
        capabilities: PluginManager.PluginCapability,
        configuration: PluginManager.PluginConfiguration = PluginManager.PluginConfiguration()
    ) {
        self.info = info
        self.capabilities = capabilities
        self.configuration = configuration
    }
    
    // Protocol conformance methods
    public func initialize() async throws {
        // Default implementation - subclasses can override
        try await initializeAsync()
    }
    
    public func process(_ input: PluginManager.PluginInput) async throws -> PluginManager.PluginOutput {
        // Default implementation - subclasses should override
        return try await processAsync(input)
    }
    
    
    // Async versions for actual implementation
    open func initializeAsync() async throws {
        // Override in subclasses
    }
    
    open func shutdown() async {
        // Override in subclasses
    }
    
    open func processAsync(_ input: PluginManager.PluginInput) async throws -> PluginManager.PluginOutput {
        // Override in subclasses
        return PluginManager.PluginOutput(
            video: input.video,
            audio: input.audio,
            metadata: input.metadata,
            processingTime: 0
        )
    }
    
    open func createUI() -> AnyView? {
        // Override in subclasses
        return nil
    }
}

// MARK: - Built-in Effect Plugins

class GlowEffectPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Glow Effect",
                author: "AutoResolve",
                description: "Adds a customizable glow effect to video",
                category: .effect
            ),
            capabilities: [.videoProcessing, .realtimePreview, .gpuAccelerated, .keyframeAnimation] as PluginManager.PluginCapability
        )
        
        // Break up complex array literal for faster compilation
        let intensityParam = PluginManager.PluginConfiguration.PluginParameter(
            id: "intensity",
            name: "Intensity",
            type: .slider,
            value: .float(0.5),
            defaultValue: .float(0.5),
            range: PluginManager.PluginConfiguration.PluginParameter.ParameterRange(min: 0, max: 1, step: nil),
            options: nil,
            group: "Basic",
            isAnimatable: true
        )
        
        let radiusParam = PluginManager.PluginConfiguration.PluginParameter(
            id: "radius",
            name: "Radius",
            type: .slider,
            value: .float(10),
            defaultValue: .float(10),
            range: PluginManager.PluginConfiguration.PluginParameter.ParameterRange(min: 1, max: 100, step: nil),
            options: nil,
            group: "Basic",
            isAnimatable: true
        )
        
        let colorParam = PluginManager.PluginConfiguration.PluginParameter(
            id: "color",
            name: "Glow Color",
            type: .color,
            value: .color(r: 1, g: 1, b: 1, a: 1),
            defaultValue: .color(r: 1, g: 1, b: 1, a: 1),
            range: nil,
            options: nil,
            group: "Color",
            isAnimatable: true
        )
        
        configuration.parameters = [intensityParam, radiusParam, colorParam]
    }
    
    open override func processAsync(_ input: PluginManager.PluginInput) async throws -> PluginManager.PluginOutput {
        let startTime = Date()
        
        guard let pixelBuffer = input.video else {
            return PluginManager.PluginOutput(
                video: nil,
                audio: input.audio,
                metadata: input.metadata,
                processingTime: 0
            )
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        let intensity = configuration.parameters.first { $0.id == "intensity" }?.value
        let radius = configuration.parameters.first { $0.id == "radius" }?.value
        
        var outputImage = ciImage
        
        if let glowFilter = CIFilter(name: "CIGaussianBlur") {
            glowFilter.setValue(ciImage, forKey: kCIInputImageKey)
            
            if case .float(let r) = radius {
                glowFilter.setValue(r, forKey: kCIInputRadiusKey)
            }
            
            if let blurred = glowFilter.outputImage,
               let compositeFilter = CIFilter(name: "CIAdditionCompositing") {
                compositeFilter.setValue(blurred, forKey: kCIInputImageKey)
                compositeFilter.setValue(ciImage, forKey: kCIInputBackgroundImageKey)
                
                if let composite = compositeFilter.outputImage {
                    outputImage = composite
                }
            }
        }
        
        let context = CIContext()
        context.render(outputImage, to: pixelBuffer)
        
        return PluginManager.PluginOutput(
            video: pixelBuffer,
            audio: input.audio,
            metadata: input.metadata,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }
}

class ChromaticAberrationPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Chromatic Aberration",
                author: "AutoResolve",
                description: "Simulates lens chromatic aberration",
                category: .effect
            ),
            capabilities: [.videoProcessing, .realtimePreview, .gpuAccelerated] as PluginManager.PluginCapability
        )
    }
}

class FilmGrainPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Film Grain",
                author: "AutoResolve",
                description: "Adds realistic film grain to video",
                category: .effect
            ),
            capabilities: [.videoProcessing, .realtimePreview, .gpuAccelerated] as PluginManager.PluginCapability
        )
    }
}

class LensFlaresPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Lens Flares",
                author: "AutoResolve",
                description: "Creates cinematic lens flares",
                category: .effect
            ),
            capabilities: [.videoProcessing, .realtimePreview, .gpuAccelerated, .keyframeAnimation] as PluginManager.PluginCapability
        )
    }
}

class ParticleSystemPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Particle System",
                author: "AutoResolve",
                description: "Advanced particle effects generator",
                category: .generator
            ),
            capabilities: [.videoProcessing, .realtimePreview, .gpuAccelerated, .keyframeAnimation] as PluginManager.PluginCapability
        )
    }
}

class TextAnimatorPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Text Animator",
                author: "AutoResolve",
                description: "Animated text and titles",
                category: .title
            ),
            capabilities: [.videoProcessing, .realtimePreview, .keyframeAnimation, .customUI] as PluginManager.PluginCapability
        )
    }
}

class AudioVisualizerPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Audio Visualizer",
                author: "AutoResolve",
                description: "Creates visual representations of audio",
                category: .generator
            ),
            capabilities: [.audioProcessing, .videoProcessing, .realtimePreview, .gpuAccelerated] as PluginManager.PluginCapability
        )
    }
}

class ColorMatchPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Color Match",
                author: "AutoResolve",
                description: "Matches colors between clips",
                category: .colorGrading
            ),
            capabilities: [.videoProcessing, .colorManagement, .gpuAccelerated] as PluginManager.PluginCapability
        )
    }
}

class StabilizerPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Video Stabilizer",
                author: "AutoResolve",
                description: "Removes camera shake from footage",
                category: .effect
            ),
            capabilities: [.videoProcessing, .gpuAccelerated, .multiThreaded] as PluginManager.PluginCapability
        )
    }
}

class DenoiserPlugin: BasePlugin {
    init() {
        super.init(
            info: PluginManager.PluginInfo(
                name: "Denoiser",
                author: "AutoResolve",
                description: "Removes noise from video and audio",
                category: .effect
            ),
            capabilities: [.videoProcessing, .audioProcessing, .gpuAccelerated] as PluginManager.PluginCapability
        )
    }
}

// MARK: - Logger

import os

