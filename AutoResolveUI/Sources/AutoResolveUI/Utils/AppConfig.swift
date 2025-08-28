// AUTORESOLVE V3.0 - APPLICATION CONFIGURATION
// Centralized configuration management

import Foundation
import SwiftUI

/// Global application configuration
public struct AppConfig {
    
    // MARK: - API Configuration
    
    public struct API {
        public static let baseURL = URL(string: "http://localhost:8000/api")!
        public static let timeout: TimeInterval = 30
        public static let maxRetries = 3
        public static let retryDelay: TimeInterval = 1.0
        
        // WebSocket
        public static let websocketURL = URL(string: "ws://localhost:8000/ws/progress")!
        public static let websocketReconnectDelay: TimeInterval = 5.0
        
        // Endpoints
        public struct Endpoints {
            public static let auth = "auth/login"
            public static let pipeline = "pipeline"
            public static let telemetry = "telemetry/metrics"
            public static let export = "export"
            public static let health = "health"
        }
        
        // Headers
        public static let defaultHeaders: [String: String] = [
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AutoResolve/3.0"
        ]
    }
    
    // MARK: - Cache Configuration
    
    public struct Cache {
        public static let maxMemorySize = 50 * 1024 * 1024  // 50MB
        public static let maxDiskSize = 200 * 1024 * 1024   // 200MB
        public static let defaultTTL: TimeInterval = 300     // 5 minutes
        public static let cleanupInterval: TimeInterval = 60 // 1 minute
        public static let maxCacheItems = 1000
    }
    
    // MARK: - Performance Limits
    
    public struct Performance {
        public static let maxConcurrentOperations = 4
        public static let maxMemoryUsage = 4 * 1024 * 1024 * 1024  // 4GB
        public static let targetProcessingSpeed: TimeInterval = 5.0  // 5s per minute of video
        public static let chunkSize = 1024 * 1024  // 1MB for file operations
        
        // Timeouts
        public static let operationTimeout: TimeInterval = 120
        public static let backgroundTaskTimeout: TimeInterval = 600
        
        // Thread pool
        public static let workerThreads = ProcessInfo.processInfo.processorCount
        public static let backgroundQueueQoS: DispatchQoS = .userInitiated
    }
    
    // MARK: - File Handling
    
    public struct Files {
        public static let maxUploadSize: Int64 = 10 * 1024 * 1024 * 1024  // 10GB
        public static let maxBatchFiles = 100
        public static let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AutoResolve")
        
        // Supported formats
        public static let supportedVideoFormats = ["mp4", "mov", "avi", "mkv", "m4v", "webm"]
        public static let supportedAudioFormats = ["mp3", "wav", "aac", "m4a", "flac"]
        public static let supportedImageFormats = ["jpg", "jpeg", "png", "gif", "heic"]
        
        // Export formats
        public static let exportFormats = ["mp4", "mov", "prores", "fcpxml", "edl"]
        public static let exportCodecs = ["h264", "h265", "prores422", "prores4444"]
    }
    
    // MARK: - UI Configuration
    
    public struct UI {
        // Layout
        public static let sidebarWidth: CGFloat = 380
        public static let inspectorWidth: CGFloat = 380
        public static let minWindowWidth: CGFloat = 1200
        public static let minWindowHeight: CGFloat = 800
        
        // Timeline
        public static let timelineHeight: CGFloat = 300
        public static let trackHeight: CGFloat = 60
        public static let thumbnailHeight: CGFloat = 40
        public static let defaultZoom: Double = 1.0
        public static let maxZoom: Double = 10.0
        public static let minZoom: Double = 0.1
        
        // Colors
        public struct Colors {
            public static let background = Color(white: 0.11)
            public static let surface = Color(white: 0.15)
            public static let border = Color(white: 0.25)
            public static let text = Color.white
            public static let textSecondary = Color(white: 0.7)
            public static let accent = Color.blue
            public static let success = Color.green
            public static let warning = Color.orange
            public static let error = Color.red
            
            // Neural overlay colors
            public static let neuralOverlay = Color.blue.opacity(0.4)
            public static let confidenceHigh = Color.green
            public static let confidenceMedium = Color.yellow
            public static let confidenceLow = Color.red
        }
        
        // Animation
        public static let animationDuration: TimeInterval = 0.3
        public static let animationCurve = Animation.smooth
    }
    
    // MARK: - Neural Analysis
    
    public struct Neural {
        public static let defaultEmbedder = "clip"
        public static let vjepaPath = "/Users/hawzhin/vjepa2-vitl-fpc64-256"
        public static let confidenceThreshold: Double = 0.8
        public static let minSilenceDuration: TimeInterval = 0.5
        public static let sceneChangeThreshold: Double = 0.3
        
        // Processing gates
        public static let maxProcessingTime: TimeInterval = 5.0  // per minute
        public static let maxMemoryUsage = 892 * 1024 * 1024  // 892MB
    }
    
    // MARK: - Security
    
    public struct Security {
        public static let tokenExpiration: TimeInterval = 3600  // 1 hour
        public static let maxLoginAttempts = 5
        public static let lockoutDuration: TimeInterval = 300  // 5 minutes
        public static let requireBiometric = false
        public static let allowedMediaRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Videos")
    }
    
    // MARK: - Debug
    
    public struct Debug {
        #if DEBUG
        public static let isDebugMode = true
        public static let logLevel: LogLevel = .debug
        public static let mockBackend = false
        public static let showPerformanceOverlay = true
        #else
        public static let isDebugMode = false
        public static let logLevel: LogLevel = .info
        public static let mockBackend = false
        public static let showPerformanceOverlay = false
        #endif
    }
    
    // MARK: - Feature Flags
    
    public struct Features {
        public static let enableNeuralAnalysis = true
        public static let enableBRollSelection = true
        public static let enableColorGrading = true
        public static let enableAudioProcessing = true
        public static let enableMultiCam = true
        public static let enableCloudSync = false
        public static let enableCollaboration = false
    }
    
    // MARK: - User Defaults Keys
    
    public struct UserDefaultsKeys {
        public static let lastProject = "lastProject"
        public static let recentProjects = "recentProjects"
        public static let windowFrame = "windowFrame"
        public static let sidebarVisible = "sidebarVisible"
        public static let inspectorVisible = "inspectorVisible"
        public static let selectedEmbedder = "selectedEmbedder"
        public static let exportPreset = "exportPreset"
        public static let autoSaveEnabled = "autoSaveEnabled"
        public static let autoSaveInterval = "autoSaveInterval"
    }
    
    // MARK: - Validation
    
    public struct Validation {
        public static let minPasswordLength = 8
        public static let maxPasswordLength = 128
        public static let minUsernameLength = 3
        public static let maxUsernameLength = 20
        public static let maxFileNameLength = 255
        public static let maxPathLength = 4096
    }
    
    // MARK: - Export Presets
    
    public struct ExportPresets {
        public static let youtube1080p = ExportPreset(
            name: "YouTube 1080p",
            format: "mp4",
            codec: "h264",
            resolution: (1920, 1080),
            frameRate: 30,
            bitrate: 8000000
        )
        
        public static let youtube4K = ExportPreset(
            name: "YouTube 4K",
            format: "mp4",
            codec: "h264",
            resolution: (3840, 2160),
            frameRate: 30,
            bitrate: 35000000
        )
        
        public static let proRes422 = ExportPreset(
            name: "ProRes 422",
            format: "mov",
            codec: "prores422",
            resolution: nil,  // Use source
            frameRate: nil,   // Use source
            bitrate: nil      // ProRes has fixed bitrate
        )
        
        public static let all = [youtube1080p, youtube4K, proRes422]
    }
    
    public struct ExportPreset {
        let name: String
        let format: String
        let codec: String
        let resolution: (width: Int, height: Int)?
        let frameRate: Double?
        let bitrate: Int?
    }
}

// MARK: - Environment Keys

public struct ConfigEnvironmentKey: EnvironmentKey {
    public static let defaultValue = AppConfig.self
}

public extension EnvironmentValues {
    var appConfig: AppConfig.Type {
        get { self[ConfigEnvironmentKey.self] }
        set { self[ConfigEnvironmentKey.self] = newValue }
    }
}

// MARK: - Logging

public enum LogLevel: Int, Comparable {
    case debug = 0
    case info = 1
    case warning = 2
    case error = 3
    case critical = 4
    
    public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
    
    var emoji: String {
        switch self {
        case .debug: return "🔍"
        case .info: return "ℹ️"
        case .warning: return "⚠️"
        case .error: return "❌"
        case .critical: return "🚨"
        }
    }
}

// MARK: - Global Functions

public func logDebug(_ message: String, category: LogCategory = .general) {
    guard AppConfig.Debug.logLevel <= .debug else { return }
    print("\(LogLevel.debug.emoji) [\(category.rawValue)] \(message)")
}

public func logInfo(_ message: String, category: LogCategory = .general) {
    guard AppConfig.Debug.logLevel <= .info else { return }
    print("\(LogLevel.info.emoji) [\(category.rawValue)] \(message)")
}

public func logWarning(_ message: String, category: LogCategory = .general) {
    guard AppConfig.Debug.logLevel <= .warning else { return }
    print("\(LogLevel.warning.emoji) [\(category.rawValue)] \(message)")
}

public func logError(_ message: String, category: LogCategory = .general) {
    guard AppConfig.Debug.logLevel <= .error else { return }
    print("\(LogLevel.error.emoji) [\(category.rawValue)] \(message)")
}

public func logCritical(_ message: String, category: LogCategory = .general) {
    guard AppConfig.Debug.logLevel <= .critical else { return }
    print("\(LogLevel.critical.emoji) [\(category.rawValue)] \(message)")
}

public enum LogCategory: String {
    case general = "General"
    case api = "API"
    case cache = "Cache"
    case timeline = "Timeline"
    case neural = "Neural"
    case export = "Export"
    case storage = "Storage"
    case auth = "Auth"
    case performance = "Performance"
}