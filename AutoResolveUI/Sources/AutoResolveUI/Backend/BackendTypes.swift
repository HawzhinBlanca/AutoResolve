// Imports
import Foundation
import Combine
// MARK: - Backend Types

/// Video format options for project settings
public enum VideoFormat: String, Codable, CaseIterable {
    case hd = "hd"
    case uhd4k = "4k"
    case uhd8k = "8k"
    case cinema4k = "cinema_4k"
    case custom = "custom"
}

/// Color space options for project settings
public enum ColorSpace: String, Codable, CaseIterable {
    case rec709 = "Rec. 709"
    case rec2020 = "Rec. 2020"
    case p3 = "P3"
    case aces = "ACES"
    case log = "Log"
}

/// Settings for backend silence detection
public struct BackendSilenceDetectionSettings: Codable, Sendable {
    public let threshold: Double
    public let minDuration: Double
    public let padding: Double
    
    public init(threshold: Double, minDuration: Double, padding: Double) {
        self.threshold = threshold
        self.minDuration = minDuration
        self.padding = padding
    }
}

/// Render quality settings
public enum RenderQuality: String, Codable, CaseIterable {
    case draft = "draft"
    case good = "good"
    case best = "best"
}

/// Preview quality settings
public enum PreviewQuality: String, Codable, CaseIterable {
    case quarter = "quarter"
    case half = "half"
    case full = "full"
}

/// Backend time range for API communication
public struct BackendTimeRange: Codable, Sendable {
    public let start: Double
    public let end: Double
    
    public init(start: Double, end: Double) {
        self.start = start
        self.end = end
    }
}

/// Backend B-roll settings
public struct BackendBRollSettings: Codable, Sendable {
    public let maxResults: Int
    public let threshold: Double
    public let brollDirectory: String?
    
    public init(maxResults: Int, threshold: Double, brollDirectory: String? = nil) {
        self.maxResults = maxResults
        self.threshold = threshold
        self.brollDirectory = brollDirectory
    }
}

// TimeRange definition
public struct TimeRange: Codable, Sendable, Hashable {
    public let start: TimeInterval
    public let end: TimeInterval
    
    public init(start: TimeInterval, end: TimeInterval) {
        self.start = start
        self.end = end
    }
    
    public var duration: TimeInterval {
        end - start
    }
}

// Project type alias for compatibility
public typealias Project = BackendVideoProjectStore

// BackendVideoProjectStore - simple project container
public class BackendVideoProjectStore: ObservableObject {
    public let id: UUID
    @Published public var name: String
    @Published public var timeline: TimelineModel
    @Published public var createdAt: Date = Date()
    @Published public var modifiedAt: Date = Date()
    @Published public var cacheDirectory: URL?
    @Published public var metadata: ProjectMetadata = ProjectMetadata()
    @Published public var settings: ProjectSettings = ProjectSettings()
    
    public init(id: UUID = UUID(), name: String = "Untitled Project", timeline: TimelineModel? = nil) {
        self.id = id
        self.name = name
        self.timeline = timeline ?? TimelineModel()
    }
}

// MARK: - Project Metadata & Settings used by Inspector
public struct ProjectMetadata: Codable, Sendable {
    public var notes: String = ""
    public var creator: String = ""
    public var keywords: String = ""
    public init() {}
}

public struct ProjectSettings: Codable, Sendable {
    // Timeline/Video
    public var videoFormat: VideoFormat = .uhd4k
    public var width: Int = 1920
    public var height: Int = 1080
    public var frameRate: Double = 30.0
    public var pixelAspectRatio: Double = 1.0
    public var colorSpace: ColorSpace = .rec709
    public var workingColorSpace: String = "Rec. 709"
    public var gamma: Double = 2.2
    public var useColorManagement: Bool = false
    public var renderQuality: RenderQuality = .good
    public var motionBlurAmount: Double = 0.0

    // Audio
    public var audioSampleRate: Int = 48000
    public var audioBitDepth: Int = 24
    public var audioChannels: Int = 2
    public var audioOutputDevice: String = "default"
    public var masterVolume: Double = 1.0
    public var audioBufferSize: Int = 256
    public var realtimeAudio: Bool = true
    public var referenceLevelDB: Int = -20

    // Performance
    public var previewQuality: PreviewQuality = .full
    public var renderThreads: Int = 0 // Auto
    public var useGPUAcceleration: Bool = true
    public var backgroundRendering: Bool = false
    public var diskCacheLocation: URL? = nil

    public init() {}
}

public struct BRollSelection: Codable, Sendable {
    public let cutIndex: Int
    public let timeRange: TimeRange
    public let brollClipPath: String
    public let confidence: Double
    public let reason: String?
    
    public init(cutIndex: Int, timeRange: TimeRange, brollClipPath: String, confidence: Double = 0.5, reason: String? = nil) {
        self.cutIndex = cutIndex
        self.timeRange = timeRange
        self.brollClipPath = brollClipPath
        self.confidence = confidence
        self.reason = reason
    }
}

// MARK: - Advanced Timeline Management Types
public struct CreateProjectResponse: Codable {
    public let status: String
    public let project_id: String
}

public struct MoveClipResponse: Codable {
    public let status: String
    public let collisions: [String]
}

public struct AddClipResponse: Codable {
    public let status: String
    public let clip_id: String
    public let collisions: Bool
}

public struct DeleteClipResponse: Codable {
    public let status: String
    public let clip_id: String
}

// MARK: - Timeline Persistence Types
public struct TimelineSaveResponse: Codable {
    public let status: String
    public let project_name: String
    public let path: String
    public let clips_count: Int
}

public struct TimelineLoadResponse: Codable {
    public let status: String
    public let timeline: TimelineData
}

public struct TimelineData: Codable {
    public let version: String
    public let project_name: String
    public let saved_at: String
    public let clips: [[String: String]]
    public let metadata: [String: String]
    public let settings: [String: String]
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(String.self, forKey: .version)
        project_name = try container.decode(String.self, forKey: .project_name)
        saved_at = try container.decode(String.self, forKey: .saved_at)
        metadata = try container.decode([String: String].self, forKey: .metadata)
        settings = try container.decode([String: String].self, forKey: .settings)
        clips = (try? container.decode([[String: String]].self, forKey: .clips)) ?? []
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(version, forKey: .version)
        try container.encode(project_name, forKey: .project_name)
        try container.encode(saved_at, forKey: .saved_at)
        try container.encode(metadata, forKey: .metadata)
        try container.encode(settings, forKey: .settings)
        try container.encode(clips, forKey: .clips)
    }
    
    public init(version: String, project_name: String, saved_at: String, clips: [[String: String]], metadata: [String: String], settings: [String: String]) {
        self.version = version
        self.project_name = project_name
        self.saved_at = saved_at
        self.clips = clips
        self.metadata = metadata
        self.settings = settings
    }
    
    private enum CodingKeys: String, CodingKey {
        case version, project_name, saved_at, clips, metadata, settings
    }
}

public struct TimelineListResponse: Codable {
    public let status: String
    public let timelines: [BackendTimelineInfo]
}

public struct BackendTimelineInfo: Codable {
    public let project_name: String
    public let saved_at: String?
    public let clips_count: Int
    public let file_name: String
    
    public init(project_name: String, saved_at: String?, clips_count: Int, file_name: String) {
        self.project_name = project_name
        self.saved_at = saved_at
        self.clips_count = clips_count
        self.file_name = file_name
    }
}

// Helper for generic JSON decoding
// Local flexible JSON wrapper to avoid cross-file name clashes with WebSocket's AnyCodable
// (Removed flexible JSON wrapper here to avoid duplicate definitions)

public struct ProcessingTelemetry: Codable, Sendable {
    public let processingTime: Double
    public let realtimeFactor: Double
    public let memoryUsed: Double
    public let cpuUsage: Double
    
    public init(processingTime: Double, realtimeFactor: Double, memoryUsed: Double, cpuUsage: Double) {
        self.processingTime = processingTime
        self.realtimeFactor = realtimeFactor
        self.memoryUsed = memoryUsed
        self.cpuUsage = cpuUsage
    }
}

public struct MemoryUsage: Codable, Sendable {
    public let total: Double
    public let used: Double
    public let free: Double
    
    public init(total: Double, used: Double, free: Double) {
        self.total = total
        self.used = used
        self.free = free
    }
}

public struct GPUInfo: Codable, Sendable {
    public let name: String
    public let utilization: Double
    public let memoryUsed: Double
    
    public init(name: String, utilization: Double, memoryUsed: Double) {
        self.name = name
        self.utilization = utilization
        self.memoryUsed = memoryUsed
    }
}

public struct DiskSpace: Codable, Sendable {
    public let total: Double
    public let free: Double
    
    public init(total: Double, free: Double) {
        self.total = total
        self.free = free
    }
}

public struct BRollSettings: Codable, Sendable {
    public let brollDirectory: String
    public let maxResults: Int
    public let minDuration: Double
    public let maxDuration: Double
    
    public init(brollDirectory: String, maxResults: Int, minDuration: Double, maxDuration: Double) {
        self.brollDirectory = brollDirectory
        self.maxResults = maxResults
        self.minDuration = minDuration
        self.maxDuration = maxDuration
    }
}

public struct SilenceDetectionSettings: Codable, Sendable {
    public let threshold: Double
    public let minDuration: Double
    
    public init(threshold: Double, minDuration: Double) {
        self.threshold = threshold
        self.minDuration = minDuration
    }
}

public struct ResolveProjectResult: Codable, Sendable {
    public let success: Bool
    public let projectPath: String
    public let error: String?
    
    public init(success: Bool, projectPath: String, error: String?) {
        self.success = success
        self.projectPath = projectPath
        self.error = error
    }
}

public struct BRollSelectionResult: Codable, Sendable {
    public let selections: [BRollSelection]
    public let success: Bool
    public let error: String?
    
    public init(selections: [BRollSelection], success: Bool, error: String?) {
        self.selections = selections
        self.success = success
        self.error = error
    }
}

public struct SilenceDetectionResult: Codable, Sendable {
    public let silenceSegments: [TimeRange]
    public let success: Bool
    public let error: String?
    
    public init(silenceSegments: [TimeRange], success: Bool, error: String?) {
        self.silenceSegments = silenceSegments
        self.success = success
        self.error = error
    }
}

public struct SystemStatus: Codable, Sendable {
    public let status: String
    public let version: String
    public let uptime: Double
    public let memoryUsage: MemoryUsage
    public let gpuInfo: GPUInfo?
    public let diskSpace: DiskSpace
    public let activeOperations: [String]
    
    public init(status: String, version: String, uptime: Double, memoryUsage: MemoryUsage, gpuInfo: GPUInfo?, diskSpace: DiskSpace, activeOperations: [String]) {
        self.status = status
        self.version = version
        self.uptime = uptime
        self.memoryUsage = memoryUsage
        self.gpuInfo = gpuInfo
        self.diskSpace = diskSpace
        self.activeOperations = activeOperations
    }
}

public struct ProcessingHistoryItem: Codable, Sendable {
    public let id: UUID
    public let date: Date
    public let status: String
    public let processingTime: Double
    public let videoName: String
    
    public init(id: UUID = UUID(), date: Date = Date(), status: String, processingTime: Double, videoName: String) {
        self.id = id
        self.date = date
        self.status = status
        self.processingTime = processingTime
        self.videoName = videoName
    }
}

// Progress Update for WebSocket
public struct ProgressUpdate: Codable, Sendable {
    public let progress: Double
    public let message: String
    public let operation: String?
    
    public init(progress: Double, message: String, operation: String? = nil) {
        self.progress = progress
        self.message = message
        self.operation = operation
    }
}

// Performance Metrics
public struct PerformanceMetrics: Codable, Sendable {
    public let latency: Double
    public let throughput: Double
    public let errorRate: Double
    public let queueDepth: Int
    
    public init(latency: Double = 0, throughput: Double = 0, errorRate: Double = 0, queueDepth: Int = 0) {
        self.latency = latency
        self.throughput = throughput
        self.errorRate = errorRate
        self.queueDepth = queueDepth
    }
}

// Motion Vector for analysis
public struct MotionVector: Codable, Sendable {
    public let x: Double
    public let y: Double
    public let magnitude: Double
    
    public init(x: Double, y: Double, magnitude: Double) {
        self.x = x
        self.y = y
        self.magnitude = magnitude
    }
}

// Type aliases for compatibility (renamed to avoid redeclaration conflicts)
public typealias BackendSilenceDetectionResponse = SilenceDetectionResult
public typealias BackendBRollSelectionResponse = BRollSelectionResult
public typealias BackendResolveProjectResponse = ResolveProjectResult

// Continuity Score
public struct ContinuityScore: Codable, Sendable {
    public let time: TimeInterval
    public let score: Double
    
    public init(time: TimeInterval, score: Double) {
        self.time = time
        self.score = score
    }
}
