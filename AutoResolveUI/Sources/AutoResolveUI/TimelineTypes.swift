// Imports
import Foundation
import SwiftUI
import AVFoundation
import Combine

// MARK: - CMTime Codable Support

extension CMTime: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let seconds = try container.decode(TimeInterval.self, forKey: .seconds)
        let timescale = try container.decode(CMTimeScale.self, forKey: .timescale)
        self = CMTimeMakeWithSeconds(seconds, preferredTimescale: timescale)
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(CMTimeGetSeconds(self), forKey: .seconds)
        try container.encode(self.timescale, forKey: .timescale)
    }
    
    enum CodingKeys: String, CodingKey {
        case seconds
        case timescale
    }
}

// MARK: - Timeline Core Types

// Type alias for compatibility
public typealias TimelineClip = SimpleTimelineClip

public struct Timeline: Codable, Sendable {
    public let id: UUID
    public var name: String
    public var tracks: [TimelineTrack]
    public var duration: CMTime
    public var frameRate: Double
    public var resolution: CGSize
    
    public init(id: UUID = UUID(), name: String = "Untitled Timeline", tracks: [TimelineTrack] = [], duration: CMTime = .zero, frameRate: Double = 30.0, resolution: CGSize = CGSize(width: 1920, height: 1080)) {
        self.id = id
        self.name = name
        self.tracks = tracks
        self.duration = duration
        self.frameRate = frameRate
        self.resolution = resolution
    }
}

public struct LegacyUITimelineClip: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public var url: URL
    public var inPoint: CMTime
    public var outPoint: CMTime
    public var duration: CMTime
    public var name: String
    public var trackIndex: Int
    public var startTime: CMTime = .zero
    public var timelineStartTime: TimeInterval = 0  // For compatibility with ProjectVideoClip
    public var thumbnailData: Data?
    public var colorHex: String = "0066CC"  // Hex color instead of Color for Codable
    public var type: ClipType = .video
    public var thumbnail: Data?
    public var isSelected: Bool = false
    public var volume: Float = 1.0
    
    public enum ClipType: String, Codable, Sendable {
        case video, audio, image, text, effect, title, transition
    }
    
    public init(id: UUID = UUID(), url: URL, inPoint: CMTime = .zero, outPoint: CMTime = .zero, duration: CMTime = .zero, name: String = "", trackIndex: Int = 0, startTime: CMTime = .zero, timelineStartTime: TimeInterval = 0, thumbnailData: Data? = nil, colorHex: String = "0066CC", type: ClipType = .video) {
        self.id = id
        self.url = url
        self.inPoint = inPoint
        self.outPoint = outPoint
        self.duration = duration
        self.name = name
        self.trackIndex = trackIndex
        self.startTime = startTime
        self.timelineStartTime = timelineStartTime
        self.thumbnailData = thumbnailData
        self.thumbnail = thumbnailData
        self.colorHex = colorHex
        self.type = type
    }
    
    // Convenience initializer accepting TimeInterval values
    public init(id: UUID = UUID(), url: URL, inPoint: TimeInterval = 0, outPoint: TimeInterval = 0, duration: TimeInterval = 0, name: String = "", trackIndex: Int = 0, startTime: TimeInterval = 0, timelineStartTime: TimeInterval = 0, thumbnailData: Data? = nil, colorHex: String = "0066CC", type: ClipType = .video) {
        self.id = id
        self.url = url
        self.inPoint = CMTime(seconds: inPoint, preferredTimescale: 600)
        self.outPoint = CMTime(seconds: outPoint, preferredTimescale: 600)
        self.duration = CMTime(seconds: duration, preferredTimescale: 600)
        self.name = name
        self.trackIndex = trackIndex
        self.startTime = CMTime(seconds: startTime, preferredTimescale: 600)
        self.timelineStartTime = timelineStartTime
        self.thumbnailData = thumbnailData
        self.thumbnail = thumbnailData
        self.colorHex = colorHex
        self.type = type
    }
    
    // Computed property for SwiftUI Color
    public var color: Color {
        switch colorHex {
        case "0066CC": return .blue
        case "FF0000": return .red
        case "00FF00": return .green
        case "FFFF00": return .yellow
        case "FF00FF": return .purple
        case "00FFFF": return .cyan
        default: return .blue
        }
    }
    
    // Computed property for end time
    public var endTime: CMTime {
        return CMTimeAdd(startTime, duration)
    }
    
    // Computed property for backward compatibility
    public var sourceURL: URL {
        get { return url }
        set { url = newValue }
    }
    
    public nonisolated func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    public nonisolated static func == (lhs: LegacyUITimelineClip, rhs: LegacyUITimelineClip) -> Bool {
        lhs.id == rhs.id
    }
    
    /// Check if clip contains a time point
    public func contains(time: TimeInterval) -> Bool {
        let timeInSeconds = CMTimeGetSeconds(startTime)
        let endTimeInSeconds = CMTimeGetSeconds(endTime)
        return time >= timeInSeconds && time <= endTimeInSeconds
    }
    
    /// Check if clip overlaps with time range
    public func overlaps(start: TimeInterval, end: TimeInterval) -> Bool {
        let startTimeInSeconds = CMTimeGetSeconds(startTime)
        let endTimeInSeconds = CMTimeGetSeconds(endTime)
        return startTimeInSeconds < end && endTimeInSeconds > start
    }
}

public struct TimelineTrack: Codable, Sendable, Identifiable {
    public let id: UUID
    public var name: String
    public var type: TrackType
    public var index: Int
    public var height: Double
    public var isLocked: Bool
    public var isMuted: Bool
    public var clips: [LegacyUITimelineClip]
    
    public enum TrackType: String, Codable, CaseIterable, Sendable {
        case video = "V"
        case audio = "A"
        case title = "T"
        case effect = "E"
        case director = "D"
        case transcription = "TR"
        
        public var displayName: String {
            switch self {
            case .video: return "Video"
            case .audio: return "Audio"
            case .title: return "Title"
            case .effect: return "Effect"
            case .director: return "Director"
            case .transcription: return "Transcription"
            }
        }
    }
    
    public init(id: UUID = UUID(), name: String = "", type: TrackType, index: Int = 0, height: Double = 60, isLocked: Bool = false, isMuted: Bool = false, clips: [LegacyUITimelineClip] = []) {
        self.id = id
        self.name = name
        self.type = type
        self.index = index
        self.height = height
        self.isLocked = isLocked
        self.isMuted = isMuted
        self.clips = clips
    }
    
    public mutating func addClip(_ clip: LegacyUITimelineClip) {
        clips.append(clip)
    }
    
    public mutating func removeClip(id: UUID) {
        clips.removeAll { $0.id == id }
    }
}

public struct TimelineMarker: Identifiable, Codable {
    public let id: UUID
    public var time: TimeInterval
    public var type: MarkerType
    public var name: String
    
    public enum MarkerType: String, Codable, Sendable {
        case silence
        case cut
        case bookmark
    }
    
    public init(id: UUID = UUID(), time: TimeInterval, type: MarkerType, name: String = "") {
        self.id = id
        self.time = time
        self.type = type
        self.name = name
    }
}


