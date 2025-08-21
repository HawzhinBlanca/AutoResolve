// AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
// Professional Video Project Management System

import Foundation
import SwiftUI
import AVFoundation
import Combine

// MARK: - Core Video Project Structure
struct VideoProject: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var createdAt: Date
    var modifiedAt: Date
    var timeline: Timeline
    var settings: ProjectSettings
    var mediaPool: MediaPool
    var metadata: ProjectMetadata
    var workspaces: [Workspace]
    
    init(name: String = "Untitled Project") {
        self.name = name
        self.createdAt = Date()
        self.modifiedAt = Date()
        self.timeline = Timeline()
        self.settings = ProjectSettings()
        self.mediaPool = MediaPool()
        self.metadata = ProjectMetadata()
        self.workspaces = [Workspace.default]
    }
    
    mutating func updateModifiedDate() {
        modifiedAt = Date()
    }
}

// MARK: - Professional Timeline Structure
struct Timeline: Codable, Equatable {
    var videoTracks: [VideoTrack] = []
    var audioTracks: [AudioTrack] = []
    var effectTracks: [EffectTrack] = []
    var titleTracks: [TitleTrack] = []
    var markerTrack: MarkerTrack = MarkerTrack()
    var duration: TimeInterval = 0
    var frameRate: Double = 23.976
    var timecodeStart: TimeInterval = 0
    var aspectRatio: AspectRatio = .widescreen
    
    init() {
        // Initialize with default tracks
        videoTracks = [VideoTrack(name: "V1"), VideoTrack(name: "V2")]
        audioTracks = [AudioTrack(name: "A1"), AudioTrack(name: "A2")]
        effectTracks = [EffectTrack(name: "Effects")]
        titleTracks = [TitleTrack(name: "Titles")]
    }
}

// MARK: - Track Definitions
struct VideoTrack: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var clips: [VideoClip] = []
    var isEnabled: Bool = true
    var isMuted: Bool = false
    var isLocked: Bool = false
    var height: CGFloat = 80
    var color: CodableColor = CodableColor(.blue)
}

struct AudioTrack: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var clips: [AudioClip] = []
    var isEnabled: Bool = true
    var isMuted: Bool = false
    var isLocked: Bool = false
    var volume: Float = 1.0
    var pan: Float = 0.0
    var height: CGFloat = 60
    var color: CodableColor = CodableColor(.green)
}

struct EffectTrack: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var effects: [EffectClip] = []
    var isEnabled: Bool = true
    var isLocked: Bool = false
    var height: CGFloat = 40
}

struct TitleTrack: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var titles: [TitleClip] = []
    var isEnabled: Bool = true
    var isLocked: Bool = false
    var height: CGFloat = 50
}

struct MarkerTrack: Codable, Equatable {
    var markers: [Marker] = []
    var chapters: [Chapter] = []
}

// MARK: - Clip Definitions
struct VideoClip: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var sourceURL: URL?
    var startTime: TimeInterval = 0
    var duration: TimeInterval = 10
    var timelineStartTime: TimeInterval = 0
    var sourceStartTime: TimeInterval = 0
    var sourceDuration: TimeInterval = 10
    var isEnabled: Bool = true
    var volume: Float = 1.0
    var speed: Double = 1.0
    var effects: [String] = [] // Effect IDs - simplified for now
    var keyframes: [TimeInterval: Double] = [:] // Time -> Value mapping
    var thumbnailCache: [TimeInterval: Data] = [:]
    
    var endTime: TimeInterval {
        timelineStartTime + duration
    }
    
    var timeRange: ClosedRange<TimeInterval> {
        timelineStartTime...endTime
    }
}

struct AudioClip: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var sourceURL: URL?
    var startTime: TimeInterval = 0
    var duration: TimeInterval = 10
    var timelineStartTime: TimeInterval = 0
    var sourceStartTime: TimeInterval = 0
    var isEnabled: Bool = true
    var volume: Float = 1.0
    var pan: Float = 0.0
    var effects: [String] = [] // Effect IDs - simplified for now
    var waveformCache: [Float] = []
    
    var endTime: TimeInterval {
        timelineStartTime + duration
    }
}

struct EffectClip: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var effectType: String
    var startTime: TimeInterval = 0
    var duration: TimeInterval = 5
    var parameters: [String: Double] = [:] // Simplified parameter storage
    var isEnabled: Bool = true
}

struct TitleClip: Codable, Identifiable, Equatable {
    let id = UUID()
    var text: String
    var startTime: TimeInterval = 0
    var duration: TimeInterval = 5
    var fontSize: Double = 24
    var fontName: String = "Helvetica"
    var textColor: CodableColor = CodableColor(.white)
    var animation: String = "none" // Animation type as string
    var isEnabled: Bool = true
}

// MARK: - Project Settings
struct ProjectSettings: Codable, Equatable {
    var resolution: Resolution = .uhd4K
    var frameRate: FrameRate = .fps23_976
    var colorSpace: ColorSpace = .rec709
    var audioSampleRate: AudioSampleRate = .rate48kHz
    var audioChannels: AudioChannels = .stereo
    var renderQuality: RenderQuality = .full
    var proxySettings: ProxySettings = ProxySettings()
    var autoSave: Bool = true
    var autoSaveInterval: TimeInterval = 300 // 5 minutes
}

// MARK: - Media Pool
struct MediaPool: Codable, Equatable {
    var mediaItems: [MediaItem] = []
    var bins: [MediaBin] = []
    var smartCollections: [String] = [] // Collection IDs - simplified
    
    init() {
        bins = [MediaBin(name: "Video"), MediaBin(name: "Audio"), MediaBin(name: "Graphics")]
    }
}

struct MediaItem: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var url: URL
    var type: MediaType
    var duration: TimeInterval?
    var frameRate: Double?
    var resolution: CGSize?
    var colorSpace: String?
    var fileSize: Int64
    var createdAt: Date
    var thumbnailData: Data?
    var metadata: [String: String] = [:]
    
    enum MediaType: String, Codable, CaseIterable {
        case video, audio, image, graphic
    }
}

struct MediaBin: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var items: [UUID] = []
    var subBins: [MediaBin] = []
    var color: CodableColor = CodableColor(.gray)
}

// MARK: - Professional Enums
enum Resolution: String, Codable, CaseIterable {
    case sd480 = "720x480"
    case hd720 = "1280x720"
    case hd1080 = "1920x1080"
    case uhd4K = "3840x2160"
    case dci4K = "4096x2160"
    case uhd8K = "7680x4320"
    
    var size: CGSize {
        let components = rawValue.split(separator: "x")
        let width = Double(components[0]) ?? 1920
        let height = Double(components[1]) ?? 1080
        return CGSize(width: width, height: height)
    }
}

enum FrameRate: Double, Codable, CaseIterable {
    case fps23_976 = 23.976
    case fps24 = 24.0
    case fps25 = 25.0
    case fps29_97 = 29.97
    case fps30 = 30.0
    case fps50 = 50.0
    case fps59_94 = 59.94
    case fps60 = 60.0
}

enum AspectRatio: String, Codable, CaseIterable {
    case standard = "4:3"
    case widescreen = "16:9"
    case cinema = "21:9"
    case vertical = "9:16"
    case square = "1:1"
}

enum ColorSpace: String, Codable, CaseIterable {
    case rec709 = "Rec. 709"
    case p3 = "Display P3"
    case rec2020 = "Rec. 2020"
    case hlg = "HLG"
    case pq = "PQ"
}

enum AudioSampleRate: Int, Codable, CaseIterable {
    case rate44_1kHz = 44100
    case rate48kHz = 48000
    case rate96kHz = 96000
    case rate192kHz = 192000
}

enum AudioChannels: String, Codable, CaseIterable {
    case mono = "Mono"
    case stereo = "Stereo" 
    case surround5_1 = "5.1 Surround"
    case surround7_1 = "7.1 Surround"
}

enum RenderQuality: String, Codable, CaseIterable {
    case proxy = "Proxy"
    case half = "Half Resolution"
    case full = "Full Resolution"
    case superSample = "Super Sample"
}

// MARK: - Supporting Types
struct Marker: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var time: TimeInterval
    var color: CodableColor = CodableColor(.red)
    var note: String = ""
    var type: MarkerType = .standard
    
    enum MarkerType: String, Codable, CaseIterable {
        case standard, chapter, todo, favorite
    }
}

struct Chapter: Codable, Identifiable, Equatable {
    let id = UUID()
    var title: String
    var startTime: TimeInterval
    var thumbnailData: Data?
}

struct Workspace: Codable, Identifiable, Equatable {
    let id = UUID()
    var name: String
    var layout: WorkspaceLayout
    var isDefault: Bool = false
    
    static let `default` = Workspace(
        name: "Default",
        layout: WorkspaceLayout.standard,
        isDefault: true
    )
}

enum WorkspaceLayout: String, Codable, CaseIterable {
    case standard = "Standard"
    case editing = "Editing"
    case color = "Color & Effects"
    case audio = "Audio"
    case delivery = "Delivery"
}

struct ProjectMetadata: Codable, Equatable {
    var notes: String = ""
    var tags: [String] = []
    var rating: Int = 0
    var customFields: [String: String] = [:]
}

struct ProxySettings: Codable, Equatable {
    var enableProxy: Bool = false
    var proxyResolution: Resolution = .hd720
    var proxyCodec: String = "H.264"
    var proxyQuality: String = "Medium"
}

// MARK: - Helper Types for Codable Colors
struct CodableColor: Codable, Equatable {
    let red: Double
    let green: Double
    let blue: Double
    let alpha: Double
    
    init(_ color: Color) {
        // This is a simplified implementation
        // In practice, you'd need proper color space conversion
        self.red = 0.5
        self.green = 0.5
        self.blue = 0.5
        self.alpha = 1.0
    }
    
    var color: Color {
        Color(red: red, green: green, blue: blue, opacity: alpha)
    }
}