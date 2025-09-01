import AppKit
import Foundation
import Combine
import SwiftUI
import CoreMedia

// MARK: - Core Timeline Data Models

// TimelineClip - Main clip type for timeline operations
public typealias TimelineClip = SimpleTimelineClip

// MARK: - Type definitions for SimpleTimelineClip
public enum ClipType: String, Codable {
    case video = "video"
    case audio = "audio"
    case image = "image"
}

public struct VideoEffect: Codable {
    public let name: String
    public let enabled: Bool
    
    public init(name: String, enabled: Bool = true) {
        self.name = name
        self.enabled = enabled
    }
}

public struct SimpleTimelineClip: Identifiable, Equatable {
    public let id: UUID
    public var name: String
    public var sourceURL: URL?
    public var trackIndex: Int
    public var startTime: TimeInterval
    public var duration: TimeInterval
    public var inPoint: TimeInterval = 0
    public var outPoint: TimeInterval
    public var isSelected: Bool = false
    public var colorData: Data? = nil
    public var thumbnailData: Data? = nil
    public var sourceStartTime: TimeInterval = 0
    public var type: ClipType = .video
    public var color: Color = .blue
    public var effects: [VideoEffect] = []
    
    // Computed property for compatibility
    public var url: URL {
        get { sourceURL ?? URL(fileURLWithPath: "/") }
    }
    
    public var thumbnail: NSImage? {
        get {
            guard let data = thumbnailData else { return nil }
            return NSImage(data: data)
        }
        set {
            thumbnailData = newValue?.tiffRepresentation
        }
    }
    
    public init(id: UUID = UUID(), name: String, trackIndex: Int, startTime: TimeInterval, duration: TimeInterval, sourceURL: URL? = nil, inPoint: TimeInterval = 0, isSelected: Bool = false, colorData: Data? = nil, thumbnailData: Data? = nil, sourceStartTime: TimeInterval = 0, type: ClipType = .video, color: Color = .blue, effects: [VideoEffect] = []) {
        self.id = id
        self.name = name
        self.trackIndex = trackIndex
        self.startTime = startTime
        self.duration = duration
        self.outPoint = duration
        self.sourceURL = sourceURL
        self.inPoint = inPoint
        self.isSelected = isSelected
        self.colorData = colorData
        self.thumbnailData = thumbnailData
        self.sourceStartTime = sourceStartTime
        self.type = type
        self.color = color
        self.effects = effects
    }
    
    public var endTime: TimeInterval { startTime + duration }
    public func contains(time: TimeInterval) -> Bool { time >= startTime && time <= endTime }
}

// MARK: - Timeline clip utilities
extension SimpleTimelineClip {
    public func overlaps(start: TimeInterval, end: TimeInterval) -> Bool {
        let aStart = self.startTime
        let aEnd = self.endTime
        return max(aStart, start) < min(aEnd, end)
    }
}

// MARK: - Codable conformance for SimpleTimelineClip
extension SimpleTimelineClip: Codable {
    private enum CodingKeys: String, CodingKey {
        case id, name, sourceURL, trackIndex, startTime, duration
        case inPoint, outPoint, isSelected, colorData, thumbnailData
        case sourceStartTime, type
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(sourceURL, forKey: .sourceURL)
        try container.encode(trackIndex, forKey: .trackIndex)
        try container.encode(startTime, forKey: .startTime)
        try container.encode(duration, forKey: .duration)
        try container.encode(inPoint, forKey: .inPoint)
        try container.encode(outPoint, forKey: .outPoint)
        try container.encode(isSelected, forKey: .isSelected)
        try container.encode(colorData, forKey: .colorData)
        try container.encode(thumbnailData, forKey: .thumbnailData)
        try container.encode(sourceStartTime, forKey: .sourceStartTime)
        try container.encode(type, forKey: .type)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        sourceURL = try container.decodeIfPresent(URL.self, forKey: .sourceURL)
        trackIndex = try container.decode(Int.self, forKey: .trackIndex)
        startTime = try container.decode(TimeInterval.self, forKey: .startTime)
        duration = try container.decode(TimeInterval.self, forKey: .duration)
        inPoint = try container.decodeIfPresent(TimeInterval.self, forKey: .inPoint) ?? 0
        outPoint = try container.decode(TimeInterval.self, forKey: .outPoint)
        isSelected = try container.decodeIfPresent(Bool.self, forKey: .isSelected) ?? false
        colorData = try container.decodeIfPresent(Data.self, forKey: .colorData)
        thumbnailData = try container.decodeIfPresent(Data.self, forKey: .thumbnailData)
        sourceStartTime = try container.decodeIfPresent(TimeInterval.self, forKey: .sourceStartTime) ?? 0
        type = try container.decodeIfPresent(ClipType.self, forKey: .type) ?? .video
        color = .blue  // Default color since it's not persisted
    }
}

public struct UITimelineTrack: Identifiable, Codable {
    public let id = UUID()
    public var name: String
    public var type: TrackType
    public var height: CGFloat = 60
    public var isEnabled: Bool = true
    public var isLocked: Bool = false
    public var clips: [SimpleTimelineClip] = []
    
    public enum TrackType: String, Codable { 
        case video = "V"
        case audio = "A"
        case title = "T"
        case effect = "FX"
        case director = "D"
        case transcription = "TR"
    }
    
    public init(name: String, type: TrackType) {
        self.name = name
        self.type = type
        switch type {
        case .video: self.height = 80
        case .audio: self.height = 50
        case .title, .effect: self.height = 60
        case .director: self.height = 100
        case .transcription: self.height = 40
        }
    }
    
    public mutating func addClip(_ clip: SimpleTimelineClip) {
        clips.append(clip)
        clips.sort { $0.startTime < $1.startTime }
    }
    
    public mutating func removeClip(id: UUID) {
        clips.removeAll { $0.id == id }
    }
}

public struct UITimelineMarker: Identifiable, Codable {
    public let id = UUID()
    public var time: TimeInterval
    public var type: MarkerType
    public var name: String
    public enum MarkerType: String, Codable { case silence, cut, bookmark }
}

/// Main timeline data model
public class TimelineModel: ObservableObject {
    public let id = UUID()
    @Published public var tracks: [UITimelineTrack] = []
    @Published public var duration: TimeInterval = 1800  // 30 minutes default
    @Published public var playheadPosition: TimeInterval = 0
    @Published public var selectedClips: Set<UUID> = []
    @Published public var zoomLevel: Double = 1.0  // Pixels per second
    @Published public var scrollOffset: CGFloat = 0
    @Published public var markers: [UITimelineMarker] = []
    @Published public var selectedTimeRange: (start: TimeInterval, end: TimeInterval)? = nil
    @Published public var workAreaStart: TimeInterval = 0
    @Published public var workAreaEnd: TimeInterval = 1800
    @Published public var playbackRate: Double = 1.0
    
    // Timeline properties
    public var frameRate: Int = 30
    public var resolution: CGSize = CGSize(width: 1920, height: 1080)
    public var name: String = "Untitled Timeline"
    
    // Timeline Editing Tools (simple reference for now)
    
    // MARK: - Computed Properties
    
    /// Get only video tracks
    public var videoTracks: [UITimelineTrack] {
        tracks.filter { $0.type == .video }
    }
    
    /// Get only audio tracks  
    public var audioTracks: [UITimelineTrack] {
        tracks.filter { $0.type == .audio }
    }
    
    // MARK: - Codable
    enum CodingKeys: CodingKey {
        case tracks, duration, frameRate, resolution, name, markers
    }
    
    public init() {
        setupDefaultTracks()
    }
    
    /// Set up default video and audio tracks
    private func setupDefaultTracks() {
        tracks = [
            UITimelineTrack(name: "V1", type: .video),
            UITimelineTrack(name: "V2", type: .video),
            UITimelineTrack(name: "A1", type: .audio),
            UITimelineTrack(name: "A2", type: .audio)
        ]
    }
    
    /// Add a new track
    public func addTrack(type: UITimelineTrack.TrackType) {
        let count = tracks.filter { $0.type == type }.count
        let name = "\(type.rawValue)\(count + 1)"
        tracks.append(UITimelineTrack(name: name, type: type))
    }
    
    /// Remove track by ID
    public func removeTrack(id: UUID) {
        tracks.removeAll { $0.id == id }
    }
    
    /// Add clip to specific track
    public func addClip(_ clip: SimpleTimelineClip, toTrack trackID: UUID) {
        if let index = tracks.firstIndex(where: { $0.id == trackID }) {
            tracks[index].addClip(clip)
            updateDuration()
        }
    }
    
    /// Remove clip from timeline
    public func removeClip(id: UUID) {
        for i in tracks.indices {
            tracks[i].removeClip(id: id)
        }
        selectedClips.remove(id)
        updateDuration()
    }
    
    /// Select clip
    public func selectClip(id: UUID, multi: Bool = false) {
        if multi {
            if selectedClips.contains(id) {
                selectedClips.remove(id)
            } else {
                selectedClips.insert(id)
            }
        } else {
            selectedClips = [id]
        }
    }
    
    /// Clear selection
    public func clearSelection() {
        selectedClips.removeAll()
    }
    
    /// Update timeline duration based on clips
    private func updateDuration() {
        let maxEndTime = tracks.flatMap { $0.clips }.map { $0.endTime }.max() ?? 0
        duration = max(maxEndTime + 60, 1800)  // Add 1 minute padding, min 30 minutes
    }
    
    /// Move playhead to position
    public func setPlayhead(to position: TimeInterval) {
        playheadPosition = max(0, min(position, duration))
    }
    
    /// Convert time to timecode string
    public func timecode(for time: TimeInterval) -> String {
        let hours = Int(time) / 3600
        let minutes = (Int(time) % 3600) / 60
        let seconds = Int(time) % 60
        let frames = Int((time.truncatingRemainder(dividingBy: 1)) * Double(frameRate))
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, seconds, frames)
    }
    
    /// Convert pixel position to time
    public func timeFromX(_ x: CGFloat) -> TimeInterval {
        return Double(x - scrollOffset) / zoomLevel
    }
    
    /// Convert time to pixel position
    public func xFromTime(_ time: TimeInterval) -> CGFloat {
        return CGFloat(time * zoomLevel) + scrollOffset
    }
    
    // MARK: - Additional Properties and Methods
    @Published public var isPlaying: Bool = false
    
    /// Find a clip by ID across all tracks
    public func findClip(id: UUID) -> SimpleTimelineClip? {
        for track in tracks {
            if let clip = track.clips.first(where: { $0.id == id }) {
                return clip
            }
        }
        return nil
    }
    
    /// Deselect a specific clip
    public func deselectClip(id: UUID) {
        selectedClips.remove(id)
    }
    
    /// Find the nearest snap point to the given time
    public func findSnapPoint(near time: TimeInterval, threshold: TimeInterval = 0.5) -> TimeInterval? {
        var snapPoints: [TimeInterval] = []
        
        // Add clip start and end points
        for track in tracks {
            for clip in track.clips {
                snapPoints.append(clip.startTime)
                snapPoints.append(clip.endTime)
            }
        }
        
        // Add marker positions
        for marker in markers {
            snapPoints.append(marker.time)
        }
        
        // Add playhead position
        snapPoints.append(playheadPosition)
        
        // Find nearest point within threshold
        let nearestPoint = snapPoints.min(by: { abs($0 - time) < abs($1 - time) })
        
        if let point = nearestPoint, abs(point - time) <= threshold {
            return point
        }
        
        return nil
    }
    
    /// Move a clip to a new time position
    public func moveClip(id: UUID, to newTime: TimeInterval) {
        for trackIndex in tracks.indices {
            if let clipIndex = tracks[trackIndex].clips.firstIndex(where: { $0.id == id }) {
                tracks[trackIndex].clips[clipIndex].startTime = max(0, newTime)
                tracks[trackIndex].clips.sort { $0.startTime < $1.startTime }
                updateDuration()
                break
            }
        }
    }
    
    /// Resize a clip
    public func resizeClip(id: UUID, newStartTime: TimeInterval? = nil, newDuration: TimeInterval? = nil) {
        for trackIndex in tracks.indices {
            if let clipIndex = tracks[trackIndex].clips.firstIndex(where: { $0.id == id }) {
                if let newStart = newStartTime {
                    tracks[trackIndex].clips[clipIndex].startTime = max(0, newStart)
                }
                if let newDuration = newDuration, newDuration > 0 {
                    tracks[trackIndex].clips[clipIndex].duration = max(0.1, newDuration)  // Minimum 0.1 second
                }
                updateDuration()
                break
            }
        }
    }
    
    /// Update a clip with new values
    public func updateClip(id: UUID, startTime: TimeInterval? = nil, duration: TimeInterval? = nil) {
        resizeClip(id: id, newStartTime: startTime, newDuration: duration)
    }
    
    /// Cut a clip at a specific time
    public func cutClip(id: UUID, at cutTime: TimeInterval) {
        for trackIndex in tracks.indices {
            if let clipIndex = tracks[trackIndex].clips.firstIndex(where: { $0.id == id }) {
                let clip = tracks[trackIndex].clips[clipIndex]
                
                // Only cut if the time is within the clip
                if cutTime > clip.startTime && cutTime < clip.endTime {
                    // Resize the original clip to end at cut point
                    tracks[trackIndex].clips[clipIndex].duration = cutTime - clip.startTime
                    
                    // Create new clip from cut point
                    var newClip = SimpleTimelineClip(id: UUID(), 
                        name: clip.name,
                        trackIndex: clip.trackIndex,
                        startTime: cutTime,
                        duration: clip.endTime - cutTime
                    )
                    newClip.sourceURL = clip.sourceURL
                    newClip.sourceStartTime = clip.sourceStartTime + (cutTime - clip.startTime)
                    newClip.type = clip.type
                    newClip.color = clip.color
                    
                    tracks[trackIndex].addClip(newClip)
                }
                break
            }
        }
    }
}

// MARK: - Edit Operations

extension TimelineModel {
    
    /// Cut clip at playhead position
    func cutAtPlayhead() {
        let cutTime = playheadPosition
        
        for trackIndex in tracks.indices {
            if let clipIndex = tracks[trackIndex].clips.firstIndex(where: { $0.contains(time: cutTime) }) {
                var clip = tracks[trackIndex].clips[clipIndex]
                
                // Create two clips from the cut
                let originalEnd = clip.endTime
                
                // First part: original start to cut point
                clip.duration = cutTime - clip.startTime
                tracks[trackIndex].clips[clipIndex] = clip
                
                // Second part: cut point to original end
                var secondClip = SimpleTimelineClip(id: UUID(), 
                    name: clip.name,
                    trackIndex: clip.trackIndex,
                    startTime: cutTime,
                    duration: originalEnd - cutTime
                )
                secondClip.sourceURL = clip.sourceURL
                secondClip.inPoint = clip.inPoint + (cutTime - clip.startTime)
                secondClip.outPoint = clip.outPoint
                secondClip.isSelected = clip.isSelected
                secondClip.colorData = clip.colorData
                secondClip.thumbnailData = clip.thumbnailData
                
                tracks[trackIndex].addClip(secondClip)
            }
        }
    }
    
    /// Delete selected clips
    func deleteSelected() {
        for id in selectedClips {
            removeClip(id: id)
        }
        clearSelection()
    }
    
    /// Duplicate selected clips
    func duplicateSelected() {
        var newClips: [(SimpleTimelineClip, UUID)] = []
        
        for trackIndex in tracks.indices {
            for clip in tracks[trackIndex].clips where selectedClips.contains(clip.id) {
                var newClip = SimpleTimelineClip(id: UUID(), 
                    name: clip.name,
                    trackIndex: clip.trackIndex,
                    startTime: clip.startTime + (clip.duration ?? 0),  // Place after original
                    duration: clip.duration ?? 0
                )
                newClip.sourceURL = clip.sourceURL
                newClip.inPoint = clip.inPoint
                newClip.outPoint = clip.outPoint
                newClip.isSelected = false  // Don't select duplicated clips
                newClip.colorData = clip.colorData
                newClip.thumbnailData = clip.thumbnailData
                newClips.append((newClip, tracks[trackIndex].id))
            }
        }
        
        // Add duplicated clips
        for (clip, trackID) in newClips {
            addClip(clip, toTrack: trackID)
        }
    }
    
    /// Move selected clips by delta
    func moveSelectedClips(by delta: TimeInterval) {
        for trackIndex in tracks.indices {
            var updatedClips = tracks[trackIndex].clips
            for idx in updatedClips.indices {
                if selectedClips.contains(updatedClips[idx].id) {
                    updatedClips[idx].startTime = max(0, updatedClips[idx].startTime + delta)
                }
            }
            updatedClips.sort { $0.startTime < $1.startTime }
            tracks[trackIndex].clips = updatedClips
        }
        updateDuration()
    }
}

// MARK: - Snap and Alignment

extension TimelineModel {
    
    /// Find snap points near a time position
    func snapPoints(near time: TimeInterval, threshold: TimeInterval = 0.5) -> [TimeInterval] {
        var points: [TimeInterval] = [0, playheadPosition]
        
        // Add clip edges as snap points
        for track in tracks {
            for clip in track.clips {
                points.append(clip.startTime)
                points.append(clip.endTime)
            }
        }
        
        // Filter points within threshold
        return points.filter { abs($0 - time) <= threshold }.sorted()
    }
    
    /// Snap time to nearest snap point
    func snap(time: TimeInterval, threshold: TimeInterval = 0.5) -> TimeInterval {
        let points = snapPoints(near: time, threshold: threshold)
        
        if let nearest = points.min(by: { abs($0 - time) < abs($1 - time) }) {
            return nearest
        }
        
        return time
    }
}

// MARK: - Import/Export

extension TimelineModel {
    
    /// Export to EDL format
    func exportEDL() -> String {
        var edl = "TITLE: \(name)\n\n"
        edl += "FCM: NON-DROP FRAME\n\n"
        
        var eventNum = 1
        let videoTracks = tracks.filter { $0.type == .video }
        
        for track in videoTracks {
            for clip in track.clips {
                let srcIn = timecode(for: clip.inPoint)
                let srcOut = timecode(for: clip.outPoint)
                let recIn = timecode(for: clip.startTime)
                let recOut = timecode(for: clip.endTime)
                
                edl += String(format: "%03d  %-8s V     C        %@ %@ %@ %@\n",
                             eventNum, String(clip.name.prefix(8)), srcIn, srcOut, recIn, recOut)
                edl += "* FROM CLIP NAME: \(clip.name)\n\n"
                
                eventNum += 1
            }
        }
        
        return edl
    }
    
    /// Import from cuts data
    func importFromCuts(_ cuts: [[String: Any]]) {
        clearSelection()
        tracks[0].clips.removeAll()
        
        for (index, cut) in cuts.enumerated() {
            if let t0 = cut["t0"] as? TimeInterval,
               let t1 = cut["t1"] as? TimeInterval {
                let clip = SimpleTimelineClip(id: UUID(), 
                    name: "Clip \(index + 1)",
                    trackIndex: 0,
                    startTime: t0,
                    duration: t1 - t0
                )
                tracks[0].addClip(clip)
            }
        }
        
        updateDuration()
    }
}

// MARK: - Missing Properties Fix for Compilation
public extension TimelineModel {
    var totalDuration: TimeInterval {
        duration
    }
    
    var currentTime: TimeInterval {
        get { playheadPosition }
        set { playheadPosition = newValue }
    }
    
    var selectedClipID: UUID? {
        selectedClips.first
    }
    
    var cmDuration: CMTime {
        CMTime(seconds: duration, preferredTimescale: 600)
    }
}
