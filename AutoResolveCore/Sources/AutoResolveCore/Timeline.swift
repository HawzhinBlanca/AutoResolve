import Foundation

// MARK: - Timeline Core Types

public struct Clip: Identifiable, Codable, Hashable {
    public let id: UUID
    public var name: String
    public var sourceURL: URL
    public var trackIndex: Int
    public var startTick: Tick
    public var duration: Tick
    public var inPoint: Tick
    public var outPoint: Tick
    
    public init(
        id: UUID = UUID(),
        name: String,
        sourceURL: URL,
        trackIndex: Int,
        startTick: Tick,
        duration: Tick,
        inPoint: Tick = .zero,
        outPoint: Tick? = nil
    ) {
        self.id = id
        self.name = name
        self.sourceURL = sourceURL
        self.trackIndex = trackIndex
        self.startTick = startTick
        self.duration = duration
        self.inPoint = inPoint
        self.outPoint = outPoint ?? duration
    }
    
    public var endTick: Tick {
        startTick + duration
    }
    
    public func contains(_ tick: Tick) -> Bool {
        tick >= startTick && tick < endTick
    }
}

public struct Track: Identifiable, Codable {
    public let id: UUID
    public var name: String
    public var index: Int
    public var clips: [Clip]
    public var isLocked: Bool
    public var isVisible: Bool
    
    public init(
        id: UUID = UUID(),
        name: String,
        index: Int,
        clips: [Clip] = [],
        isLocked: Bool = false,
        isVisible: Bool = true
    ) {
        self.id = id
        self.name = name
        self.index = index
        self.clips = clips
        self.isLocked = isLocked
        self.isVisible = isVisible
    }
}

public struct Timeline: Codable {
    public var tracks: [Track]
    public var duration: Tick
    public var playhead: Tick
    public var selection: Set<UUID>
    
    public init(
        tracks: [Track] = [],
        duration: Tick = .zero,
        playhead: Tick = .zero,
        selection: Set<UUID> = []
    ) {
        self.tracks = tracks
        self.duration = duration
        self.playhead = playhead
        self.selection = selection
    }
    
    // MARK: - Timeline Operations
    
    public mutating func blade(at tick: Tick, trackIndex: Int) {
        guard trackIndex < tracks.count else { return }
        
        var track = tracks[trackIndex]
        var newClips: [Clip] = []
        
        for clip in track.clips {
            if clip.contains(tick) {
                // Split clip
                let firstDuration = tick - clip.startTick
                let secondDuration = clip.endTick - tick
                
                var firstClip = clip
                firstClip.duration = firstDuration
                
                let secondClip = Clip(
                    id: UUID(),
                    name: clip.name,
                    sourceURL: clip.sourceURL,
                    trackIndex: clip.trackIndex,
                    startTick: tick,
                    duration: secondDuration,
                    inPoint: clip.inPoint + firstDuration,
                    outPoint: clip.outPoint
                )
                
                newClips.append(firstClip)
                newClips.append(secondClip)
            } else {
                newClips.append(clip)
            }
        }
        
        track.clips = newClips
        tracks[trackIndex] = track
    }
    
    public mutating func deleteClip(_ clipId: UUID, ripple: Bool = false) {
        for i in 0..<tracks.count {
            if let clipIndex = tracks[i].clips.firstIndex(where: { $0.id == clipId }) {
                let deletedClip = tracks[i].clips[clipIndex]
                tracks[i].clips.remove(at: clipIndex)
                
                if ripple {
                    // Move all subsequent clips back
                    for j in clipIndex..<tracks[i].clips.count {
                        tracks[i].clips[j].startTick = tracks[i].clips[j].startTick - deletedClip.duration
                    }
                }
                break
            }
        }
    }
    
    public mutating func trim(clipId: UUID, edge: Command.Edge, to newTick: Tick) {
        for i in 0..<tracks.count {
            if let clipIndex = tracks[i].clips.firstIndex(where: { $0.id == clipId }) {
                var clip = tracks[i].clips[clipIndex]
                
                switch edge {
                case .leading:
                    let delta = newTick - clip.startTick
                    clip.startTick = newTick
                    clip.duration = clip.duration - delta
                    clip.inPoint = clip.inPoint + delta
                    
                case .trailing:
                    clip.duration = newTick - clip.startTick
                    clip.outPoint = clip.inPoint + clip.duration
                }
                
                tracks[i].clips[clipIndex] = clip
                break
            }
        }
    }
    
    public func hash() -> String {
        // Stable hash for diff.json
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        
        if let data = try? encoder.encode(self) {
            return data.base64EncodedString()
        }
        return ""
    }
}