import Foundation
import AVFoundation

// Extensions to fix timeline mutation issues
extension TimelineModel {
    /// Add clip to specific track index
    public func addClipToTrack(_ clip: SimpleTimelineClip, trackIndex: Int) {
        guard trackIndex < tracks.count else { return }
        tracks[trackIndex].clips.append(clip)
    }
    
    /// Add clip to first video track
    public func addClipToFirstVideoTrack(_ clip: SimpleTimelineClip) {
        if let firstVideoTrackIndex = tracks.firstIndex(where: { $0.type == .video }) {
            tracks[firstVideoTrackIndex].clips.append(clip)
        }
    }
}
