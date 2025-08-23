import SwiftUI
import AppKit

// MARK: - Timeline Interaction Manager

class TimelineInteractionManager: ObservableObject {
    @Published var editMode: EditMode = .normal
    @Published var magnetEnabled: Bool = true
    @Published var snapThreshold: TimeInterval = 0.5
    @Published var rippleEditEnabled: Bool = false
    
    enum EditMode {
        case normal
        case ripple
        case roll
        case slip
        case slide
        
        var description: String {
            switch self {
            case .normal: return "Normal Edit"
            case .ripple: return "Ripple Edit"
            case .roll: return "Roll Edit"
            case .slip: return "Slip Edit"
            case .slide: return "Slide Edit"
            }
        }
        
        var cursor: NSCursor {
            switch self {
            case .normal: return .arrow
            case .ripple, .roll: return .resizeLeftRight
            case .slip, .slide: return .openHand
            }
        }
    }
    
    // Perform ripple edit
    func performRippleEdit(clip: TimelineClip, delta: TimeInterval, in timeline: TimelineModel) {
        guard let trackIndex = timeline.tracks.firstIndex(where: { $0.clips.contains(where: { $0.id == clip.id }) }) else { return }
        
        var track = timeline.tracks[trackIndex]
        guard let clipIndex = track.clips.firstIndex(where: { $0.id == clip.id }) else { return }
        
        // Move clip
        track.clips[clipIndex].startTime += delta
        
        // Ripple all following clips
        for i in (clipIndex + 1)..<track.clips.count {
            track.clips[i].startTime += delta
        }
        
        timeline.tracks[trackIndex] = track
    }
    
    // Perform roll edit
    func performRollEdit(clip: TimelineClip, delta: TimeInterval, in timeline: TimelineModel, edge: Edge) {
        guard let trackIndex = timeline.tracks.firstIndex(where: { $0.clips.contains(where: { $0.id == clip.id }) }) else { return }
        
        var track = timeline.tracks[trackIndex]
        guard let clipIndex = track.clips.firstIndex(where: { $0.id == clip.id }) else { return }
        
        if edge == .leading {
            // Adjust current clip start and previous clip end
            if clipIndex > 0 {
                let maxDelta = min(delta, track.clips[clipIndex].duration - 0.1)
                let minDelta = max(delta, -track.clips[clipIndex - 1].duration + 0.1)
                let actualDelta = max(minDelta, min(maxDelta, delta))
                
                track.clips[clipIndex].startTime += actualDelta
                track.clips[clipIndex].duration -= actualDelta
                track.clips[clipIndex - 1].duration += actualDelta
            }
        } else {
            // Adjust current clip end and next clip start
            if clipIndex < track.clips.count - 1 {
                let maxDelta = min(delta, track.clips[clipIndex + 1].duration - 0.1)
                let minDelta = max(delta, -track.clips[clipIndex].duration + 0.1)
                let actualDelta = max(minDelta, min(maxDelta, delta))
                
                track.clips[clipIndex].duration += actualDelta
                track.clips[clipIndex + 1].startTime += actualDelta
                track.clips[clipIndex + 1].duration -= actualDelta
            }
        }
        
        timeline.tracks[trackIndex] = track
    }
    
    // Perform slip edit (change in/out points without moving clip)
    func performSlipEdit(clip: TimelineClip, delta: TimeInterval, in timeline: TimelineModel) {
        guard let trackIndex = timeline.tracks.firstIndex(where: { $0.clips.contains(where: { $0.id == clip.id }) }) else { return }
        guard let clipIndex = timeline.tracks[trackIndex].clips.firstIndex(where: { $0.id == clip.id }) else { return }
        
        var modifiedClip = timeline.tracks[trackIndex].clips[clipIndex]
        
        // Adjust in and out points
        let newInPoint = max(0, modifiedClip.inPoint + delta)
        let newOutPoint = modifiedClip.outPoint + delta
        
        // Ensure we don't exceed source duration
        if newOutPoint - newInPoint == modifiedClip.duration {
            modifiedClip.inPoint = newInPoint
            modifiedClip.outPoint = newOutPoint
            timeline.tracks[trackIndex].clips[clipIndex] = modifiedClip
        }
    }
    
    // Perform slide edit (move clip and adjust neighbors)
    func performSlideEdit(clip: TimelineClip, delta: TimeInterval, in timeline: TimelineModel) {
        guard let trackIndex = timeline.tracks.firstIndex(where: { $0.clips.contains(where: { $0.id == clip.id }) }) else { return }
        
        var track = timeline.tracks[trackIndex]
        guard let clipIndex = track.clips.firstIndex(where: { $0.id == clip.id }) else { return }
        
        // Can only slide if there are adjacent clips
        guard clipIndex > 0 && clipIndex < track.clips.count - 1 else { return }
        
        let maxDelta = track.clips[clipIndex + 1].duration - 0.1
        let minDelta = -(track.clips[clipIndex - 1].duration - 0.1)
        let actualDelta = max(minDelta, min(maxDelta, delta))
        
        // Adjust previous clip
        track.clips[clipIndex - 1].duration += actualDelta
        
        // Move current clip
        track.clips[clipIndex].startTime += actualDelta
        
        // Adjust next clip
        track.clips[clipIndex + 1].startTime += actualDelta
        track.clips[clipIndex + 1].duration -= actualDelta
        
        timeline.tracks[trackIndex] = track
    }
}

// MARK: - Multi-Selection Rectangle

struct TimelineSelectionRectangle: View {
    @Binding var startPoint: CGPoint
    @Binding var endPoint: CGPoint
    @Binding var isSelecting: Bool
    
    var body: some View {
        if isSelecting {
            Path { path in
                let rect = CGRect(
                    x: min(startPoint.x, endPoint.x),
                    y: min(startPoint.y, endPoint.y),
                    width: abs(endPoint.x - startPoint.x),
                    height: abs(endPoint.y - startPoint.y)
                )
                path.addRect(rect)
            }
            .stroke(Color.accentColor, lineWidth: 1)
            .background(
                Rectangle()
                    .fill(Color.accentColor.opacity(0.1))
            )
        }
    }
}

// MARK: - Snap Indicator

struct SnapIndicator: View {
    let position: CGFloat
    let height: CGFloat
    @Binding var isVisible: Bool
    
    var body: some View {
        if isVisible {
            Rectangle()
                .fill(Color.yellow)
                .frame(width: 2, height: height)
                .offset(x: position)
                .transition(.opacity)
                .animation(.easeInOut(duration: 0.1), value: isVisible)
        }
    }
}

// MARK: - Timeline Grid

struct TimelineGrid: View {
    @ObservedObject var timeline: TimelineModel
    let width: CGFloat
    let height: CGFloat
    
    var body: some View {
        Canvas { context, size in
            let secondsPerPixel = 1.0 / timeline.zoomLevel
            let gridInterval = getGridInterval(secondsPerPixel: secondsPerPixel)
            
            // Vertical grid lines
            var time: TimeInterval = 0
            while time <= timeline.duration {
                let x = CGFloat(time * timeline.zoomLevel)
                
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: x, y: 0))
                        path.addLine(to: CGPoint(x: x, y: size.height))
                    },
                    with: .color(.secondary.opacity(0.1)),
                    lineWidth: 0.5
                )
                
                time += gridInterval
            }
            
            // Horizontal grid lines (track separators)
            var y: CGFloat = 0
            for track in timeline.tracks {
                y += track.height
                
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: size.width, y: y))
                    },
                    with: .color(.secondary.opacity(0.2)),
                    lineWidth: 0.5
                )
            }
        }
        .frame(width: width, height: height)
        .allowsHitTesting(false)
    }
    
    private func getGridInterval(secondsPerPixel: Double) -> TimeInterval {
        if secondsPerPixel > 2 {
            return 30  // 30 second grid
        } else if secondsPerPixel > 0.5 {
            return 10  // 10 second grid
        } else {
            return 1   // 1 second grid
        }
    }
}

// MARK: - Edit Mode Indicator

struct EditModeIndicator: View {
    @ObservedObject var interactionManager: TimelineInteractionManager
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: iconForMode(interactionManager.editMode))
                .font(.system(size: 14))
            
            Text(interactionManager.editMode.description)
                .font(.system(size: 12, weight: .medium))
            
            if interactionManager.magnetEnabled {
                Image(systemName: "arrow.up.and.down.and.arrow.left.and.right")
                    .font(.system(size: 12))
                    .foregroundColor(.yellow)
                    .help("Magnetic Timeline Enabled")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(6)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
        )
    }
    
    private func iconForMode(_ mode: TimelineInteractionManager.EditMode) -> String {
        switch mode {
        case .normal: return "arrow.up.left"
        case .ripple: return "arrow.left.and.right"
        case .roll: return "arrow.up.and.down.and.arrow.left.and.right"
        case .slip: return "arrow.2.squarepath"
        case .slide: return "arrow.left.arrow.right"
        }
    }
}

// MARK: - Timeline Keyboard Shortcuts

struct TimelineKeyboardShortcuts: ViewModifier {
    @ObservedObject var timeline: TimelineModel
    @ObservedObject var interactionManager: TimelineInteractionManager
    
    func body(content: Content) -> some View {
        content
            // Navigation
            .keyboardShortcut(.leftArrow, modifiers: []) {
                timeline.setPlayhead(to: timeline.playheadPosition - 1/30.0)
            }
            .keyboardShortcut(.rightArrow, modifiers: []) {
                timeline.setPlayhead(to: timeline.playheadPosition + 1/30.0)
            }
            .keyboardShortcut(.home, modifiers: []) {
                timeline.setPlayhead(to: 0)
            }
            .keyboardShortcut(.end, modifiers: []) {
                timeline.setPlayhead(to: timeline.duration)
            }
            
            // Edit modes
            .keyboardShortcut("q", modifiers: []) {
                interactionManager.editMode = .normal
            }
            .keyboardShortcut("w", modifiers: []) {
                interactionManager.editMode = .ripple
            }
            .keyboardShortcut("e", modifiers: []) {
                interactionManager.editMode = .roll
            }
            .keyboardShortcut("r", modifiers: []) {
                interactionManager.editMode = .slip
            }
            .keyboardShortcut("t", modifiers: []) {
                interactionManager.editMode = .slide
            }
            
            // Timeline actions
            .keyboardShortcut("n", modifiers: []) {
                interactionManager.magnetEnabled.toggle()
            }
            .keyboardShortcut("+", modifiers: []) {
                timeline.zoomLevel = min(timeline.zoomLevel * 1.2, 200)
            }
            .keyboardShortcut("-", modifiers: []) {
                timeline.zoomLevel = max(timeline.zoomLevel / 1.2, 10)
            }
    }
}

// Extension to add keyboard shortcuts
extension View {
    func keyboardShortcut(_ key: KeyEquivalent, modifiers: EventModifiers = [], action: @escaping () -> Void) -> some View {
        self.background(
            Button("") { action() }
                .keyboardShortcut(key, modifiers: modifiers)
                .hidden()
        )
    }
}

// MARK: - Collision Detection

extension TimelineModel {
    
    /// Check if placing a clip would cause collision
    func wouldCollide(clip: TimelineClip, at time: TimeInterval, on trackIndex: Int) -> Bool {
        guard trackIndex < tracks.count else { return false }
        
        let track = tracks[trackIndex]
        let endTime = time + clip.duration
        
        for existingClip in track.clips {
            if existingClip.id != clip.id {
                if existingClip.overlaps(start: time, end: endTime) {
                    return true
                }
            }
        }
        
        return false
    }
    
    /// Find nearest valid position for clip
    func nearestValidPosition(for clip: TimelineClip, near time: TimeInterval, on trackIndex: Int) -> TimeInterval {
        guard !wouldCollide(clip: clip, at: time, on: trackIndex) else {
            // Find gaps where clip could fit
            let track = tracks[trackIndex]
            let sortedClips = track.clips.sorted { $0.startTime < $1.startTime }
            
            // Check before first clip
            if sortedClips.isEmpty || sortedClips[0].startTime >= clip.duration {
                return 0
            }
            
            // Check between clips
            for i in 0..<sortedClips.count - 1 {
                let gapStart = sortedClips[i].endTime
                let gapEnd = sortedClips[i + 1].startTime
                let gapSize = gapEnd - gapStart
                
                if gapSize >= clip.duration {
                    // Found a gap
                    if abs(gapStart - time) < abs(gapEnd - clip.duration - time) {
                        return gapStart
                    } else {
                        return gapEnd - clip.duration
                    }
                }
            }
            
            // Place after last clip
            if let lastClip = sortedClips.last {
                return lastClip.endTime
            }
            
            // No valid position found, return the original time
            return time
        }
        
        return time
    }
}