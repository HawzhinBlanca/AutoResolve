import Foundation
import SwiftUI

// MARK: - Timeline Editing Tools
public class TimelineEditingTools: ObservableObject {
    @Published var activeTool: EditingTool = .selection
    @Published var isEditing: Bool = false
    
    private weak var timeline: TimelineModel?
    
    public enum EditingTool: String, CaseIterable {
        case selection = "Selection"
        case trim = "Trim"
        case slip = "Slip"
        case slide = "Slide"
        case blade = "Blade"
        case ripple = "Ripple"
        case roll = "Roll"
        
        var icon: String {
            switch self {
            case .selection: return "arrow.up.left"
            case .trim: return "scissors"
            case .slip: return "arrow.left.arrow.right"
            case .slide: return "arrow.up.arrow.down"
            case .blade: return "scissors.badge.ellipsis"
            case .ripple: return "arrow.right.to.line"
            case .roll: return "arrow.left.and.right"
            }
        }
        
        var shortcut: KeyEquivalent? {
            switch self {
            case .selection: return "v"
            case .trim: return "t"
            case .slip: return "y"
            case .slide: return "u"
            case .blade: return "b"
            case .ripple: return "r"
            case .roll: return "n"
            }
        }
    }
    
    public init(timeline: TimelineModel? = nil) {
        self.timeline = timeline
    }
    
    public func setTimeline(_ timeline: TimelineModel) {
        self.timeline = timeline
    }
    
    // MARK: - Trim Tool
    public func trimClipStart(_ clipId: UUID, to newStart: TimeInterval) {
        guard let timeline = timeline,
              let clipIndex = timeline.clips.firstIndex(where: { $0.id == clipId }),
              let clip = timeline.clips.first(where: { $0.id == clipId }) else { return }
        
        let duration = clip.duration - (newStart - clip.startTime)
        guard duration > 0 else { return }
        
        timeline.clips[clipIndex].startTime = newStart
        timeline.clips[clipIndex].duration = duration
        timeline.clips[clipIndex].inPoint = clip.inPoint + (newStart - clip.startTime)
    }
    
    public func trimClipEnd(_ clipId: UUID, to newEnd: TimeInterval) {
        guard let timeline = timeline,
              let clipIndex = timeline.clips.firstIndex(where: { $0.id == clipId }),
              let clip = timeline.clips.first(where: { $0.id == clipId }) else { return }
        
        let duration = newEnd - clip.startTime
        guard duration > 0 else { return }
        
        timeline.clips[clipIndex].duration = duration
        timeline.clips[clipIndex].outPoint = clip.inPoint + duration
    }
    
    // MARK: - Slip Tool
    public func slipClip(_ clipId: UUID, by offset: TimeInterval) {
        guard let timeline = timeline,
              let clipIndex = timeline.clips.firstIndex(where: { $0.id == clipId }),
              let clip = timeline.clips.first(where: { $0.id == clipId }) else { return }
        
        // Slip changes the source in/out without changing position
        let newInPoint = clip.inPoint + offset
        let newOutPoint = clip.outPoint + offset
        
        // For now, allow any values (would need source duration to validate properly)
        timeline.clips[clipIndex].inPoint = max(0, newInPoint)
        timeline.clips[clipIndex].outPoint = max(newInPoint, newOutPoint)
    }
    
    // MARK: - Slide Tool
    public func slideClip(_ clipId: UUID, by offset: TimeInterval) {
        guard let timeline = timeline,
              let clipIndex = timeline.clips.firstIndex(where: { $0.id == clipId }) else { return }
        
        // Slide moves the clip without changing its content
        let newStart = timeline.clips[clipIndex].startTime + offset
        if newStart >= 0 {
            timeline.clips[clipIndex].startTime = newStart
        }
        
        // Check for collisions and resolve them
        resolveCollisions()
    }
    
    // MARK: - Blade Tool
    public func bladeClip(_ clipId: UUID, at cutTime: TimeInterval) {
        guard let timeline = timeline,
              let clipIndex = timeline.clips.firstIndex(where: { $0.id == clipId }),
              let clip = timeline.clips.first(where: { $0.id == clipId }) else { return }
        
        // Ensure cut time is within clip bounds
        guard cutTime > clip.startTime && cutTime < clip.startTime + clip.duration else { return }
        
        let relativeTime = cutTime - clip.startTime
        
        // Update first part (original clip, trimmed)
        timeline.clips[clipIndex].duration = relativeTime
        timeline.clips[clipIndex].outPoint = clip.inPoint + relativeTime
        
        // Create second part (new clip)
        var secondClip = SimpleTimelineClip(
            name: clip.name + " (2)",
            trackIndex: clip.trackIndex,
            startTime: cutTime,
            duration: clip.duration - relativeTime,
            sourceURL: clip.sourceURL,
            inPoint: clip.inPoint + relativeTime
        )
        secondClip.outPoint = clip.outPoint
        secondClip.type = clip.type
        secondClip.color = clip.color
        
        timeline.addClip(secondClip)
    }
    
    // MARK: - Ripple Edit
    public func rippleEdit(_ clipId: UUID, newDuration: TimeInterval) {
        guard let timeline = timeline,
              let clipIndex = timeline.clips.firstIndex(where: { $0.id == clipId }),
              let clip = timeline.clips.first(where: { $0.id == clipId }) else { return }
        
        let deltaTime = newDuration - clip.duration
        
        // Update the clip duration
        timeline.clips[clipIndex].duration = newDuration
        timeline.clips[clipIndex].outPoint = clip.inPoint + newDuration
        
        // Ripple all clips after this one on the same track
        for i in 0..<timeline.clips.count {
            if timeline.clips[i].trackIndex == clip.trackIndex && 
               timeline.clips[i].startTime > clip.startTime {
                timeline.clips[i].startTime += deltaTime
            }
        }
    }
    
    // MARK: - Roll Edit
    public func rollEdit(_ clipId1: UUID, _ clipId2: UUID, adjustmentTime: TimeInterval) {
        guard let timeline = timeline,
              let clip1Index = timeline.clips.firstIndex(where: { $0.id == clipId1 }),
              let clip2Index = timeline.clips.firstIndex(where: { $0.id == clipId2 }),
              let clip1 = timeline.clips.first(where: { $0.id == clipId1 }),
              let clip2 = timeline.clips.first(where: { $0.id == clipId2 }) else { return }
        
        // Ensure clips are adjacent
        guard abs((clip1.startTime + clip1.duration) - clip2.startTime) < 0.01 else { return }
        
        // Adjust the out point of the first clip
        timeline.clips[clip1Index].duration += adjustmentTime
        timeline.clips[clip1Index].outPoint += adjustmentTime
        
        // Adjust the in point of the second clip
        timeline.clips[clip2Index].startTime += adjustmentTime
        timeline.clips[clip2Index].duration -= adjustmentTime
        timeline.clips[clip2Index].inPoint += adjustmentTime
    }
    
    // MARK: - Helper Methods
    private func resolveCollisions() {
        guard let timeline = timeline else { return }
        
        // Group clips by track
        var trackClips: [Int: [SimpleTimelineClip]] = [:]
        for clip in timeline.clips {
            trackClips[clip.trackIndex, default: []].append(clip)
        }
        
        // Resolve collisions on each track
        for (_, clips) in trackClips {
            let sortedClips = clips.sorted { $0.startTime < $1.startTime }
            
            for i in 1..<sortedClips.count {
                let prevClip = sortedClips[i-1]
                let currentClip = sortedClips[i]
                
                let prevEnd = prevClip.startTime + prevClip.duration
                if prevEnd > currentClip.startTime {
                    // Collision detected, push current clip
                    let adjustment = prevEnd - currentClip.startTime + 0.1 // Small gap
                    if let clipIndex = timeline.clips.firstIndex(where: { $0.id == currentClip.id }) {
                        timeline.clips[clipIndex].startTime += adjustment
                    }
                }
            }
        }
    }
    
    // MARK: - Tool Actions
    public func performAction(at position: CGPoint, in timelineView: CGRect) {
        switch activeTool {
        case .blade:
            // Convert position to time
            let time = positionToTime(position.x, in: timelineView.width)
            if let clipId = findClipAt(time: time) {
                bladeClip(clipId, at: time)
            }
            
        case .selection:
            // Handle selection
            break
            
        default:
            break
        }
    }
    
    private func positionToTime(_ xPosition: CGFloat, in width: CGFloat) -> TimeInterval {
        guard let timeline = timeline else { return 0 }
        return (xPosition / width) * timeline.duration
    }
    
    private func findClipAt(time: TimeInterval) -> UUID? {
        guard let timeline = timeline else { return nil }
        
        for clip in timeline.clips {
            if time >= clip.startTime && time <= clip.startTime + clip.duration {
                return clip.id
            }
        }
        return nil
    }
}

// MARK: - Timeline Tools View
public struct TimelineToolsView: View {
    @ObservedObject var tools: TimelineEditingTools
    
    public init(tools: TimelineEditingTools) {
        self.tools = tools
    }
    
    public var body: some View {
        HStack(spacing: 2) {
            ForEach(TimelineEditingTools.EditingTool.allCases, id: \.self) { tool in
                ToolButton(
                    tool: tool,
                    isActive: tools.activeTool == tool
                ) {
                    tools.activeTool = tool
                }
            }
        }
        .padding(4)
        .background(Color.black.opacity(0.3))
        .cornerRadius(6)
    }
    
    struct ToolButton: View {
        let tool: TimelineEditingTools.EditingTool
        let isActive: Bool
        let action: () -> Void
        
        var body: some View {
            Button(action: action) {
                Image(systemName: tool.icon)
                    .font(.system(size: 14))
                    .foregroundColor(isActive ? .blue : .gray)
                    .frame(width: 28, height: 28)
                    .background(isActive ? Color.blue.opacity(0.2) : Color.clear)
                    .cornerRadius(4)
            }
            .buttonStyle(PlainButtonStyle())
            .help(tool.rawValue)
            .keyboardShortcut(tool.shortcut ?? KeyEquivalent(""), modifiers: [])
        }
    }
}