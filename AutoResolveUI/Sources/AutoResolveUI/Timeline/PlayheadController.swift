// AUTORESOLVE V3.0 - PLAYHEAD CONTROLLER
// Professional playhead with dragging, scrubbing, and keyboard navigation

import SwiftUI
import AppKit
import AVFoundation

// MARK: - Interactive Playhead View
struct InteractivePlayheadView: View {
    @ObservedObject var timeline: TimelineModel
    @State private var isDragging = false
    @State private var dragOffset: CGFloat = 0
    @State private var isScrubbing = false
    @State private var scrubStartPosition: TimeInterval = 0
    @State private var showTimecodePopover = false
    
    private var playheadX: CGFloat {
        timeline.xFromTime(timeline.playheadPosition) + (isDragging ? dragOffset : 0)
    }
    
    public var body: some View {
        ZStack(alignment: .top) {
            // Playhead line
            Rectangle()
                .fill(Color.red)
                .frame(width: 2)
                .shadow(color: .black.opacity(0.3), radius: 2, x: 0, y: 0)
            
            // Playhead handle
            PlayheadHandle(isDragging: $isDragging, isScrubbing: $isScrubbing)
                .offset(y: -12)
                .gesture(playheadDragGesture)
                .onHover { hovering in
                    if hovering {
                        NSCursor.openHand.push()
                    } else {
                        NSCursor.pop()
                    }
                }
            
            // Timecode display (while dragging)
            if isDragging || isScrubbing {
                TimecodeDisplay(
                    currentTime: timeline.playheadPosition + timeline.timeFromX(dragOffset),
                    duration: timeline.totalDuration
                )
                .offset(y: -40)
            }
        }
        .frame(width: 2)
        .position(x: playheadX, y: timeline.totalTrackHeight / 2)
        .animation(isDragging ? .none : .spring(response: 0.2, dampingFraction: 0.8), value: playheadX)
        .onAppear {
            setupKeyboardShortcuts()
        }
    }
    
    // MARK: - Drag Gesture
    private var playheadDragGesture: some Gesture {
        DragGesture(minimumDistance: 1)
            .onChanged { value in
                handleDragChanged(value)
            }
            .onEnded { value in
                handleDragEnded(value)
            }
    }
    
    private func handleDragChanged(_ value: DragGesture.Value) {
        if !isDragging {
            isDragging = true
            scrubStartPosition = timeline.playheadPosition
            NSCursor.closedHand.push()
            
            // Start audio scrubbing if enabled
            if timeline.audioScrubbingEnabled {
                isScrubbing = true
                startAudioScrubbing()
            }
        }
        
        dragOffset = value.translation.width
        
        // Update time and check for snapping
        let proposedTime = scrubStartPosition + timeline.timeFromX(dragOffset)
        
        if let snapTime = findSnapPoint(for: proposedTime) {
            let snapX = timeline.xFromTime(snapTime - scrubStartPosition)
            dragOffset = snapX
            
            // Haptic feedback
            NSHapticFeedbackManager.defaultPerformer.perform(
                .alignment,
                performanceTime: .now
            )
        }
        
        // Update scrubbing audio
        if isScrubbing {
            updateAudioScrubbing(at: proposedTime)
        }
    }
    
    private func handleDragEnded(_ value: DragGesture.Value) {
        let finalTime = scrubStartPosition + timeline.timeFromX(dragOffset)
        
        // Apply snapping if close enough
        if let snapTime = findSnapPoint(for: finalTime) {
            timeline.setPlayhead(to: snapTime)
        } else {
            timeline.setPlayhead(to: max(0, min(timeline.duration, finalTime)))
        }
        
        isDragging = false
        dragOffset = 0
        NSCursor.pop()
        
        if isScrubbing {
            stopAudioScrubbing()
            isScrubbing = false
        }
    }
    
    // MARK: - Snapping
    private func findSnapPoint(for time: TimeInterval) -> TimeInterval? {
        let snapThreshold: TimeInterval = 0.1 // 100ms snap threshold
        
        // Check clip edges
        for track in timeline.tracks {
            for clip in track.clips {
                // Snap to clip start
                if abs(clip.startTime - time) < snapThreshold {
                    return clip.startTime
                }
                // Snap to clip end
                let clipEnd = clip.startTime + clip.duration ?? 0
                if abs(clipEnd - time) < snapThreshold {
                    return clipEnd
                }
            }
        }
        
        // Check markers
        for marker in timeline.markers {
            if abs(marker.time - time) < snapThreshold {
                return marker.time
            }
        }
        
        // Check grid if enabled
        if timeline.snapToGrid {
            let gridInterval = timeline.gridInterval
            let nearestGrid = round(time / gridInterval) * gridInterval
            if abs(nearestGrid - time) < snapThreshold {
                return nearestGrid
            }
        }
        
        return nil
    }
    
    // MARK: - Audio Scrubbing
    private func startAudioScrubbing() {
        // Initialize audio scrubbing
        timeline.audioEngine?.startScrubbing()
    }
    
    private func updateAudioScrubbing(at time: TimeInterval) {
        // Update audio playback for scrubbing
        timeline.audioEngine?.scrubTo(time: time)
    }
    
    private func stopAudioScrubbing() {
        // Stop audio scrubbing
        timeline.audioEngine?.stopScrubbing()
    }
    
    // MARK: - Keyboard Shortcuts
    private func setupKeyboardShortcuts() {
        // These would be better implemented at the app level
        // but shown here for completeness
    }
}

// MARK: - Playhead Handle
struct PlayheadHandle: View {
    @Binding var isDragging: Bool
    @Binding var isScrubbing: Bool
    @State private var isHovered = false
    
    public var body: some View {
        ZStack {
            // Main handle shape
            PlayheadShape()
                .fill(Color.red)
                .frame(width: 14, height: 20)
                .scaleEffect(isDragging ? 1.1 : (isHovered ? 1.05 : 1.0))
                .shadow(color: .black.opacity(0.3), radius: 2, x: 0, y: 1)
            
            // Inner highlight
            PlayheadShape()
                .fill(
                    LinearGradient(
                        colors: [
                            Color.red.opacity(0.8),
                            Color.red
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 10, height: 16)
            
            // Scrubbing indicator
            if isScrubbing {
                Circle()
                    .fill(Color.white.opacity(0.8))
                    .frame(width: 4, height: 4)
                    .offset(y: -2)
            }
        }
        .animation(.spring(response: 0.2, dampingFraction: 0.8), value: isDragging)
        .animation(.easeInOut(duration: 0.1), value: isHovered)
        .onHover { hovering in
            isHovered = hovering
        }
    }
}

// MARK: - Playhead Shape
struct PlayheadShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        
        // Draw inverted triangle/diamond shape
        path.move(to: CGPoint(x: rect.midX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY + 4))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + 4))
        path.closeSubpath()
        
        return path
    }
}


// MARK: - Playhead Navigation Controller
struct PlayheadNavigationController: View {
    @ObservedObject var timeline: TimelineModel
    @State private var jumpToTimeText = ""
    @State private var showJumpToDialog = false
    
    public var body: some View {
        HStack(spacing: 8) {
            // Frame-by-frame navigation
            Button(action: { navigateFrames(-1) }) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 12))
            }
            .buttonStyle(PlainButtonStyle())
            .help("Previous Frame (←)")
            .keyboardShortcut(.leftArrow, modifiers: [])
            
            Button(action: { navigateFrames(1) }) {
                Image(systemName: "chevron.right")
                    .font(.system(size: 12))
            }
            .buttonStyle(PlainButtonStyle())
            .help("Next Frame (→)")
            .keyboardShortcut(.rightArrow, modifiers: [])
            
            Divider()
                .frame(height: 16)
            
            // Jump to markers
            Button(action: { jumpToPreviousMarker() }) {
                Image(systemName: "arrow.left.to.line")
                    .font(.system(size: 12))
            }
            .buttonStyle(PlainButtonStyle())
            .help("Previous Marker (⇧M)")
            .keyboardShortcut("m", modifiers: .shift)
            
            Button(action: { jumpToNextMarker() }) {
                Image(systemName: "arrow.right.to.line")
                    .font(.system(size: 12))
            }
            .buttonStyle(PlainButtonStyle())
            .help("Next Marker (M)")
            .keyboardShortcut("m", modifiers: [])
            
            Divider()
                .frame(height: 16)
            
            // Jump to edit points
            Button(action: { jumpToPreviousEdit() }) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 12))
            }
            .buttonStyle(PlainButtonStyle())
            .help("Previous Edit (↑)")
            .keyboardShortcut(.upArrow, modifiers: [])
            
            Button(action: { jumpToNextEdit() }) {
                Image(systemName: "arrow.down")
                    .font(.system(size: 12))
            }
            .buttonStyle(PlainButtonStyle())
            .help("Next Edit (↓)")
            .keyboardShortcut(.downArrow, modifiers: [])
            
            Divider()
                .frame(height: 16)
            
            // Jump to time
            Button(action: { showJumpToDialog = true }) {
                Image(systemName: "clock")
                    .font(.system(size: 12))
            }
            .buttonStyle(PlainButtonStyle())
            .help("Jump to Time (⌘J)")
            .keyboardShortcut("j", modifiers: .command)
            .popover(isPresented: $showJumpToDialog) {
                JumpToTimeDialog(
                    timeline: timeline,
                    isPresented: $showJumpToDialog
                )
            }
        }
    }
    
    // MARK: - Navigation Methods
    private func navigateFrames(_ delta: Int) {
        let frameRate = timeline.frameRate
        let frameDuration = 1.0 / Double(frameRate)
        let newTime = timeline.playheadPosition + (frameDuration * Double(delta))
        timeline.setPlayhead(to: max(0, min(timeline.duration, newTime)))
    }
    
    private func jumpToPreviousMarker() {
        let currentTime = timeline.playheadPosition
        if let marker = timeline.markers.reversed().first(where: { $0.time < currentTime }) {
            timeline.setPlayhead(to: marker.time)
        }
    }
    
    private func jumpToNextMarker() {
        let currentTime = timeline.playheadPosition
        if let marker = timeline.markers.first(where: { $0.time > currentTime }) {
            timeline.setPlayhead(to: marker.time)
        }
    }
    
    private func jumpToPreviousEdit() {
        let currentTime = timeline.playheadPosition
        var editPoints: [TimeInterval] = []
        
        // Collect all edit points
        for track in timeline.tracks {
            for clip in track.clips {
                editPoints.append(clip.startTime)
                editPoints.append(clip.startTime + clip.duration ?? 0)
            }
        }
        
        editPoints.sort()
        if let previousEdit = editPoints.reversed().first(where: { $0 < currentTime }) {
            timeline.setPlayhead(to: previousEdit)
        }
    }
    
    private func jumpToNextEdit() {
        let currentTime = timeline.playheadPosition
        var editPoints: [TimeInterval] = []
        
        // Collect all edit points
        for track in timeline.tracks {
            for clip in track.clips {
                editPoints.append(clip.startTime)
                editPoints.append(clip.startTime + clip.duration ?? 0)
            }
        }
        
        editPoints.sort()
        if let nextEdit = editPoints.first(where: { $0 > currentTime }) {
            timeline.setPlayhead(to: nextEdit)
        }
    }
}

// MARK: - Jump to Time Dialog
struct JumpToTimeDialog: View {
    @ObservedObject var timeline: TimelineModel
    @Binding var isPresented: Bool
    @State private var timecodeText = ""
    @FocusState private var isTextFieldFocused: Bool
    
    public var body: some View {
        VStack(spacing: 12) {
            Text("Jump to Time")
                .font(.system(size: 13, weight: .semibold))
            
            TextField("00:00:00:00", text: $timecodeText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .font(.system(size: 12, design: .monospaced))
                .focused($isTextFieldFocused)
                .onSubmit {
                    jumpToTime()
                }
            
            HStack {
                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.escape, modifiers: [])
                
                Spacer()
                
                Button("Jump") {
                    jumpToTime()
                }
                .keyboardShortcut(.return, modifiers: [])
                .buttonStyle(BorderedProminentButtonStyle())
            }
            .controlSize(.small)
        }
        .padding()
        .frame(width: 250)
        .onAppear {
            timecodeText = formatTimecode(timeline.playheadPosition)
            isTextFieldFocused = true
        }
    }
    
    private func jumpToTime() {
        if let time = parseTimecode(timecodeText) {
            timeline.setPlayhead(to: min(timeline.duration, max(0, time)))
            isPresented = false
        }
    }
    
    private func formatTimecode(_ time: TimeInterval) -> String {
        let hours = Int(time) / 3600
        let minutes = (Int(time) % 3600) / 60
        let seconds = Int(time) % 60
        let frames = Int((time.truncatingRemainder(dividingBy: 1)) * 30)
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, seconds, frames)
    }
    
    private func parseTimecode(_ text: String) -> TimeInterval? {
        let components = text.split(separator: ":").compactMap { Int($0) }
        guard components.count >= 3 else { return nil }
        
        let hours = components[0]
        let minutes = components[1]
        let seconds = components[2]
        let frames = components.count > 3 ? components[3] : 0
        
        return TimeInterval(hours * 3600 + minutes * 60 + seconds) + TimeInterval(frames) / 30.0
    }
}

// MARK: - Timeline Model Extensions
extension TimelineModel {
    var totalTrackHeight: CGFloat {
        tracks.reduce(0) { $0 + $1.height } + CGFloat(tracks.count - 1)
    }
    
    var defaultFrameRate: Double {
        30.0 // Default to 30fps, should be configurable
    }
    
    var audioScrubbingEnabled: Bool {
        true // Should be a user preference
    }
    
    var snapToGrid: Bool {
        true // Should be a user preference
    }
    
    var gridInterval: TimeInterval {
        1.0 // 1 second grid, should be configurable
    }
    
    var audioEngine: PlayheadAudioEngine? {
        nil // Placeholder for audio engine
    }
}

// MARK: - Audio Engine Protocol
protocol PlayheadAudioEngine {
    func startScrubbing()
    func scrubTo(time: TimeInterval)
    func stopScrubbing()
}
