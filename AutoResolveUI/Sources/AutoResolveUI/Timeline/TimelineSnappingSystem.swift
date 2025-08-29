// AUTORESOLVE V3.0 - TIMELINE SNAPPING SYSTEM
import Combine
// Professional snapping with magnetic timeline, visual indicators, and haptic feedback

import SwiftUI
import AppKit
import AVFoundation

// MARK: - Snapping Manager
class SnappingManager: ObservableObject {
    @Published var isEnabled = true
    @Published var snapToClips = true
    @Published var snapToPlayhead = true
    @Published var snapToMarkers = true
    @Published var snapToGrid = false
    @Published var magneticTimeline = true
    
    @Published var snapThreshold: CGFloat = 10 // pixels
    @Published var gridInterval: TimeInterval = 1.0 // seconds
    @Published var activeSnapPoints: [SnapPoint] = []
    @Published var currentSnapTarget: SnapPoint?
    
    struct SnapPoint: Identifiable, Equatable {
        public let id = UUID()
        let time: TimeInterval
        let type: SnapType
        let sourceID: UUID?
        let position: CGFloat
        
        enum SnapType {
            case clipStart
            case clipEnd
            case playhead
            case marker
            case grid
            case centerLine
        }
    }
    
    func findSnapTarget(
        for time: TimeInterval,
        position: CGFloat,
        timeline: TimelineModel,
        excludingClip: UUID? = nil
    ) -> SnapPoint? {
        guard isEnabled else { return nil }
        
        var snapPoints: [SnapPoint] = []
        
        // Collect clip edges
        if snapToClips {
            for track in timeline.tracks {
                for clip in track.clips {
                    if clip.id == excludingClip { continue }
                    
                    let startPos = timeline.xFromTime(clip.startTime)
                    let endPos = timeline.xFromTime(clip.startTime + clip.duration ?? 0)
                    
                    snapPoints.append(SnapPoint(
                        time: clip.startTime,
                        type: .clipStart,
                        sourceID: clip.id,
                        position: startPos
                    ))
                    
                    snapPoints.append(SnapPoint(
                        time: clip.startTime + clip.duration ?? 0,
                        type: .clipEnd,
                        sourceID: clip.id,
                        position: endPos
                    ))
                }
            }
        }
        
        // Add playhead position
        if snapToPlayhead {
            let playheadPos = timeline.xFromTime(timeline.playheadPosition)
            snapPoints.append(SnapPoint(
                time: timeline.playheadPosition,
                type: .playhead,
                sourceID: nil,
                position: playheadPos
            ))
        }
        
        // Add markers
        if snapToMarkers {
            for marker in timeline.markers {
                let markerPos = timeline.xFromTime(marker.time)
                snapPoints.append(SnapPoint(
                    time: marker.time,
                    type: .marker,
                    sourceID: marker.id,
                    position: markerPos
                ))
            }
        }
        
        // Add grid points
        if snapToGrid {
            let startGrid = floor(time / gridInterval) * gridInterval
            for i in -2...2 {
                let gridTime = startGrid + Double(i) * gridInterval
                if gridTime >= 0 && gridTime <= timeline.duration {
                    let gridPos = timeline.xFromTime(gridTime)
                    snapPoints.append(SnapPoint(
                        time: gridTime,
                        type: .grid,
                        sourceID: nil,
                        position: gridPos
                    ))
                }
            }
        }
        
        // Find closest snap point within threshold
        let closestPoint = snapPoints
            .filter { abs($0.position - position) <= snapThreshold }
            .min { abs($0.position - position) < abs($1.position - position) }
        
        if closestPoint != currentSnapTarget {
            currentSnapTarget = closestPoint
            if closestPoint != nil {
                provideHapticFeedback()
            }
        }
        
        activeSnapPoints = snapPoints.filter { abs($0.position - position) <= snapThreshold * 3 }
        
        return closestPoint
    }
    
    func clearSnapTargets() {
        currentSnapTarget = nil
        activeSnapPoints = []
    }
    
    private func provideHapticFeedback() {
        NSHapticFeedbackManager.defaultPerformer.perform(
            .alignment,
            performanceTime: .now
        )
    }
}

// MARK: - Snap Indicator View
struct SnapIndicatorView: View {
    @ObservedObject var snappingManager: SnappingManager
    let timeline: TimelineModel
    
    public var body: some View {
        ZStack {
            // Active snap lines
            ForEach(snappingManager.activeSnapPoints) { snapPoint in
                SnapLine(
                    snapPoint: snapPoint,
                    isActive: snapPoint == snappingManager.currentSnapTarget,
                    timeline: timeline
                )
            }
            
            // Magnetic alignment indicator
            if snappingManager.magneticTimeline,
               let target = snappingManager.currentSnapTarget {
                MagneticIndicator(snapPoint: target, timeline: timeline)
            }
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Snap Line
struct SnapLine: View {
    let snapPoint: SnappingManager.SnapPoint
    let isActive: Bool
    let timeline: TimelineModel
    
    @State private var animateIn = false
    
    private var lineColor: Color {
        switch snapPoint.type {
        case .clipStart, .clipEnd:
            return .yellow
        case .playhead:
            return .red
        case .marker:
            return .green
        case .grid:
            return .gray
        case .centerLine:
            return .cyan
        }
    }
    
    private var lineOpacity: Double {
        if isActive {
            return animateIn ? 0.8 : 0.6
        } else {
            return 0.3
        }
    }
    
    public var body: some View {
        Rectangle()
            .fill(lineColor)
            .frame(width: isActive ? 2 : 1)
            .opacity(lineOpacity)
            .position(x: snapPoint.position, y: timeline.totalTrackHeight / 2)
            .scaleEffect(y: animateIn ? 1 : 0.8)
            .animation(.spring(response: 0.2, dampingFraction: 0.8), value: animateIn)
            .onAppear {
                withAnimation {
                    animateIn = true
                }
            }
    }
}

// MARK: - Magnetic Indicator
struct MagneticIndicator: View {
    let snapPoint: SnappingManager.SnapPoint
    let timeline: TimelineModel
    
    @State private var pulseAnimation = false
    
    public var body: some View {
        VStack(spacing: 0) {
            // Top indicator
            MagneticArrow(pointing: .down)
                .position(x: snapPoint.position, y: -10)
            
            // Bottom indicator
            MagneticArrow(pointing: .up)
                .position(x: snapPoint.position, y: timeline.totalTrackHeight + 10)
        }
        .scaleEffect(pulseAnimation ? 1.1 : 1.0)
        .opacity(pulseAnimation ? 0.8 : 1.0)
        .animation(
            Animation.easeInOut(duration: 0.3)
                .repeatForever(autoreverses: true),
            value: pulseAnimation
        )
        .onAppear {
            pulseAnimation = true
        }
    }
}

// MARK: - Magnetic Arrow
struct MagneticArrow: View {
    let pointing: Direction
    
    enum Direction {
        case up, down
    }
    
    public var body: some View {
        Path { path in
            switch pointing {
            case .up:
                path.move(to: CGPoint(x: 0, y: 8))
                path.addLine(to: CGPoint(x: -6, y: 0))
                path.addLine(to: CGPoint(x: 6, y: 0))
            case .down:
                path.move(to: CGPoint(x: 0, y: 0))
                path.addLine(to: CGPoint(x: -6, y: 8))
                path.addLine(to: CGPoint(x: 6, y: 8))
            }
            path.closeSubpath()
        }
        .fill(Color.yellow)
        .frame(width: 12, height: 8)
    }
}

// MARK: - Snapping Settings View
struct SnappingSettingsView: View {
    @ObservedObject var snappingManager: SnappingManager
    @State private var showAdvancedSettings = false
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Main toggle
            Toggle("Enable Snapping", isOn: $snappingManager.isEnabled)
                .toggleStyle(SwitchToggleStyle())
                .font(.system(size: 12, weight: .medium))
            
            if snappingManager.isEnabled {
                Divider()
                
                // Snap targets
                VStack(alignment: .leading, spacing: 8) {
                    Text("Snap To:")
                        .font(.system(size: 11, weight: .semibold))
                    
                    Toggle("Clip Edges", isOn: $snappingManager.snapToClips)
                        .font(.system(size: 11))
                    
                    Toggle("Playhead", isOn: $snappingManager.snapToPlayhead)
                        .font(.system(size: 11))
                    
                    Toggle("Markers", isOn: $snappingManager.snapToMarkers)
                        .font(.system(size: 11))
                    
                    Toggle("Grid", isOn: $snappingManager.snapToGrid)
                        .font(.system(size: 11))
                    
                    if snappingManager.snapToGrid {
                        HStack {
                            Text("Grid Interval:")
                                .font(.system(size: 10))
                            
                            Picker("", selection: $snappingManager.gridInterval) {
                                Text("0.1s").tag(0.1)
                                Text("0.5s").tag(0.5)
                                Text("1s").tag(1.0)
                                Text("5s").tag(5.0)
                                Text("10s").tag(10.0)
                            }
                            .pickerStyle(SegmentedPickerStyle())
                            .frame(width: 200)
                        }
                        .padding(.leading, 20)
                    }
                }
                
                Divider()
                
                // Advanced settings
                DisclosureGroup("Advanced", isExpanded: $showAdvancedSettings) {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Magnetic Timeline", isOn: $snappingManager.magneticTimeline)
                            .font(.system(size: 11))
                            .help("Automatically close gaps when moving clips")
                        
                        HStack {
                            Text("Snap Threshold:")
                                .font(.system(size: 11))
                            
                            Slider(value: $snappingManager.snapThreshold, in: 5...30, step: 5)
                                .frame(width: 100)
                            
                            Text("\(Int(snappingManager.snapThreshold))px")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.top, 8)
                }
                .font(.system(size: 11))
            }
        }
        .padding()
        .frame(width: 280)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}

// MARK: - Clip Snapping Extension
extension TimelineClip {
    func snapEdges(to timeline: TimelineModel, excluding: UUID? = nil) -> [TimeInterval] {
        var snapTimes: [TimeInterval] = []
        
        for track in timeline.tracks {
            for clip in track.clips {
                if clip.id == excluding || clip.id == self.id { continue }
                
                snapTimes.append(clip.startTime)
                snapTimes.append(clip.startTime + clip.duration ?? 0)
            }
        }
        
        return snapTimes
    }
}

// MARK: - Magnetic Timeline Helper
struct MagneticTimelineHelper {
    static func closeGaps(in track: TimelineTrack, after time: TimeInterval) -> [ClipMove] {
        var moves: [ClipMove] = []
        let sortedClips = track.clips.sorted { (a, b) in a.startTime < b.startTime }
        
        let timeAsCMTime = CMTime(seconds: time, preferredTimescale: 600)
        var lastEndTime = time
        var lastEndTimeAsCMTime = timeAsCMTime
        for clip in sortedClips where clip.startTime > timeAsCMTime {
            if clip.startTime > lastEndTimeAsCMTime {
                // There's a gap, close it
                moves.append(ClipMove(
                    clipID: clip.id,
                    newStartTime: lastEndTime
                ))
                let clipDurationSeconds = CMTimeGetSeconds(clip.duration)
                lastEndTime = lastEndTime + clipDurationSeconds
                lastEndTimeAsCMTime = CMTime(seconds: lastEndTime, preferredTimescale: 600)
            } else {
                let clipStartSeconds = CMTimeGetSeconds(clip.startTime)
                let clipDurationSeconds = CMTimeGetSeconds(clip.duration)
                lastEndTime = clipStartSeconds + clipDurationSeconds
                lastEndTimeAsCMTime = CMTimeAdd(clip.startTime, clip.duration)
            }
        }
        
        return moves
    }
    
    struct ClipMove {
        let clipID: UUID
        let newStartTime: TimeInterval
    }
}

// MARK: - Snapping Keyboard Shortcuts
struct SnappingKeyboardShortcuts: ViewModifier {
    @ObservedObject var snappingManager: SnappingManager
    
    func body(content: Content) -> some View {
        content
            .keyboardShortcut("s", modifiers: [])
            .onAppear {
                NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
                    handleKeyEvent(event)
                    return event
                }
            }
    }
    
    private func handleKeyEvent(_ event: NSEvent) {
        switch event.charactersIgnoringModifiers {
        case "s":
            snappingManager.isEnabled.toggle()
        case "g":
            if event.modifierFlags.contains(.command) {
                snappingManager.snapToGrid.toggle()
            }
        case "m":
            if event.modifierFlags.contains(.option) {
                snappingManager.magneticTimeline.toggle()
            }
        default:
            break
        }
    }
}

// MARK: - Timeline Model Extension
extension TimelineModel {
    // Removed duplicate markers property - already defined in TimelineModel
    
    // findClip method is already defined in TimelineModel
    
    func findSnapPoint(for time: TimeInterval, excluding clipID: UUID?) -> TimeInterval? {
        let threshold: TimeInterval = 0.1 // 100ms threshold
        var snapTimes: [TimeInterval] = []
        
        // Collect all snap points
        for track in tracks {
            for clip in track.clips {
                if clip.id == clipID { continue }
                snapTimes.append(clip.startTime)
                snapTimes.append(clip.startTime + clip.duration ?? 0)
            }
        }
        
        // Add playhead
        snapTimes.append(playheadPosition)
        
        // Find closest
        let closest = snapTimes.min { abs($0 - time) < abs($1 - time) }
        
        if let closest = closest, abs(closest - time) < threshold {
            return closest
        }
        
        return nil
    }
}

// MARK: - Timeline Marker
// Using TimelineMarker from TimelineModel.swift
