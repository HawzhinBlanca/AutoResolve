import SwiftUI
import AppKit

// MARK: - Main Timeline View

public struct TimelineView: View {
    @StateObject private var timeline = TimelineModel()
    @State private var selectedTool: EditTool = .select
    @State private var isDragging = false
    @State private var dragOffset: CGSize = .zero
    @State private var hoveredClipID: UUID?
    @State private var timelineWidth: CGFloat = 2000
    
    private let rulerHeight: CGFloat = 30
    private let trackHeaderWidth: CGFloat = 120
    private let minimumZoom: Double = 10  // 10 pixels per second
    private let maximumZoom: Double = 200  // 200 pixels per second
    
    enum EditTool: String, CaseIterable {
        case select = "arrow.up.left"
        case blade = "scissors"
        case hand = "hand.raised"
        case zoom = "magnifyingglass"
        
        var cursor: NSCursor {
            switch self {
            case .select: return .arrow
            case .blade: return .crosshair
            case .hand: return .openHand
            case .zoom: return .crosshair
            }
        }
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            TimelineToolbar(selectedTool: $selectedTool, timeline: timeline)
                .frame(height: 44)
                .background(.ultraThinMaterial)
            
            // Timeline area
            GeometryReader { geometry in
                VStack(spacing: 0) {
                    // Timeline ruler
                    TimelineRuler(
                        timeline: timeline,
                        width: timelineWidth,
                        offset: trackHeaderWidth
                    )
                    .frame(height: rulerHeight)
                    
                    // Tracks area
                    ScrollView([.horizontal, .vertical], showsIndicators: true) {
                        HStack(spacing: 0) {
                            // Track headers
                            VStack(spacing: 1) {
                                ForEach(timeline.tracks) { track in
                                    TrackHeader(track: track, timeline: timeline)
                                        .frame(height: track.height)
                                }
                            }
                            .frame(width: trackHeaderWidth)
                            .background(Color(NSColor.controlBackgroundColor))
                            
                            // Track content
                            ZStack(alignment: .topLeading) {
                                // Track lanes
                                LazyVStack(spacing: 1) {
                                    ForEach(Array(timeline.tracks.enumerated()), id: \.element.id) { index, track in
                                        TrackLane(
                                            track: track,
                                            trackIndex: index,
                                            timeline: timeline,
                                            width: timelineWidth,
                                            selectedTool: selectedTool,
                                            hoveredClipID: $hoveredClipID
                                        )
                                        .frame(height: track.height)
                                    }
                                }
                                
                                // Playhead
                                PlayheadView(timeline: timeline, height: totalTracksHeight)
                                    .offset(x: timeline.xFromTime(timeline.playheadPosition))
                            }
                            .frame(width: timelineWidth)
                            .background(Color(NSColor.windowBackgroundColor))
                        }
                    }
                    .offset(x: timeline.scrollOffset)
                    .animation(.spring(response: 0.7), value: timeline.scrollOffset)
                    .animation(.spring(response: 0.7), value: timeline.zoomLevel)
                    .onAppear {
                        let sp = Performance.begin("TimelineWidthUpdate")
                        updateTimelineWidth()
                        Performance.end(sp, "TimelineWidthUpdate")
                    }
                }
            }
            
            // Timeline controls
            TimelineControls(timeline: timeline)
                .frame(height: 60)
                .background(.ultraThinMaterial)
        }
        .onReceive(timeline.$duration) { _ in
            let sp = Performance.begin("TimelineWidthUpdate")
            updateTimelineWidth()
            Performance.end(sp, "TimelineWidthUpdate")
        }
        .onReceive(timeline.$zoomLevel) { _ in
            let sp = Performance.begin("TimelineWidthUpdate")
            updateTimelineWidth()
            Performance.end(sp, "TimelineWidthUpdate")
        }
    }
    
    private var totalTracksHeight: CGFloat {
        timeline.tracks.reduce(0) { $0 + $1.height } + CGFloat(timeline.tracks.count - 1)
    }
    
    private func updateTimelineWidth() {
        timelineWidth = CGFloat(timeline.duration * timeline.zoomLevel) + 200
    }
}

// MARK: - Timeline Toolbar

struct TimelineToolbar: View {
    @Binding var selectedTool: TimelineView.EditTool
    @ObservedObject var timeline: TimelineModel
    @State private var zoomSliderPos: Double = 0.5
    private let minZoom: Double = 10
    private let maxZoom: Double = 200
    
    var body: some View {
        HStack(spacing: 12) {
            // Edit tools
            ForEach(TimelineView.EditTool.allCases, id: \.self) { tool in
                Button(action: { selectedTool = tool }) {
                    Image(systemName: tool.rawValue)
                        .font(.system(size: 16))
                        .frame(width: 32, height: 32)
                        .background(selectedTool == tool ? Color.accentColor.opacity(0.2) : Color.clear)
                        .cornerRadius(6)
                }
                .buttonStyle(PlainButtonStyle())
                .help(tool.rawValue.capitalized)
            }
            
            Divider()
                .frame(height: 24)
            
            // Edit actions
            Button(action: { timeline.cutAtPlayhead() }) {
                Image(systemName: "scissors")
            }
            .help("Cut at playhead (⌘B)")
            .keyboardShortcut("b", modifiers: .command)
            
            Button(action: { timeline.deleteSelected() }) {
                Image(systemName: "trash")
            }
            .disabled(timeline.selectedClips.isEmpty)
            .help("Delete selected (Delete)")
            .keyboardShortcut(.delete, modifiers: [])
            
            Button(action: { timeline.duplicateSelected() }) {
                Image(systemName: "doc.on.doc")
            }
            .disabled(timeline.selectedClips.isEmpty)
            .help("Duplicate selected (⌘D)")
            .keyboardShortcut("d", modifiers: .command)
            
            Spacer()
            
            // Zoom controls
            Text("Zoom:")
                .font(.caption)
            
            Slider(value: $zoomSliderPos, in: 0...1)
                .frame(width: 150)
                .onChange(of: zoomSliderPos) { _, newVal in
                    let mapped = minZoom * pow(maxZoom / minZoom, newVal)
                    timeline.zoomLevel = mapped
                }
                .onChange(of: timeline.zoomLevel) { _, newZoom in
                    let pos = log(newZoom / minZoom) / log(maxZoom / minZoom)
                    zoomSliderPos = max(0, min(1, pos))
                }
                .onAppear {
                    let pos = log(timeline.zoomLevel / minZoom) / log(maxZoom / minZoom)
                    zoomSliderPos = max(0, min(1, pos))
                }
            
            Button(action: { timeline.zoomLevel = 50 }) {
                Text("Fit")
            }
            .help("Fit timeline to view")
        }
        .padding(.horizontal)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Timeline Ruler

struct TimelineRuler: View {
    @ObservedObject var timeline: TimelineModel
    let width: CGFloat
    let offset: CGFloat
    
    var body: some View {
        HStack(spacing: 0) {
            // Empty space for track headers
            Color.clear
                .frame(width: offset)
            
            // Ruler content
            Canvas { context, size in
                // Draw background
                context.fill(
                    Path(CGRect(origin: .zero, size: size)),
                    with: .color(Color(NSColor.controlBackgroundColor))
                )
                
                // Draw time markers
                let secondsPerPixel = 1.0 / timeline.zoomLevel
                let markerInterval = getMarkerInterval(secondsPerPixel: secondsPerPixel)
                
                var time: TimeInterval = 0
                while time <= timeline.duration {
                    let x = CGFloat(time * timeline.zoomLevel)
                    
                    // Draw major markers
                    if time.truncatingRemainder(dividingBy: markerInterval.major) == 0 {
                        context.stroke(
                            Path { path in
                                path.move(to: CGPoint(x: x, y: size.height - 15))
                                path.addLine(to: CGPoint(x: x, y: size.height))
                            },
                            with: .color(.primary),
                            lineWidth: 1
                        )
                        
                        // Draw timecode
                        let timecode = timeline.timecode(for: time)
                        context.draw(
                            Text(timecode)
                                .font(.system(size: 10))
                                .foregroundColor(.secondary),
                            at: CGPoint(x: x + 5, y: 10)
                        )
                    }
                    // Draw minor markers
                    else if time.truncatingRemainder(dividingBy: markerInterval.minor) == 0 {
                        context.stroke(
                            Path { path in
                                path.move(to: CGPoint(x: x, y: size.height - 8))
                                path.addLine(to: CGPoint(x: x, y: size.height))
                            },
                            with: .color(.secondary.opacity(0.5)),
                            lineWidth: 0.5
                        )
                    }
                    
                    time += markerInterval.minor
                }
            }
            .frame(width: width)
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private func getMarkerInterval(secondsPerPixel: Double) -> (major: TimeInterval, minor: TimeInterval) {
        if secondsPerPixel > 2 {
            return (30, 10)  // Show every 30s/10s
        } else if secondsPerPixel > 0.5 {
            return (10, 1)   // Show every 10s/1s
        } else {
            return (1, 0.1)  // Show every 1s/0.1s
        }
    }
}

// MARK: - Track Header

struct TrackHeader: View {
    let track: TimelineTrack
    @ObservedObject var timeline: TimelineModel
    
    var body: some View {
        HStack(spacing: 8) {
            // Track type indicator
            Text(track.type.rawValue)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .frame(width: 20)
            
            // Track name
            Text(track.name)
                .font(.system(size: 12))
                .lineLimit(1)
            
            Spacer()
            
            // Track controls
            Button(action: { toggleTrackEnabled() }) {
                Image(systemName: track.isEnabled ? "eye" : "eye.slash")
                    .font(.system(size: 12))
            }
            .buttonStyle(PlainButtonStyle())
            .help(track.isEnabled ? "Disable track" : "Enable track")
            
            Button(action: { toggleTrackLocked() }) {
                Image(systemName: track.isLocked ? "lock.fill" : "lock.open")
                    .font(.system(size: 12))
            }
            .buttonStyle(PlainButtonStyle())
            .help(track.isLocked ? "Unlock track" : "Lock track")
        }
        .padding(.horizontal, 8)
        .background(Color(NSColor.controlBackgroundColor))
        .overlay(
            Rectangle()
                .stroke(Color.secondary.opacity(0.2), lineWidth: 0.5)
        )
    }
    
    private func toggleTrackEnabled() {
        if let index = timeline.tracks.firstIndex(where: { $0.id == track.id }) {
            timeline.tracks[index].isEnabled.toggle()
        }
    }
    
    private func toggleTrackLocked() {
        if let index = timeline.tracks.firstIndex(where: { $0.id == track.id }) {
            timeline.tracks[index].isLocked.toggle()
        }
    }
}

// MARK: - Track Lane

struct TrackLane: View {
    let track: TimelineTrack
    let trackIndex: Int
    @ObservedObject var timeline: TimelineModel
    let width: CGFloat
    let selectedTool: TimelineView.EditTool
    @Binding var hoveredClipID: UUID?
    
    var body: some View {
        ZStack(alignment: .leading) {
            // Track background
            Rectangle()
                .fill(Color.secondary.opacity(0.05))
                .overlay(
                    Rectangle()
                        .stroke(Color.secondary.opacity(0.2), lineWidth: 0.5)
                )
            
            // Clips
            ForEach(track.clips) { clip in
                ClipView(
                    clip: clip,
                    timeline: timeline,
                    isSelected: timeline.selectedClips.contains(clip.id),
                    isHovered: hoveredClipID == clip.id
                )
                .offset(x: timeline.xFromTime(clip.startTime))
                .onHover { hovering in
                    hoveredClipID = hovering ? clip.id : nil
                }
                .onTapGesture {
                    handleClipClick(clip: clip)
                }
            }
        }
        .frame(width: width, height: track.height)
    }
    
    private func handleClipClick(clip: TimelineClip) {
        switch selectedTool {
        case .select:
            timeline.selectClip(id: clip.id, multi: NSEvent.modifierFlags.contains(.shift))
        case .blade:
            // Cut at click position
            timeline.cutAtPlayhead()
        default:
            break
        }
    }
}

// MARK: - Playhead View

struct PlayheadView: View {
    @ObservedObject var timeline: TimelineModel
    let height: CGFloat
    
    var body: some View {
        VStack(spacing: 0) {
            // Playhead handle
            Path { path in
                path.move(to: CGPoint(x: 0, y: 0))
                path.addLine(to: CGPoint(x: -6, y: -8))
                path.addLine(to: CGPoint(x: 6, y: -8))
                path.closeSubpath()
            }
            .fill(Color.red)
            .offset(y: -8)
            
            // Playhead line
            Rectangle()
                .fill(Color.red)
                .frame(width: 1, height: height)
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Timeline Controls

struct TimelineControls: View {
    @ObservedObject var timeline: TimelineModel
    
    var body: some View {
        HStack(spacing: 20) {
            // Transport controls
            HStack(spacing: 8) {
                Button(action: { timeline.setPlayhead(to: 0) }) {
                    Image(systemName: "backward.end.fill")
                }
                .help("Go to start")
                
                Button(action: { timeline.setPlayhead(to: timeline.playheadPosition - 1) }) {
                    Image(systemName: "backward.frame.fill")
                }
                .help("Previous frame")
                
                Button(action: { /* Play/Pause */ }) {
                    Image(systemName: "play.fill")
                }
                .help("Play/Pause (Space)")
                .keyboardShortcut(.space, modifiers: [])
                
                Button(action: { timeline.setPlayhead(to: timeline.playheadPosition + 1) }) {
                    Image(systemName: "forward.frame.fill")
                }
                .help("Next frame")
                
                Button(action: { timeline.setPlayhead(to: timeline.duration) }) {
                    Image(systemName: "forward.end.fill")
                }
                .help("Go to end")
            }
            .font(.title2)
            
            // Timecode display
            Text(timeline.timecode(for: timeline.playheadPosition))
                .font(.system(size: 18, weight: .medium, design: .monospaced))
                .frame(width: 120)
                .textSelection(.enabled)
            
            Spacer()
            
            // Timeline info
            VStack(alignment: .trailing, spacing: 4) {
                Text("Duration: \(timeline.timecode(for: timeline.duration))")
                    .font(.caption)
                Text("\(timeline.tracks.flatMap { $0.clips }.count) clips")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
    }
}