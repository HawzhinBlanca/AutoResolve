// AUTORESOLVE V3.0 - VIRTUAL SCROLLING TIMELINE
// High-performance timeline with virtual scrolling

import SwiftUI
import Combine

// MARK: - Virtual Scrolling Timeline
struct VirtualScrollingTimeline: View {
    @EnvironmentObject var timelineModel: TimelineModel
    @StateObject private var scrollManager = TimelineScrollManager()
    @State private var visibleRange: Range<Int> = 0..<10
    @State private var contentOffset: CGFloat = 0
    @State private var isDragging = false
    @State private var zoomLevel: CGFloat = 1.0
    @State private var selectedClips: Set<UUID> = []
    @State private var playheadPosition: CGFloat = 0
    @State private var isPlayheadDragging = false
    
    // Timeline constants
    let trackHeight: CGFloat = 60
    let minTrackHeight: CGFloat = 40
    let maxTrackHeight: CGFloat = 120
    let rulerHeight: CGFloat = 30
    let headerWidth: CGFloat = 120
    
    public var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                // Timeline Header with Zoom Controls
                TimelineHeader(
                    zoomLevel: $zoomLevel,
                    duration: timelineModel.duration
                )
                .frame(height: 44)
                .background(Color(red: 0.15, green: 0.15, blue: 0.15))
                
                // Main Timeline Area
                HStack(spacing: 0) {
                    // Track Headers
                    TrackHeadersView(
                        tracks: convertToTimelineTracks(timelineModel.tracks),
                        trackHeight: trackHeight * zoomLevel,
                        visibleRange: visibleRange,
                        timeline: timelineModel
                    )
                    .frame(width: headerWidth)
                    .background(Color(red: 0.12, green: 0.12, blue: 0.12))
                    
                    // Scrollable Timeline Content
                    TimelineScrollView(
                        tracks: convertToTimelineTracks(timelineModel.tracks),
                        visibleRange: visibleRange,
                        contentOffset: contentOffset,
                        trackHeight: trackHeight * zoomLevel,
                        zoomLevel: zoomLevel,
                        selectedClips: $selectedClips,
                        playheadPosition: $playheadPosition,
                        isPlayheadDragging: $isPlayheadDragging,
                        geometry: geometry
                    )
                }
                
                // Timeline Controls Bar
                TimelineControlsBar(
                    playheadPosition: $playheadPosition,
                    duration: timelineModel.duration,
                    zoomLevel: $zoomLevel
                )
                .frame(height: 32)
                .background(Color(red: 0.15, green: 0.15, blue: 0.15))
            }
            .onAppear {
                calculateVisibleRange(in: geometry)
            }
            .onChange(of: contentOffset) { _ in
                calculateVisibleRange(in: geometry)
            }
            .onChange(of: zoomLevel) { _ in
                calculateVisibleRange(in: geometry)
            }
        }
    }
    
    private func calculateVisibleRange(in geometry: GeometryProxy) {
        let viewportHeight = geometry.size.height - 44 - 32 // Minus header and controls
        let totalTracks = timelineModel.tracks.count
        let trackHeight = self.trackHeight * zoomLevel
        
        let firstVisible = max(0, Int(contentOffset / trackHeight))
        let visibleCount = Int(ceil(viewportHeight / trackHeight)) + 1 // Add buffer
        let lastVisible = min(totalTracks, firstVisible + visibleCount)
        
        visibleRange = firstVisible..<lastVisible
    }
}

// MARK: - Timeline Header
struct TimelineHeader: View {
    @Binding var zoomLevel: CGFloat
    let duration: TimeInterval
    
    public var body: some View {
        HStack {
            // Timecode Display
            Text(formatTimecode(0))
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.white)
                .frame(width: 100)
            
            Spacer()
            
            // Zoom Controls
            HStack(spacing: 12) {
                Button(action: { zoomOut() }) {
                    Image(systemName: "minus.magnifyingglass")
                        .foregroundColor(Color(red: 0.7, green: 0.7, blue: 0.7))
                }
                .buttonStyle(.plain)
                
                Slider(value: $zoomLevel, in: 0.25...4.0)
                    .frame(width: 150)
                
                Button(action: { zoomIn() }) {
                    Image(systemName: "plus.magnifyingglass")
                        .foregroundColor(Color(red: 0.7, green: 0.7, blue: 0.7))
                }
                .buttonStyle(.plain)
                
                Button(action: { zoomToFit() }) {
                    Text("Fit")
                        .font(.caption)
                        .foregroundColor(Color(red: 0.7, green: 0.7, blue: 0.7))
                }
                .buttonStyle(.plain)
            }
            
            Spacer()
            
            // Duration Display
            Text(formatTimecode(duration))
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.white)
                .frame(width: 100)
        }
        .padding(.horizontal)
    }
    
    private func zoomIn() {
        withAnimation(.easeInOut(duration: 0.15)) {
            zoomLevel = min(4.0, zoomLevel * 1.2)
        }
    }
    
    private func zoomOut() {
        withAnimation(.easeInOut(duration: 0.15)) {
            zoomLevel = max(0.25, zoomLevel / 1.2)
        }
    }
    
    private func zoomToFit() {
        withAnimation(.easeInOut(duration: 0.25)) {
            zoomLevel = 1.0
        }
    }
    
    private func formatTimecode(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        let frames = Int((seconds.truncatingRemainder(dividingBy: 1)) * 30)
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
    }
}

// MARK: - Track Headers View
struct TrackHeadersView: View {
    let tracks: [TimelineTrack]
    let trackHeight: CGFloat
    let visibleRange: Range<Int>
    @ObservedObject var timeline: TimelineModel
    
    public var body: some View {
        VStack(spacing: 0) {
            // Ruler spacer
            Rectangle()
                .fill(Color.clear)
                .frame(height: 30)
            
            // Track headers
            ForEach(Array(tracks.prefix(10).enumerated()), id: \.offset) { index, track in
                TrackHeader(
                    track: track,
                    timeline: timeline
                )
            }
            
            Spacer()
        }
    }
}

// MARK: - Track Header
struct VirtualTrackHeader: View {
    let track: TimelineTrack
    let height: CGFloat
    let index: Int
    
    @State private var isMuted = false
    @State private var isSolo = false
    @State private var isLocked = false
    @State private var isExpanded = true
    
    public var body: some View {
        HStack(spacing: 8) {
            // Track Type Icon
            Image(systemName: track.type == .video ? "video" : "waveform")
                .font(.system(size: 14))
                .foregroundColor(Color(red: 0.7, green: 0.7, blue: 0.7))
            
            // Track Name
            Text(track.name)
                .font(.caption)
                .foregroundColor(.white)
                .lineLimit(1)
            
            Spacer()
            
            // Track Controls
            HStack(spacing: 4) {
                Button(action: { isMuted.toggle() }) {
                    Image(systemName: isMuted ? "speaker.slash.fill" : "speaker.fill")
                        .font(.system(size: 10))
                        .foregroundColor(isMuted ? .orange : Color(red: 0.7, green: 0.7, blue: 0.7))
                }
                .buttonStyle(.plain)
                
                Button(action: { isSolo.toggle() }) {
                    Text("S")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(isSolo ? .yellow : Color(red: 0.7, green: 0.7, blue: 0.7))
                }
                .buttonStyle(.plain)
                
                Button(action: { isLocked.toggle() }) {
                    Image(systemName: isLocked ? "lock.fill" : "lock.open")
                        .font(.system(size: 10))
                        .foregroundColor(isLocked ? .red : Color(red: 0.7, green: 0.7, blue: 0.7))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 8)
        .frame(height: height)
        .background(index % 2 == 0 ? Color(red: 0.18, green: 0.18, blue: 0.18) : Color(red: 0.16, green: 0.16, blue: 0.16))
        .border(Color.white.opacity(0.1), width: 0.5)
    }
}

// MARK: - Timeline Scroll View
struct TimelineScrollView: View {
    let tracks: [TimelineTrack]
    let visibleRange: Range<Int>
    let contentOffset: CGFloat
    let trackHeight: CGFloat
    let zoomLevel: CGFloat
    @Binding var selectedClips: Set<UUID>
    @Binding var playheadPosition: CGFloat
    @Binding var isPlayheadDragging: Bool
    let geometry: GeometryProxy
    
    @State private var isDraggingClip = false
    @State private var draggedClip: UITimelineClip?
    @State private var snapIndicators: [CGFloat] = []
    
    public var body: some View {
        ScrollView([.horizontal, .vertical], showsIndicators: true) {
            ZStack(alignment: .topLeading) {
                // Timeline Grid Background
                VirtualTimelineGrid(
                    width: timelineWidth,
                    height: timelineHeight,
                    zoomLevel: zoomLevel
                )
                
                // Ruler
                VirtualTimelineRuler(
                    width: timelineWidth,
                    zoomLevel: zoomLevel,
                    duration: totalDuration
                )
                .frame(height: 30)
                
                // Tracks with Clips
                VStack(spacing: 0) {
                    Rectangle()
                        .fill(Color.clear)
                        .frame(height: 30) // Ruler spacer
                    
                    ForEach(Array(visibleRange.lowerBound..<visibleRange.upperBound), id: \.self) { index in
                        if index < tracks.count {
                            VirtualTimelineTrackView(
                                track: tracks[index],
                                trackHeight: trackHeight,
                                zoomLevel: zoomLevel,
                                selectedClips: $selectedClips,
                                isDraggingClip: $isDraggingClip,
                                draggedClip: $draggedClip,
                                snapIndicators: $snapIndicators
                            )
                        }
                    }
                }
                
                // Playhead
                TimelinePlayhead(
                    position: $playheadPosition,
                    height: timelineHeight,
                    isDragging: $isPlayheadDragging
                )
                
                // Snap Indicators
                ForEach(snapIndicators, id: \.self) { position in
                    Rectangle()
                        .fill(Color.yellow.opacity(0.5))
                        .frame(width: 2, height: timelineHeight)
                        .offset(x: position)
                }
            }
            .frame(width: timelineWidth, height: timelineHeight)
        }
    }
    
    var timelineWidth: CGFloat {
        max(geometry.size.width - 120, totalDuration * pixelsPerSecond * zoomLevel)
    }
    
    var timelineHeight: CGFloat {
CGFloat(tracks.count) * trackHeight + 30 // Plus ruler
    }
    
    var totalDuration: TimeInterval {
        tracks.flatMap { $0.clips }.map { $0.endTime.seconds }.max() ?? 120
    }
    
    var pixelsPerSecond: CGFloat {
        10 // Base pixels per second
    }
}

// MARK: - Timeline Grid
struct VirtualTimelineGrid: View {
    let width: CGFloat
    let height: CGFloat
    let zoomLevel: CGFloat
    
    public var body: some View {
        Canvas { context, size in
            // Vertical grid lines (time markers)
            let spacing = 50 * zoomLevel // Grid spacing
            var x: CGFloat = 0
            
            while x < width {
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: x, y: 0))
                        path.addLine(to: CGPoint(x: x, y: height))
                    },
                    with: .color(.white.opacity(0.1)),
                    lineWidth: 0.5
                )
                x += spacing
            }
            
            // Horizontal grid lines (track separators)
            let trackHeight: CGFloat = 60 * zoomLevel
            var y: CGFloat = 30 // Start after ruler
            
            while y < height {
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: width, y: y))
                    },
                    with: .color(.white.opacity(0.1)),
                    lineWidth: 0.5
                )
                y += trackHeight
            }
        }
    }
}

// MARK: - Timeline Ruler
struct VirtualTimelineRuler: View {
    let width: CGFloat
    let zoomLevel: CGFloat
    let duration: TimeInterval
    
    public var body: some View {
        Canvas { context, size in
            let pixelsPerSecond: CGFloat = 10 * zoomLevel
            let majorTickInterval: TimeInterval = max(1, 10 / zoomLevel) // Adaptive tick interval
            var time: TimeInterval = 0
            
            while time <= duration {
                let x = time * pixelsPerSecond
                
                // Major tick
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: x, y: 20))
                        path.addLine(to: CGPoint(x: x, y: 30))
                    },
                    with: .color(.white),
                    lineWidth: 1
                )
                
                // Time label
                let text = Text(formatTime(time))
                    .font(.system(size: 10))
                    .foregroundColor(Color(red: 0.7, green: 0.7, blue: 0.7))
                
                context.draw(
                    text,
                    at: CGPoint(x: x, y: 10),
                    anchor: .center
                )
                
                time += majorTickInterval
            }
        }
        .background(Color(red: 0.12, green: 0.12, blue: 0.12))
    }
    
    private func formatTime(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

// MARK: - Timeline Track View
struct VirtualTimelineTrackView: View {
    let track: TimelineTrack
    let trackHeight: CGFloat
    let zoomLevel: CGFloat
    @Binding var selectedClips: Set<UUID>
    @Binding var isDraggingClip: Bool
    @Binding var draggedClip: UITimelineClip?
    @Binding var snapIndicators: [CGFloat]
    
    public var body: some View {
        ZStack(alignment: .leading) {
            // Track background
            Rectangle()
                .fill(Color(red: 0.18, green: 0.18, blue: 0.18).opacity(0.3))
                .frame(height: trackHeight)
            
            // Clips
            ForEach(track.clips) { clip in
                UITimelineClipView(
                    clip: clip,
                    isSelected: selectedClips.contains(clip.id),
                    trackHeight: trackHeight,
                    zoomLevel: zoomLevel,
                    onSelect: {
                        toggleSelection(clip.id)
                    },
                    onDrag: { isDragging in
                        if isDragging {
                            isDraggingClip = true
                            draggedClip = clip
                            calculateSnapPoints()
                        } else {
                            isDraggingClip = false
                            draggedClip = nil
                            snapIndicators = []
                        }
                    }
                )
                .offset(x: CGFloat(clip.startTime.seconds) * 10 * zoomLevel)
            }
        }
        .frame(height: trackHeight)
    }
    
    private func toggleSelection(_ clipId: UUID) {
        if selectedClips.contains(clipId) {
            selectedClips.remove(clipId)
        } else {
            selectedClips.insert(clipId)
        }
    }
    
    private func calculateSnapPoints() {
        // Calculate snap points from all clips
        var points: [CGFloat] = []
        
        for clip in track.clips {
            if clip.id != draggedClip?.id {
                points.append(CGFloat(clip.startTime.seconds) * 10 * zoomLevel)
                points.append(CGFloat(clip.endTime.seconds) * 10 * zoomLevel)
            }
        }
        
        snapIndicators = points
    }
}

// MARK: - Timeline Clip View
struct UITimelineClipView: View {
    let clip: UITimelineClip
    let isSelected: Bool
    let trackHeight: CGFloat
    let zoomLevel: CGFloat
    let onSelect: () -> Void
    let onDrag: (Bool) -> Void
    
    @State private var isDragging = false
    @State private var dragOffset = CGSize.zero
    @State private var isHovering = false
    
    public var body: some View {
        ZStack(alignment: .leading) {
            // Clip background
            RoundedRectangle(cornerRadius: 4)
                .fill(clipColor)
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
                )
            
            // Clip content
            HStack {
                // Thumbnail (if available)
                if let image = clip.thumbnail {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: trackHeight - 8, height: trackHeight - 8)
                        .clipped()
                        .cornerRadius(2)
                        .padding(4)
                }
                
                // Clip name
                Text(clip.name)
                    .font(.caption)
                    .foregroundColor(.white)
                    .lineLimit(1)
                    .padding(.horizontal, 8)
                
                Spacer()
            }
            
            // Hover overlay
            if isHovering {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.white.opacity(0.1))
            }
        }
        .frame(width: CGFloat(clip.duration ?? 0.seconds) * 10 * zoomLevel, height: trackHeight - 4)
        .offset(dragOffset)
        .onTapGesture {
            onSelect()
        }
        .onHover { hovering in
            isHovering = hovering
        }
        .gesture(
            DragGesture()
                .onChanged { value in
                    if !isDragging {
                        isDragging = true
                        onDrag(true)
                    }
                    dragOffset = value.translation
                }
                .onEnded { value in
                    isDragging = false
                    onDrag(false)
                    // Apply the drag to the clip position
                    applyDrag(value.translation.width)
                    dragOffset = .zero
                }
        )
        .animation(.interactiveSpring(), value: dragOffset)
    }
    
    var clipColor: Color {
        switch clip.type {
        case .video: return .blue
        case .audio: return .green
        case .title: return .orange
        case .transition: return .purple
        case .effect: return .red
        }
    }
    
    private func applyDrag(_ deltaX: CGFloat) {
        // Calculate new position with snapping
        let pixelsPerSecond = 10 * zoomLevel
        let deltaTime = deltaX / pixelsPerSecond
        
        // Update clip position (would update model here)
        // clip.startTime += deltaTime
    }
}

// MARK: - Timeline Playhead
struct TimelinePlayhead: View {
    @Binding var position: CGFloat
    let height: CGFloat
    @Binding var isDragging: Bool
    
    public var body: some View {
        ZStack {
            // Playhead line
            Rectangle()
                .fill(Color.red)
                .frame(width: 2, height: height)
            
            // Playhead handle
            VStack {
                Path { path in
                    path.move(to: CGPoint(x: -8, y: 0))
                    path.addLine(to: CGPoint(x: 8, y: 0))
                    path.addLine(to: CGPoint(x: 0, y: 10))
                    path.closeSubpath()
                }
                .fill(Color.red)
                .frame(width: 16, height: 10)
                
                Spacer()
            }
        }
        .offset(x: position)
        .gesture(
            DragGesture()
                .onChanged { value in
                    isDragging = true
                    position = max(0, position + value.translation.width)
                }
                .onEnded { _ in
                    isDragging = false
                }
        )
    }
}

// MARK: - Timeline Controls Bar
struct TimelineControlsBar: View {
    @Binding var playheadPosition: CGFloat
    let duration: TimeInterval
    @Binding var zoomLevel: CGFloat
    
    public var body: some View {
        HStack(spacing: 16) {
            // Transport controls
            HStack(spacing: 8) {
                Button(action: {}) {
                    Image(systemName: "backward.end.fill")
                        .font(.system(size: 14))
                }
                
                Button(action: {}) {
                    Image(systemName: "backward.fill")
                        .font(.system(size: 14))
                }
                
                Button(action: {}) {
                    Image(systemName: "play.fill")
                        .font(.system(size: 16))
                }
                
                Button(action: {}) {
                    Image(systemName: "forward.fill")
                        .font(.system(size: 14))
                }
                
                Button(action: {}) {
                    Image(systemName: "forward.end.fill")
                        .font(.system(size: 14))
                }
            }
            .foregroundColor(.white)
            
            Spacer()
            
            // Track height controls
            HStack(spacing: 8) {
                Image(systemName: "square.stack.3d.up")
                    .font(.system(size: 12))
                    .foregroundColor(Color(red: 0.7, green: 0.7, blue: 0.7))
                
                Slider(value: $zoomLevel, in: 0.5...2.0)
                    .frame(width: 100)
            }
        }
        .padding(.horizontal)
    }
}

// MARK: - Helper Functions
private func convertToTimelineTracks(_ uiTracks: [UITimelineTrack]) -> [TimelineTrack] {
    return uiTracks.enumerated().map { (index, uiTrack) in
        let clips: [UITimelineClip] = uiTrack.clips.map { simple in
            UITimelineClip(id: UUID(), 
                name: simple.name,
                type: .video,
                startTime: simple.startTime,
                duration: simple.duration,
                sourceURL: simple.sourceURL,
                trackIndex: simple.trackIndex
            )
        }
        return TimelineTrack(
            id: uiTrack.id,
            name: uiTrack.name,
            type: .video,
            index: index,
            height: Double(uiTrack.height),
            isLocked: false,
            isMuted: !uiTrack.isEnabled,
            clips: clips
        )
    }
}

// MARK: - Timeline Scroll Manager
class TimelineScrollManager: ObservableObject {
    @Published var horizontalOffset: CGFloat = 0
    @Published var verticalOffset: CGFloat = 0
    @Published var visibleTimeRange: ClosedRange<TimeInterval> = 0...10
    @Published var visibleTrackRange: Range<Int> = 0..<5
    
    func scrollTo(time: TimeInterval, animated: Bool = true) {
        // Calculate pixel position for time
        // Update horizontal offset
    }
    
    func scrollTo(track: Int, animated: Bool = true) {
        // Calculate pixel position for track
        // Update vertical offset
    }
    
    func centerPlayhead() {
        // Center view on playhead position
    }
}
