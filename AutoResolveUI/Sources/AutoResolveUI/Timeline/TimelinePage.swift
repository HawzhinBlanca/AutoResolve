import SwiftUI
import CoreMedia

public struct TimelinePage: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    @State private var visibleRange: Range<CMTime> = CMTime.zero..<CMTime(seconds: 60, preferredTimescale: 600)
    @State private var isDraggingPlayhead = false
    @State private var dragOffset: CGFloat = 0
    
    public var body: some View {
        VStack(spacing: 0) {
            // Timeline ruler
            TimelineRuler()
                .frame(height: UITheme.Sizes.timelineRulerHeight)
                .background(UITheme.Colors.surface)
            
            // Timeline tracks with Metal renderer
            ScrollView([.horizontal, .vertical]) {
                ZStack(alignment: .topLeading) {
                    // SwiftUI-based timeline (more responsive)
                    if let timeline = appState.timeline {
                        VStack(spacing: 2) {
                            ForEach(timeline.tracks) { track in
                                TimelineTrackView(track: track, zoomLevel: appState.zoomLevel)
                                    .frame(height: UITheme.Sizes.timelineTrackHeight)
                            }
                            
                            // AI analysis lanes
                            AILaneRenderer(zoomLevel: appState.zoomLevel, timelineWidth: timelineWidth)
                                .padding(.top, UITheme.Sizes.spacingS)
                        }
                        .frame(
                            width: timelineWidth,
                            height: timelineHeight + 80 // Extra space for AI lanes
                        )
                    } else {
                        // Empty timeline placeholder
                        VStack {
                            Text("Drop video files here")
                                .foregroundColor(.gray)
                                .font(.title2)
                            Text("Drag from Finder or use Cmd+I")
                                .foregroundColor(.gray)
                                .font(.caption)
                        }
                        .frame(width: 1000, height: 300)
                        .background(Color.black.opacity(0.1))
                        .cornerRadius(8)
                    }
                    
                    // Playhead overlay
                    PlayheadOverlay()
                        .offset(x: playheadPosition)
                }
                .contentShape(Rectangle())
                .onTapGesture { location in
                    seekToTimelinePosition(location)
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            seekToTimelinePosition(value.location)
                        }
                )
            }
            .background(UITheme.Colors.background)
            .onAppear {
                updateVisibleRange()
            }
            .onChange(of: appState.scrollOffset) { _ in
                updateVisibleRange()
            }
            .onChange(of: appState.zoomLevel) { _ in
                updateVisibleRange()
            }
            // Add drag and drop support for video files
            .onDrop(of: [.fileURL, .movie, .video], isTargeted: nil) { providers in
                handleDrop(providers: providers)
            }
            
            // Timeline scrollbar
            TimelineScrollbar()
                .frame(height: UITheme.Sizes.timelineScrollbarHeight)
                .background(UITheme.Colors.surface)
        }
    }
    
    var timelineWidth: CGFloat {
        guard let timeline = appState.timeline else { return 1000 }
        let calculatedWidth = CGFloat(timeline.duration * appState.zoomLevel * 100)
        // Prevent excessive width that breaks Metal rendering
        return min(calculatedWidth, 50000)
    }
    
    var timelineHeight: CGFloat {
        guard let timeline = appState.timeline else { return 300 }
        let calculatedHeight = CGFloat(timeline.tracks.count) * UITheme.Sizes.timelineTrackHeight + 100
        // Prevent excessive height
        return min(calculatedHeight, 2000)
    }
    
    var playheadPosition: CGFloat {
        let seconds = CMTimeGetSeconds(transport.currentTime)
        return CGFloat(seconds * appState.zoomLevel * 100) - appState.scrollOffset
    }
    
    func updateVisibleRange() {
        let startTime = CMTime(seconds: Double(appState.scrollOffset) / (appState.zoomLevel * 100), preferredTimescale: 600)
        let endTime = CMTimeAdd(startTime, CMTime(seconds: 60, preferredTimescale: 600)) // 60 second window
        visibleRange = startTime..<endTime
    }
    
    func handleDrop(providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            // Handle file URLs (video files dragged from Finder)
            if provider.canLoadObject(ofClass: URL.self) {
                provider.loadObject(ofClass: URL.self) { url, error in
                    guard let url = url, error == nil else { return }
                    
                    Task { @MainActor in
                        // Calculate drop position on timeline
                        let dropTime = CMTime(seconds: Double(appState.scrollOffset) / (appState.zoomLevel * 100), preferredTimescale: 600)
                        
                        // Add video to timeline at drop position
                        appState.addVideoToTimeline(url: url, at: dropTime)
                        
                        // Optional: Start processing the video
                        if appState.autoProcessOnImport {
                            await appState.processVideo(url: url)
                        }
                    }
                }
                return true
            }
        }
        return false
    }
    
    func seekToTimelinePosition(_ location: CGPoint) {
        let pixelsPerSecond = appState.zoomLevel * 100
        let timeSeconds = (Double(location.x) + Double(appState.scrollOffset)) / pixelsPerSecond
        let seekTime = CMTime(seconds: timeSeconds, preferredTimescale: 600)
        transport.seek(to: seekTime)
    }
}

struct TimelineRuler: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                // Draw background
                context.fill(
                    Path(CGRect(origin: .zero, size: size)),
                    with: .color(UITheme.Colors.surface)
                )
                
                // Draw time markers
                let pixelsPerSecond = appState.zoomLevel * 100
                let startTime = appState.scrollOffset / pixelsPerSecond
                let visibleSeconds = size.width / pixelsPerSecond
                
                // Determine marker interval based on zoom
                let markerInterval: Double
                if appState.zoomLevel > 2 {
                    markerInterval = 1 // Every second
                } else if appState.zoomLevel > 0.5 {
                    markerInterval = 5 // Every 5 seconds
                } else {
                    markerInterval = 10 // Every 10 seconds
                }
                
                let firstMarker = ceil(startTime / markerInterval) * markerInterval
                var markerTime = firstMarker
                
                while markerTime < startTime + visibleSeconds {
                    let x = CGFloat((markerTime - startTime) * pixelsPerSecond)
                    
                    // Draw marker line
                    context.stroke(
                        Path { path in
                            path.move(to: CGPoint(x: x, y: size.height - 10))
                            path.addLine(to: CGPoint(x: x, y: size.height))
                        },
                        with: .color(UITheme.Colors.textSecondary),
                        lineWidth: 1
                    )
                    
                    // Draw time text
                    let timecode = appState.timebase.timecodeFromTime(
                        CMTime(seconds: markerTime, preferredTimescale: 600)
                    )
                    
                    context.draw(
                        Text(timecode)
                            .font(UITheme.Typography.caption)
                            .foregroundColor(UITheme.Colors.textSecondary),
                        at: CGPoint(x: x + 4, y: size.height / 2)
                    )
                    
                    markerTime += markerInterval
                }
            }
        }
    }
}

struct PlayheadOverlay: View {
    @EnvironmentObject var transport: Transport
    
    var body: some View {
        VStack(spacing: 0) {
            // Playhead handle
            Path { path in
                path.move(to: CGPoint(x: 0, y: 0))
                path.addLine(to: CGPoint(x: -6, y: -8))
                path.addLine(to: CGPoint(x: 6, y: -8))
                path.closeSubpath()
            }
            .fill(UITheme.Colors.playhead)
            .frame(width: 12, height: 8)
            
            // Playhead line
            Rectangle()
                .fill(UITheme.Colors.playhead)
                .frame(width: UITheme.Sizes.playheadWidth, height: 1000)
        }
    }
}

struct TimelineScrollbar: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background
                Rectangle()
                    .fill(UITheme.Colors.surfaceLight)
                
                // Visible area indicator
                Rectangle()
                    .fill(UITheme.Colors.selection.opacity(0.3))
                    .frame(width: visibleAreaWidth(in: geometry.size.width))
                    .offset(x: visibleAreaOffset(in: geometry.size.width))
                
                // Playhead indicator
                Rectangle()
                    .fill(UITheme.Colors.playhead)
                    .frame(width: 2)
                    .offset(x: playheadOffset(in: geometry.size.width))
            }
        }
    }
    
    func visibleAreaWidth(in totalWidth: CGFloat) -> CGFloat {
        guard let timeline = appState.timeline else { return totalWidth }
        let visibleDuration = 60.0 // 60 second window
        let totalDuration = timeline.duration
        return totalWidth * CGFloat(visibleDuration / totalDuration)
    }
    
    func visibleAreaOffset(in totalWidth: CGFloat) -> CGFloat {
        guard let timeline = appState.timeline else { return 0 }
        let scrollPercent = appState.scrollOffset / (CGFloat(timeline.duration) * appState.zoomLevel * 100)
        return totalWidth * scrollPercent
    }
    
    func playheadOffset(in totalWidth: CGFloat) -> CGFloat {
        guard let timeline = appState.timeline else { return 0 }
        let playheadPercent = CMTimeGetSeconds(transport.currentTime) / timeline.duration
        return totalWidth * CGFloat(playheadPercent)
    }
}

struct EmptyTimelinePlaceholder: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack {
            Spacer()
            
            Image(systemName: "film.stack")
                .font(.system(size: 48))
                .foregroundColor(UITheme.Colors.textDisabled)
            
            Text("Import a video to begin editing")
                .font(UITheme.Typography.headline)
                .foregroundColor(UITheme.Colors.textSecondary)
                .padding(.top)
            
            Button("Import Video") {
                appState.showImporter = true
            }
            .buttonStyle(.borderedProminent)
            .padding(.top)
            
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(UITheme.Colors.background)
    }
}