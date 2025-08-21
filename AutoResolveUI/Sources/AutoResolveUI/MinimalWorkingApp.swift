// AUTORESOLVE NEURAL TIMELINE - MINIMAL WORKING VERSION
// Simplified version to get a working build

import SwiftUI
import AVFoundation
import AVKit

// MARK: - Simple Working Timeline
struct MinimalTimeline: View {
    @EnvironmentObject private var projectStore: VideoProjectStore
    @EnvironmentObject private var timelineViewModel: TimelineViewModel
    @State private var importingMedia = false
    @State private var showExportPanel = false
    
    var body: some View {
        VSplitView {
            // Top: Video Preview
            VideoPreviewArea()
                .frame(minHeight: 200)
            
            // Bottom: Timeline
            TimelineArea()
                .frame(minHeight: 300)
        }
        .toolbar {
            ToolbarItemGroup {
                Button("New") {
                    projectStore.createNewProject()
                }
                
                Button("Import") {
                    importingMedia = true
                }
                
                Button(timelineViewModel.isPlaying ? "Pause" : "Play") {
                    timelineViewModel.isPlaying.toggle()
                }
                .keyboardShortcut(.space, modifiers: [])
                
                Button("Export") {
                    showExportPanel = true
                }
                .disabled(projectStore.currentProject == nil)
            }
        }
        .navigationTitle(projectStore.currentProject?.name ?? "AutoResolve")
        .fileImporter(
            isPresented: $importingMedia,
            allowedContentTypes: [.movie, .video, .mpeg4Movie, .quickTimeMovie],
            allowsMultipleSelection: true
        ) { result in
            handleImport(result)
        }
        .sheet(isPresented: $showExportPanel) {
            ExportPanel(isPresented: $showExportPanel)
        }
    }
    
    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            for url in urls {
                importMediaFile(url)
            }
        case .failure(let error):
            print("Import failed: \(error)")
        }
    }
    
    private func importMediaFile(_ url: URL) {
        guard var project = projectStore.currentProject else { return }
        
        // Add to media pool
        let mediaItem = MediaItem(
            name: url.lastPathComponent,
            url: url,
            type: .video,
            duration: nil,
            frameRate: nil,
            resolution: nil,
            colorSpace: nil,
            fileSize: 0,
            createdAt: Date()
        )
        
        project.mediaPool.mediaItems.append(mediaItem)
        
        // Add to timeline
        if project.timeline.videoTracks.isEmpty {
            project.timeline.videoTracks.append(VideoTrack(name: "V1"))
        }
        
        let clip = VideoClip(
            name: url.lastPathComponent,
            sourceURL: url,
            startTime: 0,
            duration: 10,
            timelineStartTime: project.timeline.duration,
            sourceStartTime: 0,
            sourceDuration: 10,
            isEnabled: true,
            volume: 1.0,
            speed: 1.0
        )
        
        project.timeline.videoTracks[0].clips.append(clip)
        project.timeline.duration += 10
        
        projectStore.currentProject = project
    }
}

// MARK: - Video Preview with Real AVPlayer
struct VideoPreviewArea: View {
    @EnvironmentObject private var projectStore: VideoProjectStore
    @EnvironmentObject private var videoPlayerViewModel: VideoPlayerViewModel
    @EnvironmentObject private var timelineViewModel: TimelineViewModel
    @State private var player: AVPlayer?
    @State private var isHovering = false
    
    var body: some View {
        ZStack {
            Rectangle()
                .fill(Color.black)
            
            if let currentProject = projectStore.currentProject,
               !currentProject.timeline.videoTracks.isEmpty,
               let firstClip = currentProject.timeline.videoTracks.first?.clips.first {
                
                // Real video player
                VideoPlayer(player: player)
                    .onAppear {
                        setupPlayer(with: firstClip.sourceURL)
                    }
                    .overlay(alignment: .bottom) {
                        // Professional playback controls
                        if isHovering {
                            PlaybackControlsOverlay(
                                isPlaying: timelineViewModel.isPlaying,
                                currentTime: videoPlayerViewModel.currentTime,
                                duration: videoPlayerViewModel.duration,
                                onPlayPause: togglePlayback,
                                onSkipBackward: skipBackward,
                                onSkipForward: skipForward
                            )
                        }
                    }
                    .onHover { hovering in
                        withAnimation(.easeInOut(duration: 0.2)) {
                            isHovering = hovering
                        }
                    }
            } else if projectStore.currentProject != nil {
                VStack(spacing: 16) {
                    Image(systemName: "film.stack")
                        .font(.system(size: 48))
                        .foregroundColor(.gray)
                    Text("Import media to begin")
                        .foregroundColor(.gray)
                    Text("⌘I or drag files here")
                        .font(.caption)
                        .foregroundColor(.gray.opacity(0.7))
                }
            } else {
                VStack(spacing: 16) {
                    Image(systemName: "video.slash")
                        .font(.system(size: 48))
                        .foregroundColor(.gray)
                    
                    Text("No Project Open")
                        .font(.title2)
                        .foregroundColor(.gray)
                    
                    Button("Create New Project") {
                        projectStore.createNewProject()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
        .cornerRadius(8)
        .padding()
        .onChange(of: timelineViewModel.isPlaying) { isPlaying in
            if isPlaying {
                player?.play()
            } else {
                player?.pause()
            }
        }
        .onChange(of: timelineViewModel.currentTime) { time in
            seekTo(time)
        }
    }
    
    private func setupPlayer(with url: URL) {
        let playerItem = AVPlayerItem(url: url)
        player = AVPlayer(playerItem: playerItem)
        videoPlayerViewModel.player = player
        
        // Observe player time for smooth scrubbing
        player?.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 1.0/30.0, preferredTimescale: 600),
            queue: .main
        ) { time in
            if !timelineViewModel.isScrubbing {
                videoPlayerViewModel.currentTime = time.seconds
                timelineViewModel.currentTime = time.seconds
            }
        }
        
        // Get duration
        Task {
            if let duration = try? await playerItem.asset.load(.duration) {
                videoPlayerViewModel.duration = CMTimeGetSeconds(duration)
                timelineViewModel.duration = CMTimeGetSeconds(duration)
            }
        }
    }
    
    private func togglePlayback() {
        timelineViewModel.isPlaying.toggle()
    }
    
    private func skipBackward() {
        let newTime = max(0, videoPlayerViewModel.currentTime - 10)
        seekTo(newTime)
    }
    
    private func skipForward() {
        let newTime = min(videoPlayerViewModel.duration, videoPlayerViewModel.currentTime + 10)
        seekTo(newTime)
    }
    
    private func seekTo(_ time: TimeInterval) {
        player?.seek(to: CMTime(seconds: time, preferredTimescale: 600)) { _ in
            videoPlayerViewModel.currentTime = time
            timelineViewModel.currentTime = time
        }
    }
}

// Playback Controls Overlay
struct PlaybackControlsOverlay: View {
    let isPlaying: Bool
    let currentTime: TimeInterval
    let duration: TimeInterval
    let onPlayPause: () -> Void
    let onSkipBackward: () -> Void
    let onSkipForward: () -> Void
    
    var body: some View {
        HStack(spacing: 20) {
            Button(action: onSkipBackward) {
                Image(systemName: "gobackward.10")
                    .foregroundColor(.white)
            }
            .buttonStyle(.plain)
            
            Button(action: onPlayPause) {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.title)
                    .foregroundColor(.white)
            }
            .buttonStyle(.plain)
            
            Button(action: onSkipForward) {
                Image(systemName: "goforward.10")
                    .foregroundColor(.white)
            }
            .buttonStyle(.plain)
            
            Spacer()
            
            // Timecode display
            Text(formatTimecode(currentTime))
                .foregroundColor(.white)
                .font(.system(.caption, design: .monospaced))
            Text("/")
                .foregroundColor(.gray)
                .font(.caption)
            Text(formatTimecode(duration))
                .foregroundColor(.gray)
                .font(.system(.caption, design: .monospaced))
        }
        .padding()
        .background(Color.black.opacity(0.8))
        .cornerRadius(8)
        .padding()
    }
    
    private func formatTimecode(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        let frames = Int((seconds - Double(Int(seconds))) * 30)
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
    }
}

// Video Player NSView wrapper
struct VideoPlayer: NSViewRepresentable {
    let player: AVPlayer?
    
    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .none
        return view
    }
    
    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        nsView.player = player
    }
}

// MARK: - Timeline Area
struct TimelineArea: View {
    @EnvironmentObject private var projectStore: VideoProjectStore
    @EnvironmentObject private var timelineViewModel: TimelineViewModel
    
    var body: some View {
        VStack(spacing: 0) {
            // Timeline Ruler
            TimelineRuler()
                .frame(height: 30)
                .background(Color(white: 0.15))
            
            // Tracks
            ScrollView {
                VStack(spacing: 2) {
                    if let project = projectStore.currentProject {
                        // Video Tracks
                        ForEach(project.timeline.videoTracks) { track in
                            SimpleTrackView(track: track, type: .video)
                                .frame(height: 60)
                        }
                        
                        // Audio Tracks
                        ForEach(project.timeline.audioTracks) { track in
                            SimpleTrackView(track: track, type: .audio)
                                .frame(height: 40)
                        }
                        
                        // Add track button
                        HStack {
                            Button("+ Video Track") {
                                addVideoTrack()
                            }
                            .buttonStyle(.borderless)
                            
                            Button("+ Audio Track") {
                                addAudioTrack()
                            }
                            .buttonStyle(.borderless)
                            
                            Spacer()
                        }
                        .padding(8)
                    } else {
                        Text("Import media to begin editing")
                            .foregroundColor(.gray)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .frame(minHeight: 200)
                    }
                }
            }
            .background(Color(white: 0.1))
        }
    }
    
    private func addVideoTrack() {
        guard var project = projectStore.currentProject else { return }
        let trackNumber = project.timeline.videoTracks.count + 1
        project.timeline.videoTracks.append(VideoTrack(name: "V\(trackNumber)"))
        projectStore.currentProject = project
    }
    
    private func addAudioTrack() {
        guard var project = projectStore.currentProject else { return }
        let trackNumber = project.timeline.audioTracks.count + 1
        project.timeline.audioTracks.append(AudioTrack(name: "A\(trackNumber)"))
        projectStore.currentProject = project
    }
}

// MARK: - Timeline Ruler
struct TimelineRuler: View {
    @EnvironmentObject private var timelineViewModel: TimelineViewModel
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Time markers
                ForEach(0..<Int(geometry.size.width / 50), id: \.self) { index in
                    VStack(spacing: 0) {
                        Rectangle()
                            .fill(Color.gray)
                            .frame(width: 1, height: 10)
                        
                        Text("\(index * 5)s")
                            .font(.system(size: 9))
                            .foregroundColor(.gray)
                    }
                    .offset(x: CGFloat(index * 50))
                }
                
                // Playhead
                Rectangle()
                    .fill(Color.red)
                    .frame(width: 2)
                    .offset(x: CGFloat(timelineViewModel.playhead * 10))
            }
        }
    }
}

// MARK: - Simple Track View
struct SimpleTrackView: View {
    let track: Any
    let type: TrackType
    
    enum TrackType {
        case video, audio
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Track background
                Rectangle()
                    .fill(type == .video ? Color.blue.opacity(0.1) : Color.green.opacity(0.1))
                
                // Track label
                HStack {
                    Text(trackName)
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                    
                    Spacer()
                }
                
                // Clips
                if type == .video, let videoTrack = track as? VideoTrack {
                    ForEach(videoTrack.clips) { clip in
                        SimpleClipView(clip: clip)
                            .offset(x: CGFloat(clip.timelineStartTime * 10))
                    }
                } else if type == .audio, let audioTrack = track as? AudioTrack {
                    ForEach(audioTrack.clips) { clip in
                        SimpleAudioClipView(clip: clip)
                            .offset(x: CGFloat(clip.timelineStartTime * 10))
                    }
                }
            }
        }
        .border(Color.gray.opacity(0.3), width: 1)
    }
    
    private var trackName: String {
        if let videoTrack = track as? VideoTrack {
            return videoTrack.name
        } else if let audioTrack = track as? AudioTrack {
            return audioTrack.name
        }
        return "Track"
    }
}

// MARK: - Simple Clip View
struct SimpleClipView: View {
    let clip: VideoClip
    
    var body: some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(LinearGradient(
                colors: [Color.blue, Color.blue.opacity(0.7)],
                startPoint: .leading,
                endPoint: .trailing
            ))
            .frame(width: CGFloat(clip.duration * 10))
            .overlay(
                Text(clip.name)
                    .font(.caption2)
                    .foregroundColor(.white)
                    .lineLimit(1)
                    .padding(4)
            )
    }
}

struct SimpleAudioClipView: View {
    let clip: AudioClip
    
    var body: some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(LinearGradient(
                colors: [Color.green, Color.green.opacity(0.7)],
                startPoint: .leading,
                endPoint: .trailing
            ))
            .frame(width: CGFloat(clip.duration * 10))
            .overlay(
                // Simple waveform representation
                HStack(spacing: 1) {
                    ForEach(0..<Int(clip.duration), id: \.self) { _ in
                        Rectangle()
                            .fill(Color.white.opacity(0.3))
                            .frame(width: 2, height: CGFloat.random(in: 10...30))
                    }
                }
            )
    }
}

// MARK: - Export Panel
struct ExportPanel: View {
    @Binding var isPresented: Bool
    @State private var exportFormat = "MP4"
    @State private var exportQuality = "High"
    @State private var exportPath = ""
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Export Project")
                .font(.title2)
                .fontWeight(.semibold)
            
            Form {
                Picker("Format", selection: $exportFormat) {
                    Text("MP4").tag("MP4")
                    Text("MOV").tag("MOV")
                    Text("ProRes").tag("ProRes")
                }
                
                Picker("Quality", selection: $exportQuality) {
                    Text("Low").tag("Low")
                    Text("Medium").tag("Medium")
                    Text("High").tag("High")
                    Text("Ultra").tag("Ultra")
                }
                
                TextField("Export Path", text: $exportPath)
                    .disabled(true)
                
                Button("Choose Location...") {
                    chooseExportLocation()
                }
            }
            .padding()
            
            HStack {
                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.escape)
                
                Spacer()
                
                Button("Export") {
                    performExport()
                }
                .keyboardShortcut(.return)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .frame(width: 400, height: 300)
    }
    
    private func chooseExportLocation() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.movie]
        panel.nameFieldStringValue = "Untitled.mp4"
        
        if panel.runModal() == .OK {
            exportPath = panel.url?.path ?? ""
        }
    }
    
    private func performExport() {
        // Implement actual export
        print("Exporting to: \(exportPath)")
        isPresented = false
    }
}

// MARK: - Enhanced Clip View with Thumbnails
struct EnhancedClipView: View {
    let clip: Any
    let type: TrackType
    @State private var isDragging = false
    @State private var isHovering = false
    @State private var thumbnailImage: NSImage?
    
    enum TrackType {
        case video, audio
    }
    
    var body: some View {
        Group {
            if type == .video, let videoClip = clip as? VideoClip {
                videoClipView(videoClip)
            } else if type == .audio, let audioClip = clip as? AudioClip {
                audioClipView(audioClip)
            }
        }
        .scaleEffect(isDragging ? 1.05 : 1.0)
        .shadow(radius: isDragging ? 8 : 0)
        .animation(.easeInOut(duration: 0.2), value: isDragging)
    }
    
    private func videoClipView(_ clip: VideoClip) -> some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(LinearGradient(
                colors: [Color.blue.opacity(0.8), Color.blue.opacity(0.6)],
                startPoint: .leading,
                endPoint: .trailing
            ))
            .frame(width: clipWidth(clip.duration), height: 56)
            .overlay(
                HStack(spacing: 2) {
                    // Thumbnail
                    if let thumbnail = thumbnailImage {
                        Image(nsImage: thumbnail)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 40, height: 40)
                            .clipped()
                            .cornerRadius(2)
                    } else {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.black.opacity(0.3))
                            .frame(width: 40, height: 40)
                            .overlay(
                                Image(systemName: "photo")
                                    .foregroundColor(.white.opacity(0.5))
                            )
                    }
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text(clip.name)
                            .font(.caption)
                            .foregroundColor(.white)
                            .lineLimit(1)
                        
                        Text("\(formatDuration(clip.duration))")
                            .font(.system(size: 9))
                            .foregroundColor(.white.opacity(0.7))
                    }
                    
                    Spacer()
                    
                    // Trim handles
                    if isHovering {
                        HStack(spacing: 0) {
                            Rectangle()
                                .fill(Color.white)
                                .frame(width: 4)
                                .cursor(.resizeLeftRight)
                            
                            Spacer()
                            
                            Rectangle()
                                .fill(Color.white)
                                .frame(width: 4)
                                .cursor(.resizeLeftRight)
                        }
                    }
                }
                .padding(4)
            )
            .onHover { hovering in
                isHovering = hovering
            }
            .onAppear {
                if let url = clip.sourceURL {
                    loadThumbnail(for: url)
                }
            }
            .draggable(clip.id.uuidString) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.blue.opacity(0.5))
                    .frame(width: clipWidth(clip.duration), height: 56)
                    .overlay(
                        Text(clip.name)
                            .font(.caption)
                            .foregroundColor(.white)
                    )
            }
    }
    
    private func audioClipView(_ clip: AudioClip) -> some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(LinearGradient(
                colors: [Color.green.opacity(0.8), Color.green.opacity(0.6)],
                startPoint: .leading,
                endPoint: .trailing
            ))
            .frame(width: clipWidth(clip.duration), height: 36)
            .overlay(
                HStack {
                    Image(systemName: "waveform")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.8))
                    
                    Text(clip.name)
                        .font(.caption)
                        .foregroundColor(.white)
                        .lineLimit(1)
                    
                    Spacer()
                }
                .padding(.horizontal, 6)
            )
            .overlay(
                // Waveform visualization
                GeometryReader { geometry in
                    Path { path in
                        let width = geometry.size.width
                        let height = geometry.size.height
                        let midY = height / 2
                        
                        for x in stride(from: 0, to: width, by: 2) {
                            let amplitude = CGFloat.random(in: 0.2...0.8)
                            path.move(to: CGPoint(x: x, y: midY - height * amplitude / 2))
                            path.addLine(to: CGPoint(x: x, y: midY + height * amplitude / 2))
                        }
                    }
                    .stroke(Color.white.opacity(0.3), lineWidth: 1)
                }
            )
    }
    
    private func clipWidth(_ duration: TimeInterval) -> CGFloat {
        return CGFloat(duration * 10) // 10 pixels per second
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        let frames = Int((duration - Double(Int(duration))) * 30)
        return String(format: "%02d:%02d.%02d", minutes, seconds, frames)
    }
    
    private func loadThumbnail(for url: URL) {
        Task {
            let asset = AVAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 80, height: 80)
            
            do {
                let cgImage = try generator.copyCGImage(at: .zero, actualTime: nil)
                await MainActor.run {
                    thumbnailImage = NSImage(cgImage: cgImage, size: NSSize(width: 40, height: 40))
                }
            } catch {
                print("Failed to generate thumbnail: \(error)")
            }
        }
    }
}

// MARK: - Cursor extension for resize
extension View {
    func cursor(_ cursor: NSCursor) -> some View {
        self.onHover { inside in
            if inside {
                cursor.push()
            } else {
                NSCursor.pop()
            }
        }
    }
}