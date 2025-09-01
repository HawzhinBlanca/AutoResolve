import SwiftUI
import AVKit
import AppKit
import Combine

@main
struct AutoResolvePro: App {
    var body: some Scene {
        WindowGroup {
            MainWindow()
                .frame(minWidth: 1600, minHeight: 900)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentMinSize)
    }
}

struct MainWindow: View {
    @StateObject private var model = AutoResolveModel()
    
    var body: some View {
        VSplitView {
            // Top area - viewers and controls
            HSplitView {
                // Left - Media pool & browser
                MediaPoolView()
                    .frame(minWidth: 250, idealWidth: 300)
                
                // Center - Viewers
                ViewerSection()
                    .frame(minWidth: 600)
                
                // Right - Inspector
                InspectorView()
                    .frame(minWidth: 250, idealWidth: 300)
            }
            .frame(minHeight: 400)
            
            // Bottom - Timeline
            TimelineSection()
                .frame(minHeight: 300, idealHeight: 400)
        }
        .environmentObject(model)
        .onAppear {
            model.connectToBackend()
        }
    }
}

@MainActor
class AutoResolveModel: ObservableObject {
    @Published var currentVideo: URL?
    @Published var player: AVPlayer?
    @Published var currentTime = CMTime.zero
    @Published var duration = CMTime.zero
    @Published var isPlaying = false
    
    @Published var clips: [TimelineClip] = []
    @Published var selectedClip: TimelineClip?
    @Published var playheadPosition: Double = 0
    
    @Published var statusMessage = "Ready"
    @Published var isProcessing = false
    @Published var backendConnected = false
    private let backend = BackendClient()
    
    @Published var silenceSegments: [SilenceSegment] = []
    @Published var transcriptionSegments: [TranscriptionSegment] = []
    @Published var storyBeats: [StoryBeat] = []
    
    private var timeObserver: Any?
    private var cancellables = Set<AnyCancellable>()
    private var healthCheckTask: Task<Void, Never>?
    private var connectionPool: URLSession
    
    init() {
        // Configure connection pooling
        let config = URLSessionConfiguration.default
        config.httpMaximumConnectionsPerHost = 5
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        connectionPool = URLSession(configuration: config)
    }
    
    deinit {
        cleanup()
    }
    
    func cleanup() {
        // Cancel health check task
        healthCheckTask?.cancel()
        healthCheckTask = nil
        
        // Remove time observer
        if let observer = timeObserver {
            player?.removeTimeObserver(observer)
            timeObserver = nil
        }
        
        // Cancel all subscriptions
        cancellables.forEach { $0.cancel() }
        cancellables.removeAll()
    }
    
    func connectToBackend() {
        // Cancel previous task if exists
        healthCheckTask?.cancel()
        
        // Start new health check with exponential backoff
        healthCheckTask = Task { [weak self] in
            var retryDelay: UInt64 = 1_000_000_000 // 1 second in nanoseconds
            let maxDelay: UInt64 = 30_000_000_000 // 30 seconds max
            
            while !Task.isCancelled {
                await self?.checkBackendHealth()
                
                // Exponential backoff if not connected
                if self?.backendConnected == false {
                    retryDelay = min(retryDelay * 2, maxDelay)
                } else {
                    retryDelay = 5_000_000_000 // 5 seconds when connected
                }
                
                try? await Task.sleep(nanoseconds: retryDelay)
            }
        }
    }
    
    func checkBackendHealth() async {
        let ok = (try? await backend.checkHealth()) == true
        await MainActor.run {
            self.backendConnected = ok
        }
    }
    
    func importVideo(url: URL) {
        // Clean up previous observer FIRST
        if let observer = timeObserver {
            player?.removeTimeObserver(observer)
            timeObserver = nil
        }
        
        currentVideo = url
        player = AVPlayer(url: url)
        
        if let item = player?.currentItem {
            duration = item.duration
        }
        
        // Add to timeline
        let clip = TimelineClip(
            id: UUID(),
            name: url.lastPathComponent,
            url: url,
            startTime: 0,
            duration: CMTimeGetSeconds(duration)
        )
        clips.append(clip)
        
        // Start NEW time observer
        timeObserver = player?.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.03, preferredTimescale: 600),
            queue: .main
        ) { [weak self] time in
            self?.currentTime = time
            if let duration = self?.duration, duration.seconds > 0 {
                self?.playheadPosition = time.seconds / duration.seconds
            }
        }
        
        statusMessage = "Imported: \(url.lastPathComponent)"
    }
    
    func playPause() {
        if isPlaying {
            player?.pause()
        } else {
            player?.play()
        }
        isPlaying.toggle()
    }
    
    func seekToPosition(_ position: Double) {
        let time = CMTime(seconds: position * duration.seconds, preferredTimescale: 600)
        player?.seek(to: time)
        playheadPosition = position
    }
    
    // AI Processing
    func detectSilence() async {
        guard let videoPath = currentVideo?.path else { return }
        isProcessing = true
        statusMessage = "Detecting silence..."
        
        do {
            let res = try await backend.analyzeSilence(path: videoPath)
            silenceSegments = res.ranges.map { SilenceSegment(start: $0.s, end: $0.e) }
            statusMessage = "Found \(silenceSegments.count) silence segments"
        } catch {
            statusMessage = "Silence detection failed"
        }
        
        isProcessing = false
    }
    
    func transcribe() async {
        guard let videoPath = currentVideo?.path else { return }
        isProcessing = true
        statusMessage = "Transcribing..."
        
        do {
            let asr = try await backend.asr(path: videoPath, lang: "en")
            transcriptionSegments = asr.words.map { TranscriptionSegment(start: $0.t0, end: $0.t1, text: $0.text) }
            statusMessage = "Transcribed \(transcriptionSegments.count) segments"
        } catch {
            statusMessage = "Transcription failed"
        }
        
        isProcessing = false
    }
    
    func analyzeStoryBeats() async {
        guard let videoPath = currentVideo?.path else { return }
        isProcessing = true
        statusMessage = "Analyzing story beats..."
        
        do {
            var request = URLRequest(url: URL(string: "http://localhost:8000/api/story-beats")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let body = ["video_path": videoPath]
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            
            let (data, _) = try await URLSession.shared.data(for: request)
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let beats = json["beats"] as? [[String: Any]] {
                storyBeats = beats.compactMap { dict in
                    guard let time = dict["time"] as? Double,
                          let type = dict["type"] as? String,
                          let description = dict["description"] as? String else { return nil }
                    return StoryBeat(time: time, type: type, description: description)
                }
                statusMessage = "Found \(storyBeats.count) story beats"
            }
        } catch {
            statusMessage = "Story analysis failed"
        }
        
        isProcessing = false
    }
    
    func cutAtPlayhead() {
        let cutTime = CMTimeGetSeconds(currentTime)
        var newClips: [TimelineClip] = []
        
        for clip in clips {
            if cutTime > clip.startTime && cutTime < clip.startTime + clip.duration {
                // Split this clip
                let leftClip = TimelineClip(
                    id: UUID(),
                    name: clip.name + " (1)",
                    url: clip.url,
                    startTime: clip.startTime,
                    duration: cutTime - clip.startTime
                )
                let rightClip = TimelineClip(
                    id: UUID(),
                    name: clip.name + " (2)",
                    url: clip.url,
                    startTime: cutTime,
                    duration: clip.startTime + clip.duration - cutTime
                )
                newClips.append(leftClip)
                newClips.append(rightClip)
            } else {
                newClips.append(clip)
            }
        }
        
        clips = newClips
        statusMessage = "Cut at \(formatTime(cutTime))"
    }
    
    func formatTime(_ seconds: Double) -> String {
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        let frames = Int((seconds - Double(Int(seconds))) * 30)
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
    }
}

// Data structures
struct TimelineClip: Identifiable {
    let id: UUID
    var name: String
    var url: URL
    var startTime: Double
    var duration: Double
}

struct SilenceSegment {
    let start: Double
    let end: Double
}

struct TranscriptionSegment {
    let start: Double
    let end: Double
    let text: String
}

struct StoryBeat {
    let time: Double
    let type: String
    let description: String
}

// UI Components
struct MediaPoolView: View {
    @EnvironmentObject var model: AutoResolveModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Media Pool")
                .font(.headline)
                .padding()
            
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    if let video = model.currentVideo {
                        MediaItem(url: video)
                    }
                    
                    Button("Import Video...") {
                        let panel = NSOpenPanel()
                        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
                        if panel.runModal() == .OK, let url = panel.url {
                            model.importVideo(url: url)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .padding()
                }
            }
            
            Spacer()
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct MediaItem: View {
    let url: URL
    
    var body: some View {
        HStack {
            Image(systemName: "video.fill")
                .foregroundColor(.accentColor)
            VStack(alignment: .leading) {
                Text(url.lastPathComponent)
                    .font(.caption)
                Text(url.path)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 5)
    }
}

struct ViewerSection: View {
    @EnvironmentObject var model: AutoResolveModel
    
    var body: some View {
        VStack(spacing: 0) {
            // Viewer
            if let player = model.player {
                VideoPlayer(player: player)
                    .background(Color.black)
            } else {
                Rectangle()
                    .fill(Color.black)
                    .overlay(
                        VStack {
                            Image(systemName: "video.badge.plus")
                                .font(.largeTitle)
                                .foregroundColor(.gray)
                            Text("Import video to begin")
                                .foregroundColor(.gray)
                        }
                    )
            }
            
            // Transport controls
            TransportControls()
                .padding()
                .background(Color(NSColor.controlBackgroundColor))
        }
    }
}

struct TransportControls: View {
    @EnvironmentObject var model: AutoResolveModel
    
    var body: some View {
        VStack(spacing: 10) {
            // Timecode and progress
            HStack {
                Text(model.formatTime(CMTimeGetSeconds(model.currentTime)))
                    .font(.system(.body, design: .monospaced))
                
                Slider(value: $model.playheadPosition, in: 0...1) { editing in
                    if !editing {
                        model.seekToPosition(model.playheadPosition)
                    }
                }
                
                Text(model.formatTime(CMTimeGetSeconds(model.duration)))
                    .font(.system(.body, design: .monospaced))
            }
            
            // Playback controls
            HStack(spacing: 20) {
                Button(action: { model.seekToPosition(0) }) {
                    Image(systemName: "backward.end.fill")
                }
                
                Button(action: { model.playPause() }) {
                    Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title2)
                }
                .buttonStyle(.borderedProminent)
                
                Button(action: { model.seekToPosition(1) }) {
                    Image(systemName: "forward.end.fill")
                }
                
                Divider()
                    .frame(height: 20)
                
                Button("Cut") {
                    model.cutAtPlayhead()
                }
                .keyboardShortcut("b", modifiers: [])
            }
        }
    }
}

struct InspectorView: View {
    @EnvironmentObject var model: AutoResolveModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Inspector")
                .font(.headline)
                .padding()
            
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // AI Tools
                    GroupBox("AI Analysis") {
                        VStack(alignment: .leading, spacing: 10) {
                            Button("Detect Silence") {
                                Task { await model.detectSilence() }
                            }
                            .frame(maxWidth: .infinity)
                            
                            Button("Transcribe") {
                                Task { await model.transcribe() }
                            }
                            .frame(maxWidth: .infinity)
                            
                            Button("Analyze Story") {
                                Task { await model.analyzeStoryBeats() }
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                    }
                    
                    // Results
                    if !model.silenceSegments.isEmpty {
                        GroupBox("Silence: \(model.silenceSegments.count) segments") {
                            ForEach(Array(model.silenceSegments.prefix(5).enumerated()), id: \.offset) { i, segment in
                                Text("\(model.formatTime(segment.start)) - \(model.formatTime(segment.end))")
                                    .font(.caption)
                            }
                        }
                    }
                    
                    if !model.transcriptionSegments.isEmpty {
                        GroupBox("Transcription: \(model.transcriptionSegments.count) segments") {
                            ForEach(Array(model.transcriptionSegments.prefix(3).enumerated()), id: \.offset) { i, segment in
                                VStack(alignment: .leading) {
                                    Text(segment.text)
                                        .font(.caption)
                                        .lineLimit(2)
                                    Text("\(model.formatTime(segment.start))")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                    
                    // Status
                    GroupBox("Status") {
                        HStack {
                            Circle()
                                .fill(model.backendConnected ? Color.green : Color.red)
                                .frame(width: 8, height: 8)
                            Text(model.backendConnected ? "Backend Connected" : "Backend Offline")
                                .font(.caption)
                        }
                        
                        if model.isProcessing {
                            ProgressView()
                                .scaleEffect(0.7)
                        }
                        
                        Text(model.statusMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding()
            }
            
            Spacer()
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct TimelineSection: View {
    @EnvironmentObject var model: AutoResolveModel
    
    var body: some View {
        VStack(spacing: 0) {
            // Timeline header
            HStack {
                Text("Timeline")
                    .font(.headline)
                Spacer()
                Text("Clips: \(model.clips.count)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(Color(NSColor.controlBackgroundColor))
            
            // Timeline tracks
            ScrollView([.horizontal, .vertical]) {
                VStack(alignment: .leading, spacing: 2) {
                    // Video track
                    HStack(spacing: 2) {
                        ForEach(model.clips) { clip in
                            TimelineClipView(clip: clip)
                        }
                    }
                    .frame(height: 60)
                    .background(Color.black.opacity(0.3))
                    
                    // Silence track
                    if !model.silenceSegments.isEmpty {
                        HStack(spacing: 0) {
                            ForEach(Array(model.silenceSegments.enumerated()), id: \.offset) { _, segment in
                                SilenceSegmentView(segment: segment, totalDuration: model.duration.seconds)
                            }
                        }
                        .frame(height: 30)
                        .background(Color.red.opacity(0.1))
                    }
                    
                    // Transcription track  
                    if !model.transcriptionSegments.isEmpty {
                        HStack(spacing: 0) {
                            ForEach(Array(model.transcriptionSegments.enumerated()), id: \.offset) { _, segment in
                                TranscriptionSegmentView(segment: segment, totalDuration: model.duration.seconds)
                            }
                        }
                        .frame(height: 30)
                        .background(Color.blue.opacity(0.1))
                    }
                }
                .padding()
            }
            .background(Color(NSColor.windowBackgroundColor))
            
            // Playhead overlay
            GeometryReader { geometry in
                Rectangle()
                    .fill(Color.red)
                    .frame(width: 2)
                    .offset(x: geometry.size.width * model.playheadPosition)
            }
            .frame(height: 2)
        }
    }
}

struct TimelineClipView: View {
    let clip: TimelineClip
    
    var body: some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(Color.accentColor)
            .overlay(
                Text(clip.name)
                    .font(.caption)
                    .foregroundColor(.white)
                    .lineLimit(1)
                    .padding(4)
            )
            .frame(width: max(50, clip.duration * 10))
    }
}

struct SilenceSegmentView: View {
    let segment: SilenceSegment
    let totalDuration: Double
    
    var body: some View {
        Rectangle()
            .fill(Color.red.opacity(0.5))
            .frame(width: ((segment.end - segment.start) / totalDuration) * 800)
            .offset(x: (segment.start / totalDuration) * 800)
    }
}

struct TranscriptionSegmentView: View {
    let segment: TranscriptionSegment
    let totalDuration: Double
    
    var body: some View {
        Rectangle()
            .fill(Color.blue.opacity(0.5))
            .frame(width: ((segment.end - segment.start) / totalDuration) * 800)
            .offset(x: (segment.start / totalDuration) * 800)
            .overlay(
                Text(segment.text)
                    .font(.caption2)
                    .foregroundColor(.white)
                    .lineLimit(1)
                    .padding(2)
            )
    }
}