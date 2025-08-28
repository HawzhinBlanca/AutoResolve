import SwiftUI
import AVKit
import AVFoundation
import Combine

// MARK: - Professional Video Player View

public struct VideoPlayerView: View {
    @ObservedObject var timeline: TimelineModel
    @StateObject private var playerController = VideoPlayerController()
    @StateObject private var effectsProcessor = VideoEffectsProcessor()
    // Audio analysis disabled in minimal slice
    @State private var isPlaying = false
    @State private var currentTime: TimeInterval = 0
    @State private var showControls = true
    @State private var controlsTimer: Timer?
    @State private var showTimecode = true
    @State private var showWaveform = false
    @State private var waveformData: AudioWaveformGenerator.WaveformData?
    
    private let controlsFadeDelay: TimeInterval = 3.0
    private let waveformGenerator = AudioWaveformGenerator()
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Video player
                VideoPlayerRepresentable(
                    player: playerController.player,
                    timeline: timeline,
                    onTimeUpdate: { time in
                        currentTime = time
                        timeline.setPlayhead(to: time)
                    }
                )
                .aspectRatio(16/9, contentMode: .fit)
                .background(Color.black)
                .onTapGesture(count: 2) {
                    togglePlayPause()
                }
                .onTapGesture {
                    showControlsTemporarily()
                }
                
                // Overlay controls
                if showControls {
                    VStack {
                        // Top bar with timecode
                        HStack {
                            // Current timecode
                            Text(timeline.timecode(for: currentTime))
                                .font(.system(size: 14, weight: .medium, design: .monospaced))
                                .padding(8)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(4)
                            
                            Spacer()
                            
                            // Duration
                            Text(timeline.timecode(for: playerController.duration))
                                .font(.system(size: 14, weight: .medium, design: .monospaced))
                                .padding(8)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(4)
                        }
                        .padding()
                        
                        Spacer()
                        
                        // Bottom controls
                        VStack(spacing: 12) {
                            // Scrubber
                            VideoScrubber(
                                currentTime: $currentTime,
                                duration: playerController.duration,
                                onScrub: { time in
                                    playerController.seek(to: time)
                                    timeline.setPlayhead(to: time)
                                }
                            )
                            .frame(height: 6)
                            .padding(.horizontal)
                            
                            // Playback controls
                            HStack(spacing: 20) {
                                // Skip backward
                                Button(action: { skipBackward() }) {
                                    Image(systemName: "gobackward.10")
                                        .font(.title2)
                                }
                                
                                // Previous frame
                                Button(action: { previousFrame() }) {
                                    Image(systemName: "backward.frame.fill")
                                        .font(.title3)
                                }
                                
                                // Play/Pause
                                Button(action: { togglePlayPause() }) {
                                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                                        .font(.title)
                                        .frame(width: 44, height: 44)
                                }
                                
                                // Next frame
                                Button(action: { nextFrame() }) {
                                    Image(systemName: "forward.frame.fill")
                                        .font(.title3)
                                }
                                
                                // Skip forward
                                Button(action: { skipForward() }) {
                                    Image(systemName: "goforward.10")
                                        .font(.title2)
                                }
                                
                                Spacer()
                                
                                // Volume control
                                VolumeControl(volume: $playerController.volume)
                                    .frame(width: 100)
                                
                                // Fullscreen toggle
                                Button(action: { toggleFullscreen() }) {
                                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                                        .font(.title3)
                                }
                            }
                            .padding(.horizontal)
                            .foregroundColor(.white)
                        }
                        .padding()
                        .background(
                            LinearGradient(
                                colors: [Color.clear, Color.black.opacity(0.7)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                    }
                    .transition(.opacity)
                }
                
                // Performance overlay
                if playerController.showPerformanceStats {
                    PerformanceOverlay(playerController: playerController)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                        .padding()
                }
                
                // Timecode overlay
                if showTimecode {
                    TimecodeOverlay(timeline: timeline, currentTime: currentTime)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                        .padding()
                }
                
                // Audio waveform
                if showWaveform, let waveformData = waveformData {
                    VStack {
                        Spacer()
                        WaveformView(
                            waveformData: waveformData,
                            color: .green,
                            backgroundColor: Color.black.opacity(0.3)
                        )
                        .frame(height: 60)
                        .padding(.horizontal)
                        .padding(.bottom, 100)
                    }
                }
                
                // Effects preview panel
                if effectsProcessor.hasActiveEffects {
                    EffectsPreviewPanel(effectsProcessor: effectsProcessor)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .padding()
                }
            }
        }
        .onAppear {
            let sp = Performance.begin("SetupPlayer")
            setupPlayer()
            Performance.end(sp, "SetupPlayer")
        }
        .onDisappear {
            cleanupPlayer()
        }
        .onReceive(playerController.$isPlaying) { playing in
            isPlaying = playing
        }
        .onReceive(playerController.$currentTime) { time in
            currentTime = time
            timeline.setPlayhead(to: time)
        }
    }
    
    // MARK: - Player Setup
    
    private func setupPlayer() {
        // Load video based on timeline clips
        if let firstClip = timeline.tracks.flatMap({ $0.clips }).first,
           let url = firstClip.sourceURL {
            playerController.loadVideo(url: url)
            
            // Generate waveform
            waveformGenerator.generateWaveform(from: url) { result in
                switch result {
                case .success(let data):
                    waveformData = data
                case .failure(let error):
                    print("Failed to generate waveform: \(error)")
                }
            }
            
            // Audio analysis disabled
        }
        
        // Setup effects processor
        effectsProcessor.attachToPlayer(playerController.player)
        
        // Sync with timeline
        playerController.seek(to: timeline.playheadPosition)
        
        showControlsTemporarily()
    }
    
    private func cleanupPlayer() {
        playerController.pause()
        controlsTimer?.invalidate()
        // audioAnalyzer.stopAnalyzing() // This line is removed as per the edit hint.
        effectsProcessor.detachFromPlayer()
    }
    
    // MARK: - Playback Controls
    
    private func togglePlayPause() {
        if isPlaying {
            playerController.pause()
        } else {
            playerController.play()
        }
        showControlsTemporarily()
    }
    
    private func skipBackward() {
        playerController.skip(by: -10)
        showControlsTemporarily()
    }
    
    private func skipForward() {
        playerController.skip(by: 10)
        showControlsTemporarily()
    }
    
    private func previousFrame() {
        playerController.stepByFrame(forward: false)
        showControlsTemporarily()
    }
    
    private func nextFrame() {
        playerController.stepByFrame(forward: true)
        showControlsTemporarily()
    }
    
    private func toggleFullscreen() {
        // Implementation depends on platform
        showControlsTemporarily()
    }
    
    private func showControlsTemporarily() {
        withAnimation(.easeInOut(duration: 0.3)) {
            showControls = true
        }
        
        controlsTimer?.invalidate()
        controlsTimer = Timer.scheduledTimer(withTimeInterval: controlsFadeDelay, repeats: false) { _ in
            if isPlaying {
                withAnimation(.easeInOut(duration: 0.3)) {
                    showControls = false
                }
            }
        }
    }
}

// MARK: - Video Player Controller

class VideoPlayerController: ObservableObject {
    @Published var player: AVPlayer
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var volume: Float = 1.0
    @Published var isBuffering = false
    @Published var showPerformanceStats = false
    
    // Performance metrics
    @Published var fps: Double = 0
    @Published var droppedFrames: Int = 0
    @Published var bitrate: Double = 0
    
    private var timeObserver: Any?
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        self.player = AVPlayer()
        setupObservers()
    }
    
    deinit {
        if let observer = timeObserver {
            player.removeTimeObserver(observer)
        }
    }
    
    private func setupObservers() {
        // Time observer for playback position
        let interval = CMTime(seconds: 1/30.0, preferredTimescale: CMTimeScale(NSEC_PER_SEC))
        timeObserver = player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            self?.currentTime = time.seconds
        }
        
        // Playing state observer
        player.publisher(for: \.timeControlStatus)
            .sink { [weak self] status in
                self?.isPlaying = status == .playing
                self?.isBuffering = status == .waitingToPlayAtSpecifiedRate
            }
            .store(in: &cancellables)
        
        // Volume observer
        $volume
            .sink { [weak self] volume in
                self?.player.volume = volume
            }
            .store(in: &cancellables)
    }
    
    func loadVideo(url: URL) {
        let asset = AVAsset(url: url)
        let playerItem = AVPlayerItem(asset: asset)
        
        // Add observers for the player item
        playerItem.publisher(for: \.status)
            .sink { [weak self] status in
                if status == .readyToPlay {
                    self?.duration = playerItem.duration.seconds
                }
            }
            .store(in: &cancellables)
        
        player.replaceCurrentItem(with: playerItem)
    }
    
    func play() {
        player.play()
    }
    
    func pause() {
        player.pause()
    }
    
    func seek(to time: TimeInterval) {
        let cmTime = CMTime(seconds: time, preferredTimescale: CMTimeScale(NSEC_PER_SEC))
        player.seek(to: cmTime, toleranceBefore: .zero, toleranceAfter: .zero)
    }
    
    func skip(by seconds: TimeInterval) {
        let newTime = currentTime + seconds
        seek(to: max(0, min(newTime, duration)))
    }
    
    func stepByFrame(forward: Bool) {
        if forward {
            player.currentItem?.step(byCount: 1)
        } else {
            player.currentItem?.step(byCount: -1)
        }
    }
    
    func setPlaybackSpeed(_ speed: Float) {
        player.rate = speed
    }
}

// MARK: - Video Player Representable (NSViewRepresentable)

struct VideoPlayerRepresentable: NSViewRepresentable {
    let player: AVPlayer
    let timeline: TimelineModel
    let onTimeUpdate: (TimeInterval) -> Void
    
    func makeNSView(context: Context) -> AVPlayerView {
        let playerView = AVPlayerView()
        playerView.player = player
        playerView.controlsStyle = .none
        playerView.showsFullScreenToggleButton = false
        return playerView
    }
    
    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        // Update if needed
    }
}

// MARK: - Video Scrubber

struct VideoScrubber: View {
    @Binding var currentTime: TimeInterval
    let duration: TimeInterval
    let onScrub: (TimeInterval) -> Void
    
    @State private var isDragging = false
    @State private var dragTime: TimeInterval = 0
    @State private var lastDispatchTime: CFAbsoluteTime = 0
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Track
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.white.opacity(0.3))
                
                // Progress
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.accentColor)
                    .frame(width: progressWidth(in: geometry.size.width))
                
                // Thumb
                Circle()
                    .fill(Color.white)
                    .frame(width: 12, height: 12)
                    .shadow(radius: 2)
                    .offset(x: thumbOffset(in: geometry.size.width))
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        isDragging = true
                        let progress = value.location.x / geometry.size.width
                        dragTime = duration * Double(max(0, min(1, progress)))
                        let now = CFAbsoluteTimeGetCurrent()
                        if now - lastDispatchTime >= (1.0/60.0) {
                            lastDispatchTime = now
                            onScrub(dragTime)
                        }
                    }
                    .onEnded { _ in
                        isDragging = false
                        onScrub(dragTime)
                    }
            )
        }
    }
    
    private func progressWidth(in totalWidth: CGFloat) -> CGFloat {
        guard duration > 0 else { return 0 }
        let progress = isDragging ? dragTime / duration : currentTime / duration
        return totalWidth * CGFloat(progress)
    }
    
    private func thumbOffset(in totalWidth: CGFloat) -> CGFloat {
        guard duration > 0 else { return -6 }
        let progress = isDragging ? dragTime / duration : currentTime / duration
        return totalWidth * CGFloat(progress) - 6
    }
}

// MARK: - Volume Control

struct VolumeControl: View {
    @Binding var volume: Float
    
    public var body: some View {
        HStack(spacing: 8) {
            Image(systemName: volumeIcon)
                .font(.system(size: 14))
                .frame(width: 20)
            
            Slider(value: $volume, in: 0...1)
                .accentColor(.white)
        }
        .foregroundColor(.white)
    }
    
    private var volumeIcon: String {
        if volume == 0 {
            return "speaker.slash.fill"
        } else if volume < 0.33 {
            return "speaker.fill"
        } else if volume < 0.66 {
            return "speaker.wave.1.fill"
        } else {
            return "speaker.wave.3.fill"
        }
    }
}

// MARK: - Performance Overlay

struct PerformanceOverlay: View {
    @ObservedObject var playerController: VideoPlayerController
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Performance")
                .font(.caption.bold())
            
            Text("FPS: \(String(format: "%.1f", playerController.fps))")
                .font(.caption)
            
            Text("Dropped: \(playerController.droppedFrames)")
                .font(.caption)
            
            Text("Bitrate: \(String(format: "%.1f Mbps", playerController.bitrate / 1_000_000))")
                .font(.caption)
            
            Text("Buffer: \(playerController.isBuffering ? "Loading..." : "Ready")")
                .font(.caption)
                .foregroundColor(playerController.isBuffering ? .yellow : .green)
        }
        .padding(8)
        .background(Color.black.opacity(0.7))
        .cornerRadius(6)
        .foregroundColor(.white)
    }
}
