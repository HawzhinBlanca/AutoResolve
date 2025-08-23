// AUTORESOLVE V3.2 - DAVINCI RESOLVE QUALITY VIDEO PLAYER
// Professional Frame-Accurate Video Player with Advanced Controls

import SwiftUI
import AVKit
import AVFoundation
import Combine

// MARK: - PROFESSIONAL VIDEO PLAYER
struct ProfessionalVideoPlayer: View {
    @EnvironmentObject var store: UnifiedStore
    @StateObject private var playerController = ProfessionalPlayerController()
    @State private var isPlaying = false
    @State private var currentTime: TimeInterval = 0
    @State private var duration: TimeInterval = 0
    @State private var viewerScale: ViewerScale = .fit
    @State private var showSafeAreas = false
    @State private var showOverlays = true
    @State private var isFullscreen = false
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // PROFESSIONAL VIDEO CANVAS
                ProfessionalVideoCanvas(
                    player: playerController.player,
                    timeline: store.timeline,
                    currentTime: $currentTime,
                    duration: $duration,
                    scale: viewerScale,
                    showSafeAreas: showSafeAreas,
                    onTimeUpdate: handleTimeUpdate
                )
                .background(Color.black)
                .cornerRadius(isFullscreen ? 0 : 8)
                
                // PROFESSIONAL OVERLAY GRAPHICS
                if showOverlays {
                    ProfessionalViewerOverlay(
                        currentTime: currentTime,
                        duration: duration,
                        scale: viewerScale,
                        showSafeAreas: showSafeAreas
                    )
                }
                
                // PROFESSIONAL TRANSPORT CONTROLS
                VStack {
                    Spacer()
                    
                    ProfessionalTransportControls(
                        isPlaying: $isPlaying,
                        currentTime: $currentTime,
                        duration: duration,
                        viewerScale: $viewerScale,
                        showSafeAreas: $showSafeAreas,
                        onPlay: { playerController.play() },
                        onPause: { playerController.pause() },
                        onSeek: { time in playerController.seek(to: time) },
                        onJumpFrame: { frames in playerController.jumpFrames(frames) }
                    )
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                    .padding()
                }
                .opacity(showOverlays ? 1 : 0)
            }
        }
        .onAppear {
            setupPlayer()
        }
        .keyboardShortcut(" ", modifiers: [], action: togglePlayPause)
        .keyboardShortcut("j", modifiers: [], action: { playerController.jumpFrames(-1) })
        .keyboardShortcut("k", modifiers: [], action: togglePlayPause)
        .keyboardShortcut("l", modifiers: [], action: { playerController.jumpFrames(1) })
        .keyboardShortcut(KeyEquivalent.leftArrow, modifiers: [], action: { playerController.jumpFrames(-1) })
        .keyboardShortcut(KeyEquivalent.rightArrow, modifiers: [], action: { playerController.jumpFrames(1) })
    }
    
    private func setupPlayer() {
        // Load the first video clip if available
        if let firstTrack = store.timeline.videoTracks.first,
           let firstClip = firstTrack.clips.first,
           let url = firstClip.sourceURL {
            playerController.loadVideo(url: url)
        }
    }
    
    private func togglePlayPause() {
        if isPlaying {
            playerController.pause()
        } else {
            playerController.play()
        }
        isPlaying.toggle()
    }
    
    private func handleTimeUpdate(_ time: TimeInterval) {
        currentTime = time
        store.timeline.setPlayhead(to: time)
    }
}

// MARK: - PROFESSIONAL VIDEO CANVAS
struct ProfessionalVideoCanvas: View {
    let player: AVPlayer?
    let timeline: TimelineModel
    @Binding var currentTime: TimeInterval
    @Binding var duration: TimeInterval
    let scale: ViewerScale
    let showSafeAreas: Bool
    let onTimeUpdate: (TimeInterval) -> Void
    
    var body: some View {
        ZStack {
            // AVPlayer View
            if let player = player {
                VideoPlayer(player: player)
                    .disabled(true) // Disable built-in controls
                    .scaleEffect(scaleValue)
                    .animation(.easeInOut(duration: 0.2), value: scale)
            } else {
                // No media placeholder
                Rectangle()
                    .fill(.black)
                    .overlay(
                        VStack(spacing: 16) {
                            Image(systemName: "video.slash")
                                .font(.system(size: 48))
                                .foregroundColor(.white.opacity(0.6))
                            
                            Text("No Media Loaded")
                                .font(.title2)
                                .foregroundColor(.white.opacity(0.8))
                            
                            Text("Import a video to begin editing")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.6))
                        }
                    )
            }
            
            // SAFE AREA OVERLAYS
            if showSafeAreas {
                SafeAreaOverlays()
            }
        }
    }
    
    private var scaleValue: CGFloat {
        switch scale {
        case .fit: return 1.0
        case .fill: return 1.2
        case .percent25: return 0.25
        case .percent50: return 0.5
        case .percent100: return 1.0
        case .percent200: return 2.0
        case .percent400: return 4.0
        }
    }
}

// MARK: - PROFESSIONAL TRANSPORT CONTROLS
struct ProfessionalTransportControls: View {
    @Binding var isPlaying: Bool
    @Binding var currentTime: TimeInterval
    let duration: TimeInterval
    @Binding var viewerScale: ViewerScale
    @Binding var showSafeAreas: Bool
    let onPlay: () -> Void
    let onPause: () -> Void
    let onSeek: (TimeInterval) -> Void
    let onJumpFrame: (Int) -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            // FRAME NAVIGATION
            HStack(spacing: 4) {
                Button(action: { onJumpFrame(-10) }) {
                    Image(systemName: "backward.frame")
                        .font(.title2)
                }
                
                Button(action: { onJumpFrame(-1) }) {
                    Image(systemName: "backward.frame.fill")
                        .font(.title3)
                }
                
                // PLAY/PAUSE
                Button(action: isPlaying ? onPause : onPlay) {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                        .font(.title)
                        .foregroundColor(.white)
                }
                .buttonStyle(.plain)
                .frame(width: 44, height: 44)
                .background(isPlaying ? .red : .green, in: Circle())
                
                Button(action: { onJumpFrame(1) }) {
                    Image(systemName: "forward.frame.fill")
                        .font(.title3)
                }
                
                Button(action: { onJumpFrame(10) }) {
                    Image(systemName: "forward.frame")
                        .font(.title2)
                }
            }
            
            Spacer()
            
            // PROFESSIONAL TIMECODE
            ProfessionalTimecodeDisplay(
                currentTime: currentTime,
                duration: duration
            )
            
            Spacer()
            
            // VIEWER CONTROLS
            HStack(spacing: 8) {
                // SCALE MENU
                Menu {
                    ForEach(ViewerScale.allCases, id: \.self) { scale in
                        Button(scale.displayName) {
                            viewerScale = scale
                        }
                    }
                } label: {
                    Text(viewerScale.displayName)
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.secondary.opacity(0.2), in: RoundedRectangle(cornerRadius: 4))
                }
                
                // SAFE AREAS TOGGLE
                Button(action: { showSafeAreas.toggle() }) {
                    Image(systemName: showSafeAreas ? "grid.circle.fill" : "grid.circle")
                        .foregroundColor(showSafeAreas ? .blue : .secondary)
                }
            }
        }
        .padding()
        .foregroundColor(.primary)
    }
}

// MARK: - PROFESSIONAL TIMECODE DISPLAY
struct ProfessionalTimecodeDisplay: View {
    let currentTime: TimeInterval
    let duration: TimeInterval
    
    var body: some View {
        VStack(spacing: 4) {
            // Current / Duration
            HStack(spacing: 8) {
                Text(timecode(currentTime))
                    .font(.system(.title3, design: .monospaced, weight: .medium))
                    .foregroundColor(.primary)
                
                Text("/")
                    .foregroundColor(.secondary)
                
                Text(timecode(duration))
                    .font(.system(.title3, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            
            // Frame number
            Text("Frame \(Int(currentTime * 30))")
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.secondary)
        }
    }
    
    private func timecode(_ time: TimeInterval) -> String {
        let hours = Int(time) / 3600
        let minutes = (Int(time) % 3600) / 60
        let seconds = Int(time) % 60
        let frames = Int((time.truncatingRemainder(dividingBy: 1)) * 30)
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, seconds, frames)
    }
}

// MARK: - VIEWER SCALE ENUM
enum ViewerScale: CaseIterable {
    case fit, fill, percent25, percent50, percent100, percent200, percent400
    
    var displayName: String {
        switch self {
        case .fit: return "Fit"
        case .fill: return "Fill" 
        case .percent25: return "25%"
        case .percent50: return "50%"
        case .percent100: return "100%"
        case .percent200: return "200%"
        case .percent400: return "400%"
        }
    }
}

// MARK: - SAFE AREA OVERLAYS
struct SafeAreaOverlays: View {
    var body: some View {
        ZStack {
            // ACTION SAFE (90%)
            Rectangle()
                .stroke(.yellow.opacity(0.8), lineWidth: 1)
                .scaleEffect(0.9)
            
            // TITLE SAFE (80%)
            Rectangle()
                .stroke(.red.opacity(0.8), lineWidth: 1)
                .scaleEffect(0.8)
            
            // CENTER CROSS
            Path { path in
                path.move(to: CGPoint(x: 0, y: 0.5))
                path.addLine(to: CGPoint(x: 1, y: 0.5))
                path.move(to: CGPoint(x: 0.5, y: 0))
                path.addLine(to: CGPoint(x: 0.5, y: 1))
            }
            .stroke(.white.opacity(0.6), lineWidth: 0.5)
            .scaleEffect(CGSize(width: 1, height: 1))
        }
    }
}

// MARK: - PROFESSIONAL VIEWER OVERLAY
struct ProfessionalViewerOverlay: View {
    let currentTime: TimeInterval
    let duration: TimeInterval
    let scale: ViewerScale
    let showSafeAreas: Bool
    
    var body: some View {
        VStack {
            // TOP OVERLAY
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("PROGRAM")
                        .font(.system(.caption, design: .monospaced, weight: .bold))
                        .foregroundColor(.white)
                    
                    Text("REC 709 • 23.98fps")
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundColor(.white.opacity(0.8))
                }
                .padding(8)
                .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 6))
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 2) {
                    Text(scale.displayName)
                        .font(.system(.caption, design: .monospaced, weight: .bold))
                        .foregroundColor(.white)
                    
                    Text(showSafeAreas ? "SAFE ON" : "SAFE OFF")
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundColor(showSafeAreas ? .yellow : .white.opacity(0.8))
                }
                .padding(8)
                .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 6))
            }
            .padding()
            
            Spacer()
        }
    }
}

// MARK: - PROFESSIONAL PLAYER CONTROLLER
class ProfessionalPlayerController: ObservableObject {
    @Published var player: AVPlayer?
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    
    private var timeObserver: Any?
    
    func loadVideo(url: URL) {
        let asset = AVAsset(url: url)
        let playerItem = AVPlayerItem(asset: asset)
        
        player = AVPlayer(playerItem: playerItem)
        
        // Setup time observation
        setupTimeObserver()
        
        // Get duration
        Task {
            do {
                let duration = try await asset.load(.duration)
                await MainActor.run {
                    self.duration = CMTimeGetSeconds(duration)
                }
            } catch {
                print("Error loading duration: \(error)")
            }
        }
    }
    
    func play() {
        player?.play()
        isPlaying = true
    }
    
    func pause() {
        player?.pause()
        isPlaying = false
    }
    
    func seek(to time: TimeInterval) {
        let cmTime = CMTime(seconds: time, preferredTimescale: 600)
        player?.seek(to: cmTime, toleranceBefore: .zero, toleranceAfter: .zero)
    }
    
    func jumpFrames(_ frames: Int) {
        let frameRate: Double = 30 // Could be dynamic based on media
        let timeIncrement = Double(frames) / frameRate
        let newTime = currentTime + timeIncrement
        seek(to: max(0, min(newTime, duration)))
    }
    
    private func setupTimeObserver() {
        guard let player = player else { return }
        
        let interval = CMTime(seconds: 0.1, preferredTimescale: 600)
        timeObserver = player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            self?.currentTime = CMTimeGetSeconds(time)
        }
    }
    
    deinit {
        if let observer = timeObserver {
            player?.removeTimeObserver(observer)
        }
    }
}