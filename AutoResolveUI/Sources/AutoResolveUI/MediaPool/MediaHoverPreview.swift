import AppKit
// AUTORESOLVE V3.0 - MEDIA HOVER PREVIEW
import AVFoundation
// Professional hover preview with frame scrubbing

import SwiftUI
import AVKit
import Combine

// MARK: - Color Extensions
// Use shared theme colors from Theme/Colors.swift

// MARK: - Hover Preview View
struct MediaHoverPreview: View {
    let item: MediaPoolItem
    @State private var isHovering = false
    @State private var currentTime: Double = 0
    @State private var player: AVPlayer?
    @State private var playerItem: AVPlayerItem?
    @State private var frameImages: [NSImage] = []
    @State private var currentFrameIndex = 0
    @State private var isLoadingFrames = false
    @State private var hoverLocation: CGPoint = .zero
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Base thumbnail
                if let thumbnail = item.thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .clipped()
                } else {
                    Rectangle()
                        .fill(Color.davinciDark)
                }
                
                // Scrubbing overlay
                if isHovering && !frameImages.isEmpty {
                    Image(nsImage: frameImages[currentFrameIndex])
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .clipped()
                        .transition(.opacity)
                    
                    // Progress bar
                    VStack {
                        Spacer()
                        
                        ZStack(alignment: .leading) {
                            Rectangle()
                                .fill(Color.black.opacity(0.5))
                                .frame(height: 4)
                            
                            Rectangle()
                                .fill(Color.davinciHighlight)
                                .frame(width: geometry.size.width * scrubProgress, height: 4)
                        }
                        .cornerRadius(2)
                        .padding(.horizontal, 8)
                        .padding(.bottom, 8)
                    }
                }
                
                // Loading indicator
                if isLoadingFrames {
                    ProgressView()
                        .scaleEffect(0.5)
                        .frame(width: 30, height: 30)
                        .background(Color.black.opacity(0.5))
                        .cornerRadius(4)
                }
                
                // Play button overlay
                if isHovering && !isLoadingFrames {
                    Image(systemName: "play.circle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.white)
                        .shadow(color: .black.opacity(0.5), radius: 4)
                        .opacity(hoverLocation.y < geometry.size.height * 0.7 ? 1 : 0)
                        .animation(.easeInOut(duration: 0.2), value: hoverLocation)
                }
            }
            .onHover { hovering in
                if hovering != isHovering {
                    isHovering = hovering
                    if hovering {
                        startPreview()
                    } else {
                        stopPreview()
                    }
                }
            }
            .onContinuousHover { phase in
                switch phase {
                case .active(let location):
                    hoverLocation = location
                    updateScrubPosition(location.x / geometry.size.width)
                case .ended:
                    hoverLocation = .zero
                }
            }
        }
    }
    
    var scrubProgress: CGFloat {
        guard !frameImages.isEmpty else { return 0 }
        return CGFloat(currentFrameIndex) / CGFloat(frameImages.count - 1)
    }
    
    // MARK: - Preview Control
    private func startPreview() {
        guard frameImages.isEmpty else { return }
        
        Task {
            await loadFrames()
        }
    }
    
    private func stopPreview() {
        currentFrameIndex = 0
        player?.pause()
        player = nil
        playerItem = nil
    }
    
    private func updateScrubPosition(_ normalizedX: CGFloat) {
        guard !frameImages.isEmpty else { return }
        
        let index = Int(normalizedX * CGFloat(frameImages.count - 1))
        currentFrameIndex = max(0, min(frameImages.count - 1, index))
    }
    
    // MARK: - Frame Loading
    @MainActor
    private func loadFrames() async {
        guard item.hasVideo, frameImages.isEmpty else { return }
        
        isLoadingFrames = true
        defer { isLoadingFrames = false }
        
        let asset = AVAsset(url: item.url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 320, height: 180)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        
        // Generate 10 frames across the video
        let frameCount = 10
        var times: [NSValue] = []
        
        for i in 0..<frameCount {
            let progress = Double(i) / Double(frameCount - 1)
            let time = CMTime(seconds: item.duration ?? 0 * progress, preferredTimescale: 600)
            times.append(NSValue(time: time))
        }
        
        var images: [NSImage] = []
        
        for timeValue in times {
            do {
                let time = timeValue.timeValue
                let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
                let nsImage = NSImage(cgImage: cgImage, size: CGSize(width: 320, height: 180))
                images.append(nsImage)
            } catch {
                // Use thumbnail as fallback
                if let thumbnail = item.thumbnail {
                    images.append(thumbnail)
                }
            }
        }
        
        frameImages = images
    }
}

// MARK: - Interactive Scrubber
struct MediaScrubber: View {
    let item: MediaPoolItem
    @Binding var currentTime: Double
    @State private var isDragging = false
    @State private var hoverTime: Double?
    @State private var thumbnailCache: [Double: NSImage] = [:]
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background track
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.davinciDark)
                    .frame(height: 6)
                
                // Progress
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.davinciHighlight)
                    .frame(width: geometry.size.width * progress, height: 6)
                
                // Scrubber handle
                Circle()
                    .fill(Color.white)
                    .frame(width: 14, height: 14)
                    .shadow(radius: 2)
                    .offset(x: geometry.size.width * progress - 7)
                
                // Hover preview
                if let hoverTime = hoverTime {
                    VStack(spacing: 4) {
                        // Thumbnail preview
                        if let thumbnail = thumbnailCache[hoverTime] {
                            Image(nsImage: thumbnail)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 120, height: 68)
                                .clipped()
                                .cornerRadius(4)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 4)
                                        .stroke(Color.white.opacity(0.5), lineWidth: 1)
                                )
                        } else {
                            Rectangle()
                                .fill(Color.davinciPanel)
                                .frame(width: 120, height: 68)
                                .cornerRadius(4)
                                .overlay(
                                    ProgressView()
                                        .scaleEffect(0.3)
                                )
                        }
                        
                        // Time label
                        Text(formatTime(hoverTime))
                            .font(.caption2.monospacedDigit())
                            .foregroundColor(.white)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(3)
                    }
                    .offset(x: min(max(0, geometry.size.width * (hoverTime / item.duration ?? 0) - 60), geometry.size.width - 120))
                    .offset(y: -90)
                    .transition(.opacity.combined(with: .scale))
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        isDragging = true
                        let progress = max(0, min(1, value.location.x / geometry.size.width))
                        currentTime = item.duration ?? 0 * Double(progress)
                    }
                    .onEnded { _ in
                        isDragging = false
                    }
            )
            .onHover { hovering in
                if !hovering {
                    hoverTime = nil
                }
            }
            .onContinuousHover { phase in
                switch phase {
                case .active(let location):
                    let progress = max(0, min(1, location.x / geometry.size.width))
                    let time = item.duration ?? 0 * Double(progress)
                    hoverTime = time
                    
                    // Load thumbnail for hover time
                    if thumbnailCache[time] == nil {
                        Task {
                            await loadThumbnail(at: time)
                        }
                    }
                case .ended:
                    hoverTime = nil
                }
            }
        }
        .frame(height: 6)
        .animation(.interactiveSpring(), value: currentTime)
    }
    
    var progress: CGFloat {
        guard item.duration ?? 0 > 0 else { return 0 }
        return CGFloat(currentTime / item.duration ?? 0)
    }
    
    private func formatTime(_ seconds: Double) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.unitsStyle = .positional
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: seconds) ?? "0:00"
    }
    
    @MainActor
    private func loadThumbnail(at time: Double) async {
        let asset = AVAsset(url: item.url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 240, height: 136)
        
        let cmTime = CMTime(seconds: time, preferredTimescale: 600)
        
        do {
            let cgImage = try generator.copyCGImage(at: cmTime, actualTime: nil)
            let nsImage = NSImage(cgImage: cgImage, size: CGSize(width: 240, height: 136))
            thumbnailCache[time] = nsImage
        } catch {
            print("Failed to generate thumbnail at \(time): \(error)")
        }
    }
}

// MARK: - Filmstrip Scrubber
struct FilmstripScrubber: View {
    let item: MediaPoolItem
    @Binding var currentTime: Double
    @State private var filmstripFrames: [NSImage] = []
    @State private var isLoadingFrames = false
    
    public var body: some View {
        ZStack(alignment: .leading) {
            // Filmstrip background
            if !filmstripFrames.isEmpty {
                HStack(spacing: 1) {
                    ForEach(0..<filmstripFrames.count, id: \.self) { index in
                        Image(nsImage: filmstripFrames[index])
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 40, height: 24)
                            .clipped()
                            .opacity(0.6)
                    }
                }
                .mask(
                    LinearGradient(
                        colors: [Color.clear, Color.black, Color.black, Color.clear],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
            } else {
                Rectangle()
                    .fill(Color.davinciDark)
                    .frame(height: 24)
            }
            
            // Playhead
            Rectangle()
                .fill(Color.davinciHighlight)
                .frame(width: 2, height: 30)
                .offset(x: playheadPosition)
            
            // Loading indicator
            if isLoadingFrames {
                HStack {
                    Spacer()
                    ProgressView()
                        .scaleEffect(0.3)
                    Spacer()
                }
            }
        }
        .frame(height: 24)
        .onAppear {
            Task {
                await loadFilmstrip()
            }
        }
    }
    
    var playheadPosition: CGFloat {
        guard item.duration ?? 0 > 0, !filmstripFrames.isEmpty else { return 0 }
        let frameWidth: CGFloat = 40
        let totalWidth = CGFloat(filmstripFrames.count) * (frameWidth + 1)
        return totalWidth * CGFloat(currentTime / item.duration ?? 0)
    }
    
    @MainActor
    private func loadFilmstrip() async {
        guard filmstripFrames.isEmpty else { return }
        
        isLoadingFrames = true
        defer { isLoadingFrames = false }
        
        let asset = AVAsset(url: item.url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 80, height: 48)
        
        // Generate frames
        let frameCount = 20
        var frames: [NSImage] = []
        
        for i in 0..<frameCount {
            let progress = Double(i) / Double(frameCount - 1)
            let time = CMTime(seconds: item.duration ?? 0 * progress, preferredTimescale: 600)
            
            do {
                let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
                let nsImage = NSImage(cgImage: cgImage, size: CGSize(width: 80, height: 48))
                frames.append(nsImage)
            } catch {
                // Use thumbnail as fallback
                if let thumbnail = item.thumbnail {
                    frames.append(thumbnail)
                }
            }
        }
        
        filmstripFrames = frames
    }
}
