// AUTORESOLVE V3.0 - TRACK HEIGHT CONTROLLER
import Combine
// Professional track height adjustment with presets and individual control

import SwiftUI
import AppKit

// MARK: - Track Height Manager
class TrackHeightManager: ObservableObject {
    @Published var videoTrackHeight: CGFloat = 60
    @Published var audioTrackHeight: CGFloat = 40
    @Published var titleTrackHeight: CGFloat = 50
    @Published var directorTrackHeight: CGFloat = 80
    @Published var transcriptionTrackHeight: CGFloat = 45
    
    @Published var individualHeights: [UUID: CGFloat] = [:]
    @Published var selectedPreset: HeightPreset = .medium
    
    enum HeightPreset: String, CaseIterable {
        case minimal = "Minimal"
        case small = "Small"
        case medium = "Medium"
        case large = "Large"
        case huge = "Huge"
        case custom = "Custom"
        
        var videoHeight: CGFloat {
            switch self {
            case .minimal: return 30
            case .small: return 45
            case .medium: return 60
            case .large: return 90
            case .huge: return 120
            case .custom: return 60
            }
        }
        
        var audioHeight: CGFloat {
            switch self {
            case .minimal: return 20
            case .small: return 30
            case .medium: return 40
            case .large: return 60
            case .huge: return 80
            case .custom: return 40
            }
        }
    }
    
    func applyPreset(_ preset: HeightPreset) {
        selectedPreset = preset
        if preset != .custom {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                videoTrackHeight = preset.videoHeight
                audioTrackHeight = preset.audioHeight
                titleTrackHeight = preset.videoHeight * 0.83
                directorTrackHeight = preset.videoHeight * 1.33
                transcriptionTrackHeight = preset.audioHeight * 1.125
            }
        }
    }
    
    func setTrackHeight(id: UUID, height: CGFloat) {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            individualHeights[id] = min(200, max(20, height))
            selectedPreset = .custom
        }
    }
    
    func getTrackHeight(for track: UITimelineTrack) -> CGFloat {
        if let customHeight = individualHeights[track.id] {
            return customHeight
        }
        
        switch track.type {
        case .video: return videoTrackHeight
        case .audio: return audioTrackHeight
        case .title: return titleTrackHeight
        case .effect: return titleTrackHeight // Same as title track
        case .director: return directorTrackHeight
        case .transcription: return transcriptionTrackHeight
        }
    }
    
    func resetToDefault() {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            applyPreset(.medium)
            individualHeights.removeAll()
        }
    }
}

// MARK: - Track Height Control View
struct TrackHeightControl: View {
    @StateObject private var heightManager = TrackHeightManager()
    @State private var showHeightPopover = false
    @State private var expandedTracks: Set<UUID> = []
    
    public var body: some View {
        HStack(spacing: 8) {
            // Height preset menu
            Menu {
                ForEach(TrackHeightManager.HeightPreset.allCases, id: \.self) { preset in
                    Button(action: { heightManager.applyPreset(preset) }) {
                        HStack {
                            Text(preset.rawValue)
                            if heightManager.selectedPreset == preset {
                                Spacer()
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
                
                Divider()
                
                Button("Reset to Default") {
                    heightManager.resetToDefault()
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "rectangle.expand.vertical")
                        .font(.system(size: 12))
                    Text(heightManager.selectedPreset.rawValue)
                        .font(.system(size: 11))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.secondary.opacity(0.1))
                .cornerRadius(4)
            }
            .menuStyle(BorderlessButtonMenuStyle())
            
            // Individual track height button
            Button(action: { showHeightPopover.toggle() }) {
                Image(systemName: "slider.vertical.3")
                    .font(.system(size: 14))
            }
            .help("Adjust individual track heights")
            .popover(isPresented: $showHeightPopover) {
                IndividualTrackHeightPopover(heightManager: heightManager)
            }
            
            // Expand/Collapse all
            Button(action: toggleAllTracksExpansion) {
                Image(systemName: expandedTracks.isEmpty ? "arrow.up.and.down" : "arrow.down.right.and.arrow.up.left")
                    .font(.system(size: 14))
            }
            .help(expandedTracks.isEmpty ? "Expand all tracks" : "Collapse all tracks")
        }
    }
    
    private func toggleAllTracksExpansion() {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            if expandedTracks.isEmpty {
                // Expand all
                heightManager.applyPreset(.large)
            } else {
                // Collapse all
                heightManager.applyPreset(.small)
                expandedTracks.removeAll()
            }
        }
    }
}

// MARK: - Individual Track Height Popover
struct IndividualTrackHeightPopover: View {
    @ObservedObject var heightManager: TrackHeightManager
    @State private var tracks: [UITimelineTrack] = [] // Should come from timeline
    
    public var body: some View {
        VStack(spacing: 12) {
            Text("Track Heights")
                .font(.system(size: 13, weight: .semibold))
            
            VStack(spacing: 8) {
                // Video tracks height
                TrackTypeHeightSlider(
                    title: "Video Tracks",
                    height: $heightManager.videoTrackHeight,
                    icon: "video",
                    color: .blue
                )
                
                // Audio tracks height
                TrackTypeHeightSlider(
                    title: "Audio Tracks",
                    height: $heightManager.audioTrackHeight,
                    icon: "waveform",
                    color: .green
                )
                
                // Title tracks height
                TrackTypeHeightSlider(
                    title: "Title Tracks",
                    height: $heightManager.titleTrackHeight,
                    icon: "textformat",
                    color: .purple
                )
                
                // Director track height
                TrackTypeHeightSlider(
                    title: "Director Track",
                    height: $heightManager.directorTrackHeight,
                    icon: "sparkles",
                    color: .orange
                )
                
                // Transcription track height
                TrackTypeHeightSlider(
                    title: "Transcription",
                    height: $heightManager.transcriptionTrackHeight,
                    icon: "captions.bubble",
                    color: .cyan
                )
            }
            
            Divider()
            
            HStack {
                Button("Reset All") {
                    heightManager.resetToDefault()
                }
                .buttonStyle(BorderlessButtonStyle())
                
                Spacer()
                
                Button("Apply") {
                    // Close popover
                }
                .buttonStyle(BorderedProminentButtonStyle())
            }
            .controlSize(.small)
        }
        .padding()
        .frame(width: 280)
    }
}

// MARK: - Track Type Height Slider
struct TrackTypeHeightSlider: View {
    let title: String
    @Binding var height: CGFloat
    let icon: String
    let color: Color
    
    @State private var isAdjusting = false
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .foregroundColor(color)
                
                Text(title)
                    .font(.system(size: 11))
                
                Spacer()
                
                Text("\(Int(height))px")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            
            HStack(spacing: 8) {
                Slider(value: $height, in: 20...200, step: 5)
                    .controlSize(.small)
                    .onChange(of: height) { _, _ in
                        isAdjusting = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                            isAdjusting = false
                        }
                    }
                
                // Quick preset buttons
                HStack(spacing: 2) {
                    QuickHeightButton(value: 30, currentHeight: $height, label: "S")
                    QuickHeightButton(value: 60, currentHeight: $height, label: "M")
                    QuickHeightButton(value: 90, currentHeight: $height, label: "L")
                }
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isAdjusting ? color.opacity(0.1) : Color.clear)
                .animation(.easeInOut(duration: 0.2), value: isAdjusting)
        )
    }
}

// MARK: - Quick Height Button
struct QuickHeightButton: View {
    let value: CGFloat
    @Binding var currentHeight: CGFloat
    let label: String
    
    private var isSelected: Bool {
        abs(currentHeight - value) < 5
    }
    
    public var body: some View {
        Button(action: { currentHeight = value }) {
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .frame(width: 20, height: 20)
                .background(
                    RoundedRectangle(cornerRadius: 3)
                        .fill(isSelected ? Color.accentColor : Color.secondary.opacity(0.2))
                )
                .foregroundColor(isSelected ? .white : .primary)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Resizable Track Header
struct ResizableTrackHeader: View {
    let track: TimelineTrack
    @ObservedObject var heightManager: TrackHeightManager
    @State private var isDraggingResize = false
    @State private var dragStartHeight: CGFloat = 0
    
    private var trackHeight: CGFloat {
        heightManager.getTrackHeight(for: UITimelineTrack(name: track.name, type: UITimelineTrack.TrackType(rawValue: track.type.rawValue) ?? .video))
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Main header content
            HStack {
                // Track icon and name
                HStack(spacing: 6) {
                    TrackIcon(type: track.type)
                        .font(.system(size: 12))
                    
                    Text(track.name)
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(1)
                }
                
                Spacer()
                
                // Track controls
                TrackControls(track: track)
            }
            .padding(.horizontal, 8)
            .frame(height: min(30, trackHeight))
            
            Spacer()
            
            // Resize handle
            Rectangle()
                .fill(Color.clear)
                .frame(height: 4)
                .overlay(
                    Rectangle()
                        .fill(Color.secondary.opacity(isDraggingResize ? 0.5 : 0.2))
                        .frame(height: 1)
                )
                .contentShape(Rectangle())
                .onHover { hovering in
                    if hovering {
                        NSCursor.resizeUpDown.push()
                    } else {
                        NSCursor.pop()
                    }
                }
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            if !isDraggingResize {
                                isDraggingResize = true
                                dragStartHeight = trackHeight
                            }
                            let newHeight = dragStartHeight + value.translation.height
                            heightManager.setTrackHeight(id: track.id, height: newHeight)
                        }
                        .onEnded { _ in
                            isDraggingResize = false
                        }
                )
        }
        .frame(height: trackHeight)
        .background(Color(NSColor.controlBackgroundColor))
        .overlay(
            Rectangle()
                .stroke(Color.secondary.opacity(0.2), lineWidth: 0.5)
        )
    }
}

// MARK: - Track Icon
struct TrackIcon: View {
    let type: TimelineTrack.TrackType
    
    public var body: some View {
        Image(systemName: iconName)
            .foregroundColor(iconColor)
    }
    
    private var iconName: String {
        switch type {
        case .video: return "video"
        case .audio: return "waveform"
        case .title: return "textformat"
        case .effect: return "fx"
        case .director: return "sparkles"
        case .transcription: return "captions.bubble"
        }
    }
    
    private var iconColor: Color {
        switch type {
        case .video: return .blue
        case .audio: return .green
        case .title: return .purple
        case .effect: return .yellow
        case .director: return .orange
        case .transcription: return .cyan
        }
    }
}

// MARK: - Track Controls
struct TrackControls: View {
    let track: TimelineTrack
    @State private var isMuted = false
    @State private var isSolo = false
    @State private var isLocked = false
    @State private var isVisible = true
    
    public var body: some View {
        HStack(spacing: 4) {
            // Visibility toggle
            Button(action: { isVisible.toggle() }) {
                Image(systemName: isVisible ? "eye" : "eye.slash")
                    .font(.system(size: 10))
                    .foregroundColor(isVisible ? .primary : .secondary)
            }
            .buttonStyle(PlainButtonStyle())
            .frame(width: 20, height: 20)
            .help(isVisible ? "Hide track" : "Show track")
            
            // Solo button (audio tracks only)
            if track.type == .audio {
                Button(action: { isSolo.toggle() }) {
                    Text("S")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(isSolo ? .white : .secondary)
                        .frame(width: 16, height: 16)
                        .background(
                            RoundedRectangle(cornerRadius: 2)
                                .fill(isSolo ? Color.yellow : Color.clear)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 2)
                                        .stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                                )
                        )
                }
                .buttonStyle(PlainButtonStyle())
                .help("Solo track")
            }
            
            // Mute button (audio/video tracks)
            if track.type == .audio || track.type == .video {
                Button(action: { isMuted.toggle() }) {
                    Text("M")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(isMuted ? .white : .secondary)
                        .frame(width: 16, height: 16)
                        .background(
                            RoundedRectangle(cornerRadius: 2)
                                .fill(isMuted ? Color.red : Color.clear)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 2)
                                        .stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                                )
                        )
                }
                .buttonStyle(PlainButtonStyle())
                .help("Mute track")
            }
            
            // Lock button
            Button(action: { isLocked.toggle() }) {
                Image(systemName: isLocked ? "lock.fill" : "lock.open")
                    .font(.system(size: 10))
                    .foregroundColor(isLocked ? .orange : .secondary)
            }
            .buttonStyle(PlainButtonStyle())
            .frame(width: 20, height: 20)
            .help(isLocked ? "Unlock track" : "Lock track")
        }
    }
}

