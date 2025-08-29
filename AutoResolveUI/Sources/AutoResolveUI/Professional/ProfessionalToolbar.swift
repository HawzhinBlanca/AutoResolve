// AUTORESOLVE V3.0 - PROFESSIONAL DAVINCI RESOLVE-STYLE TOOLBAR
// Compact, feature-rich toolbar matching DaVinci Resolve Edit page

import SwiftUI
import AVFoundation

// MARK: - Main Professional Toolbar
public struct ProfessionalToolbar: View {
    @EnvironmentObject private var store: UnifiedStore
    @State private var currentTool: EditTool = .selection
    @State private var isPlaying = false
    @State private var currentTime: TimeInterval = 0
    @State private var duration: TimeInterval = 100
    @State private var showRenderQueue = false
    @State private var showMediaImport = false
    @State private var selectedWorkspace: Workspace = .edit
    
    enum EditTool: String, CaseIterable {
        case selection = "arrow.up.left"
        case blade = "scissors"
        case trim = "arrow.left.and.right"
        case slip = "arrow.up.and.down.and.arrow.left.and.right"
        case slide = "arrow.left.arrow.right"
        case zoom = "magnifyingglass"
        case hand = "hand.raised"
        
        var tooltip: String {
            switch self {
            case .selection: return "Selection Mode (A)"
            case .blade: return "Blade Edit Mode (B)"
            case .trim: return "Trim Edit Mode (T)"
            case .slip: return "Slip Edit Mode (Y)"
            case .slide: return "Slide Edit Mode (U)"
            case .zoom: return "Zoom Mode (Z)"
            case .hand: return "Hand Mode (H)"
            }
        }
    }
    
    enum Workspace: String, CaseIterable {
        case media = "Media"
        case cut = "Cut"
        case edit = "Edit"
        case fusion = "Fusion"
        case color = "Color"
        case fairlight = "Fairlight"
        case deliver = "Deliver"
        
        var icon: String {
            switch self {
            case .media: return "photo.stack"
            case .cut: return "scissors.badge.ellipsis"
            case .edit: return "slider.horizontal.3"
            case .fusion: return "fx"
            case .color: return "paintpalette"
            case .fairlight: return "waveform"
            case .deliver: return "shippingbox"
            }
        }
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // TOP ROW - Workspace tabs + Main controls
            HStack(spacing: 0) {
                // Workspace Switcher (Left)
                WorkspaceSwitcher(selected: $selectedWorkspace)
                    .frame(width: 400)
                
                Spacer()
                
                // Timeline Name & Project Info (Center)
                TimelineInfo()
                
                Spacer()
                
                // Quick Actions (Right)
                QuickActions(
                    showRenderQueue: $showRenderQueue,
                    showMediaImport: $showMediaImport
                )
            }
            .frame(height: 36)
            .background(Color(white: 0.12))
            
            Divider()
            
            // BOTTOM ROW - Tools + Transport + Timeline controls
            HStack(spacing: 12) {
                // Edit Tools (Left)
                EditToolsBar(currentTool: $currentTool)
                
                Divider()
                    .frame(height: 20)
                
                // Transport Controls (Center)
                TransportControls(
                    isPlaying: $isPlaying,
                    currentTime: $currentTime,
                    duration: duration
                )
                
                Divider()
                    .frame(height: 20)
                
                // Timeline Controls (Right)
                TimelineControls()
                
                // Audio Meters
                AudioMeters()
                    .frame(width: 100)
            }
            .frame(height: 44)
            .padding(.horizontal, 12)
            .background(Color(white: 0.15))
        }
        .fileImporter(
            isPresented: $showMediaImport,
            allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie],
            allowsMultipleSelection: true
        ) { result in
            handleMediaImport(result)
        }
        .sheet(isPresented: $showRenderQueue) {
            RenderQueueView()
        }
    }
    
    private func handleMediaImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            for url in urls {
                store.importVideo(url: url)
            }
        case .failure(let error):
            print("Import failed: \(error)")
        }
    }
}

// MARK: - Workspace Switcher
struct WorkspaceSwitcher: View {
    @Binding var selected: ProfessionalToolbar.Workspace
    
    public var body: some View {
        HStack(spacing: 0) {
            ForEach(ProfessionalToolbar.Workspace.allCases, id: \.self) { workspace in
                Button(action: { selected = workspace }) {
                    VStack(spacing: 2) {
                        Image(systemName: workspace.icon)
                            .font(.system(size: 14))
                        Text(workspace.rawValue)
                            .font(.system(size: 9, weight: .medium))
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 32)
                    .foregroundColor(selected == workspace ? .white : .gray)
                    .background(
                        selected == workspace ?
                        Color.orange.opacity(0.3) : Color.clear
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }
}

// MARK: - Timeline Info
struct TimelineInfo: View {
    @EnvironmentObject private var store: UnifiedStore
    
    public var body: some View {
        VStack(spacing: 0) {
            Text("Untitled Timeline")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.white)
            Text("1920x1080 • 30fps • 00:00:00:00")
                .font(.system(size: 9))
                .foregroundColor(.gray)
        }
    }
}

// MARK: - Quick Actions
struct QuickActions: View {
    @Binding var showRenderQueue: Bool
    @Binding var showMediaImport: Bool
    @EnvironmentObject var backendService: BackendClient
    
    public var body: some View {
        HStack(spacing: 8) {
            // Import Media
            Button(action: { showMediaImport.toggle() }) {
                Label("Import", systemImage: "plus.circle")
                    .font(.system(size: 11))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("Import Media (⌘I)")
            
            // Auto Edit
            Button(action: triggerAutoEdit) {
                Label("Auto Edit", systemImage: "wand.and.stars")
                    .font(.system(size: 11))
            }
            .buttonStyle(.plain)
            .foregroundColor(.blue)
            .help("AI Auto Edit")
            
            // Render Queue
            Button(action: { showRenderQueue.toggle() }) {
                Label("Queue", systemImage: "rectangle.stack.badge.play")
                    .font(.system(size: 11))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("Render Queue")
            
            // Connection Status
            ConnectionIndicator()
        }
        .padding(.horizontal, 12)
    }
    
    private func triggerAutoEdit() {
        // Trigger AI-powered auto editing
    }
}

// MARK: - Edit Tools Bar
struct EditToolsBar: View {
    @Binding var currentTool: ProfessionalToolbar.EditTool
    
    public var body: some View {
        HStack(spacing: 2) {
            ForEach(ProfessionalToolbar.EditTool.allCases, id: \.self) { tool in
                Button(action: { currentTool = tool }) {
                    Image(systemName: tool.rawValue)
                        .font(.system(size: 16))
                        .frame(width: 28, height: 28)
                        .foregroundColor(currentTool == tool ? .orange : .gray)
                        .background(
                            currentTool == tool ?
                            Color.orange.opacity(0.2) :
                            Color.white.opacity(0.05)
                        )
                        .cornerRadius(4)
                }
                .buttonStyle(.plain)
                .help(tool.tooltip)
            }
        }
    }
}

// MARK: - Transport Controls
struct TransportControls: View {
    @Binding var isPlaying: Bool
    @Binding var currentTime: TimeInterval
    let duration: TimeInterval
    @EnvironmentObject private var store: UnifiedStore
    
    public var body: some View {
        HStack(spacing: 8) {
            // Go to Start
            Button(action: goToStart) {
                Image(systemName: "backward.end.fill")
                    .font(.system(size: 14))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("Go to Start (Home)")
            
            // Previous Frame
            Button(action: previousFrame) {
                Image(systemName: "backward.frame.fill")
                    .font(.system(size: 14))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("Previous Frame (←)")
            
            // Play Reverse
            Button(action: playReverse) {
                Image(systemName: "play.fill")
                    .rotation3DEffect(.degrees(180), axis: (x: 0, y: 1, z: 0))
                    .font(.system(size: 14))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("Play Reverse (J)")
            
            // Stop
            Button(action: stop) {
                Image(systemName: "stop.fill")
                    .font(.system(size: 14))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("Stop (K)")
            
            // Play/Pause
            Button(action: togglePlay) {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 18))
                    .frame(width: 36, height: 36)
                    .foregroundColor(.white)
                    .background(isPlaying ? Color.orange : Color.green)
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Play/Pause (Space)")
            
            // Play Forward
            Button(action: playForward) {
                Image(systemName: "play.fill")
                    .font(.system(size: 14))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("Play Forward (L)")
            
            // Next Frame
            Button(action: nextFrame) {
                Image(systemName: "forward.frame.fill")
                    .font(.system(size: 14))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("Next Frame (→)")
            
            // Go to End
            Button(action: goToEnd) {
                Image(systemName: "forward.end.fill")
                    .font(.system(size: 14))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("Go to End (End)")
            
            // Loop Toggle
            Button(action: toggleLoop) {
                Image(systemName: "repeat")
                    .font(.system(size: 14))
            }
            .buttonStyle(.plain)
            .foregroundColor(store.isLooping ? .orange : .gray)
            .help("Loop Playback")
            
            Divider()
                .frame(height: 20)
            
            // Timecode Display
            TimecodeDisplay(currentTime: currentTime, duration: duration)
        }
    }
    
    private func goToStart() {
        currentTime = 0
        store.seekToTime(0)
    }
    
    private func previousFrame() {
        let frameTime = 1.0 / 30.0 // 30fps
        currentTime = max(0, currentTime - frameTime)
        store.seekToTime(currentTime)
    }
    
    private func playReverse() {
        store.playReverse()
    }
    
    private func stop() {
        isPlaying = false
        store.pause()
    }
    
    private func togglePlay() {
        isPlaying.toggle()
        if isPlaying {
            store.play()
        } else {
            store.pause()
        }
    }
    
    private func playForward() {
        store.playForward()
    }
    
    private func nextFrame() {
        let frameTime = 1.0 / 30.0
        currentTime = min(duration, currentTime + frameTime)
        store.seekToTime(currentTime)
    }
    
    private func goToEnd() {
        currentTime = duration
        store.seekToTime(duration)
    }
    
    private func toggleLoop() {
        store.isLooping.toggle()
    }
}

// MARK: - Timecode Display
struct TimecodeDisplay: View {
    let currentTime: TimeInterval
    let duration: TimeInterval
    
    public var body: some View {
        HStack(spacing: 4) {
            Text(formatTimecode(currentTime))
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.white)
            
            Text("/")
                .foregroundColor(.gray)
            
            Text(formatTimecode(duration))
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.gray)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.black.opacity(0.5))
        .cornerRadius(4)
    }
    
    private func formatTimecode(_ time: TimeInterval) -> String {
        guard time.isFinite && time >= 0 else { return "00:00:00:00" }
        let safeTime = min(time, 359999.0) // Cap at 99:59:59:29
        let hours = Int(safeTime) / 3600
        let minutes = Int(safeTime) / 60 % 60
        let seconds = Int(safeTime) % 60
        let frames = Int((safeTime - Double(Int(safeTime))) * 30)
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, seconds, frames)
    }
}

// MARK: - Timeline Controls
struct TimelineControls: View {
    @EnvironmentObject private var store: UnifiedStore
    @State private var zoomLevel: CGFloat = 1.0
    @State private var showSnapping = true
    @State private var showWaveforms = true
    
    public var body: some View {
        HStack(spacing: 12) {
            // Snapping
            Button(action: { showSnapping.toggle() }) {
                Image(systemName: "line.horizontal.3.decrease.circle")
                    .font(.system(size: 14))
                    .foregroundColor(showSnapping ? .orange : .gray)
            }
            .buttonStyle(.plain)
            .help("Toggle Snapping (N)")
            
            // Linked Selection
            Button(action: toggleLinkedSelection) {
                Image(systemName: "link")
                    .font(.system(size: 14))
                    .foregroundColor(store.linkedSelection ? .orange : .gray)
            }
            .buttonStyle(.plain)
            .help("Toggle Linked Selection")
            
            // Show Waveforms
            Button(action: { showWaveforms.toggle() }) {
                Image(systemName: "waveform")
                    .font(.system(size: 14))
                    .foregroundColor(showWaveforms ? .orange : .gray)
            }
            .buttonStyle(.plain)
            .help("Toggle Audio Waveforms")
            
            Divider()
                .frame(height: 20)
            
            // Zoom Controls
            HStack(spacing: 4) {
                Button(action: zoomOut) {
                    Image(systemName: "minus.magnifyingglass")
                        .font(.system(size: 12))
                }
                .buttonStyle(.plain)
                .foregroundColor(.gray)
                
                Slider(value: $zoomLevel, in: 0.1...10)
                    .frame(width: 80)
                    .controlSize(.mini)
                
                Button(action: zoomIn) {
                    Image(systemName: "plus.magnifyingglass")
                        .font(.system(size: 12))
                }
                .buttonStyle(.plain)
                .foregroundColor(.gray)
                
                Button(action: zoomToFit) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                        .font(.system(size: 12))
                }
                .buttonStyle(.plain)
                .foregroundColor(.gray)
                .help("Zoom to Fit (⇧Z)")
            }
        }
    }
    
    private func toggleLinkedSelection() {
        store.linkedSelection.toggle()
    }
    
    private func zoomOut() {
        zoomLevel = max(0.1, zoomLevel - 0.5)
    }
    
    private func zoomIn() {
        zoomLevel = min(10, zoomLevel + 0.5)
    }
    
    private func zoomToFit() {
        zoomLevel = 1.0
    }
}

// MARK: - Audio Meters
struct AudioMeters: View {
    @State private var leftLevel: Float = -40
    @State private var rightLevel: Float = -35
    
    public var body: some View {
        HStack(spacing: 4) {
            VUMeter(level: leftLevel, color: .green)
            VUMeter(level: rightLevel, color: .green)
        }
        .padding(.horizontal, 8)
    }
}

struct VUMeter: View {
    let level: Float // dB level
    let color: Color
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                // Background
                Rectangle()
                    .fill(Color.black.opacity(0.5))
                
                // Level indicator
                Rectangle()
                    .fill(levelGradient)
                    .frame(height: geometry.size.height * CGFloat(normalizedLevel))
            }
        }
        .frame(width: 20)
        .cornerRadius(2)
    }
    
    private var normalizedLevel: Float {
        // Convert dB to 0-1 scale
        let minDB: Float = -60
        let maxDB: Float = 0
        return (level - minDB) / (maxDB - minDB)
    }
    
    private var levelGradient: LinearGradient {
        LinearGradient(
            colors: [.red, .yellow, .green],
            startPoint: .top,
            endPoint: .bottom
        )
    }
}

// MARK: - Connection Indicator
struct ConnectionIndicator: View {
    @EnvironmentObject var backendService: BackendClient
    
    public var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(backendService.isConnected ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            
            Text(backendService.isConnected ? "Connected" : "Offline")
                .font(.system(size: 9))
                .foregroundColor(backendService.isConnected ? .green : .red)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.black.opacity(0.3))
        .cornerRadius(4)
    }
}

// MARK: - Render Queue View
struct RenderQueueView: View {
    @Environment(\.dismiss) var dismiss
    
    public var body: some View {
        VStack {
            HStack {
                Text("Render Queue")
                    .font(.title2)
                    .bold()
                
                Spacer()
                
                Button("Close") {
                    dismiss()
                }
            }
            .padding()
            
            // Render queue content here
            Text("No items in render queue")
                .foregroundColor(.gray)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(white: 0.1))
        }
        .frame(width: 600, height: 400)
        .background(Color(white: 0.15))
    }
}
