#!/bin/bash

# Create remaining AutoResolveUI files

# Base directory
BASE="/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI"

# Create directories
mkdir -p "$BASE/App"
mkdir -p "$BASE/Timeline"
mkdir -p "$BASE/Playback"
mkdir -p "$BASE/Media"
mkdir -p "$BASE/Inspector"
mkdir -p "$BASE/Export"
mkdir -p "$BASE/Utilities"
mkdir -p "$BASE/Metal"
mkdir -p "$BASE/Security"

# ProjectManager.swift
cat > "$BASE/App/ProjectManager.swift" << 'EOF'
import Foundation
import AutoResolveCore

public class ProjectManagerUI {
    private let projectManager = ProjectManager()
    
    public init() {}
    
    public func createProject(name: String) -> Project {
        return projectManager.createProject(name: name)
    }
    
    public func openProject(at url: URL) throws -> Project {
        return try projectManager.openProject(at: url)
    }
    
    public func saveProject(_ project: Project, to url: URL) throws {
        try projectManager.saveProject(project, to: url)
    }
}
EOF

# CommandProcessor.swift
cat > "$BASE/App/CommandProcessor.swift" << 'EOF'
import Foundation
import AutoResolveCore

public class CommandProcessor {
    private var undoStack: [Command] = []
    private var redoStack: [Command] = []
    
    public init() {}
    
    public func execute(_ command: Command, on project: Project) throws {
        try project.execute(command)
        undoStack.append(command)
        redoStack.removeAll()
    }
    
    public func undo(on project: Project) throws {
        guard let command = undoStack.popLast() else { return }
        try project.undo(command)
        redoStack.append(command)
    }
    
    public func redo(on project: Project) throws {
        guard let command = redoStack.popLast() else { return }
        try project.execute(command)
        undoStack.append(command)
    }
}
EOF

# TimelineView.swift
cat > "$BASE/Timeline/TimelineView.swift" << 'EOF'
import SwiftUI
import AutoResolveCore

struct TimelineView: View {
    @EnvironmentObject var appState: AppState
    @State private var dragLocation: CGPoint = .zero
    @State private var isDragging = false
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background
                Color.black.opacity(0.9)
                
                // Timeline tracks
                ScrollView([.horizontal, .vertical]) {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(0..<3, id: \.self) { trackIndex in
                            TrackView(trackIndex: trackIndex)
                                .frame(height: 80)
                        }
                    }
                    .frame(width: max(1000, geometry.size.width * appState.zoomLevel))
                }
                
                // Playhead
                PlayheadView()
                
                // AI Suggestions overlay
                if appState.showAISuggestions {
                    AISuggestionsOverlay()
                }
            }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            handleDrop(providers)
            return true
        }
    }
    
    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        // Handle media drop
        return true
    }
}

struct TrackView: View {
    let trackIndex: Int
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HStack(spacing: 2) {
            if let timeline = appState.timeline,
               trackIndex < timeline.tracks.count {
                ForEach(timeline.tracks[trackIndex].clips, id: \.id) { clip in
                    ClipView(clip: clip)
                }
            }
            Spacer()
        }
        .background(Color.gray.opacity(0.2))
        .border(Color.gray.opacity(0.5), width: 1)
    }
}

struct ClipView: View {
    let clip: Clip
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(Color.blue.opacity(0.7))
            .frame(width: clipWidth)
            .overlay(
                Text(clip.name ?? "Clip")
                    .font(.caption)
                    .foregroundColor(.white)
            )
            .onTapGesture {
                if appState.selectedClips.contains(clip.id) {
                    appState.selectedClips.remove(clip.id)
                } else {
                    appState.selectedClips.insert(clip.id)
                }
            }
    }
    
    private var clipWidth: CGFloat {
        let duration = clip.duration.seconds
        return CGFloat(duration * 50 * appState.zoomLevel)
    }
}

struct PlayheadView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        GeometryReader { geometry in
            Rectangle()
                .fill(Color.red)
                .frame(width: 2, height: geometry.size.height)
                .offset(x: playheadPosition)
        }
        .allowsHitTesting(false)
    }
    
    private var playheadPosition: CGFloat {
        CGFloat(appState.currentTime.seconds * 50 * appState.zoomLevel)
    }
}

struct AISuggestionsOverlay: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        GeometryReader { geometry in
            ForEach(appState.aiSuggestions, id: \.id) { suggestion in
                SuggestionMarker(suggestion: suggestion)
            }
        }
    }
}

struct SuggestionMarker: View {
    let suggestion: EditSuggestion
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack {
            Image(systemName: iconName)
                .foregroundColor(color)
                .font(.caption)
            
            Text(String(format: "%.0f%%", suggestion.confidence * 100))
                .font(.caption2)
                .foregroundColor(.white)
        }
        .offset(x: position, y: 10)
        .onTapGesture {
            appState.applySuggestion(suggestion)
        }
    }
    
    private var position: CGFloat {
        CGFloat(suggestion.tick.seconds * 50 * appState.zoomLevel)
    }
    
    private var iconName: String {
        switch suggestion.type {
        case .cut: return "scissors"
        case .trim: return "arrow.left.and.right"
        case .delete: return "trash"
        case .transition: return "arrow.triangle.2.circlepath"
        }
    }
    
    private var color: Color {
        if suggestion.confidence > 0.8 {
            return .green
        } else if suggestion.confidence > 0.6 {
            return .yellow
        } else {
            return .orange
        }
    }
}
EOF

# TimelineRuler.swift
cat > "$BASE/Timeline/TimelineRuler.swift" << 'EOF'
import SwiftUI
import AutoResolveCore

struct TimelineRuler: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                drawRuler(context: context, size: size)
            }
        }
        .frame(height: 30)
        .background(Color.gray.opacity(0.2))
    }
    
    private func drawRuler(context: GraphicsContext, size: CGSize) {
        let pixelsPerSecond = 50.0 * appState.zoomLevel
        let totalSeconds = Int(size.width / pixelsPerSecond)
        
        for second in 0...totalSeconds {
            let x = CGFloat(second) * pixelsPerSecond
            
            // Major tick every 5 seconds
            let isMajor = second % 5 == 0
            let height: CGFloat = isMajor ? 15 : 8
            
            context.stroke(
                Path { path in
                    path.move(to: CGPoint(x: x, y: size.height - height))
                    path.addLine(to: CGPoint(x: x, y: size.height))
                },
                with: .color(.gray)
            )
            
            if isMajor {
                context.draw(
                    Text(formatTime(second))
                        .font(.caption2)
                        .foregroundColor(.gray),
                    at: CGPoint(x: x, y: 10)
                )
            }
        }
    }
    
    private func formatTime(_ seconds: Int) -> String {
        let minutes = seconds / 60
        let secs = seconds % 60
        return String(format: "%d:%02d", minutes, secs)
    }
}
EOF

# Create more Timeline files
cat > "$BASE/Timeline/TimelineToolbar.swift" << 'EOF'
import SwiftUI

struct TimelineToolbar: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HStack {
            Button(action: { appState.blade() }) {
                Image(systemName: "scissors")
            }
            .disabled(!appState.canBlade)
            
            Button(action: { appState.deleteSelection() }) {
                Image(systemName: "trash")
            }
            .disabled(!appState.hasSelection)
            
            Divider()
            
            Button(action: { appState.zoomOut() }) {
                Image(systemName: "minus.magnifyingglass")
            }
            
            Button(action: { appState.zoomToFit() }) {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
            }
            
            Button(action: { appState.zoomIn() }) {
                Image(systemName: "plus.magnifyingglass")
            }
            
            Spacer()
            
            if appState.aiAnalyzing {
                ProgressView()
                    .scaleEffect(0.7)
                Text("AI Analyzing...")
                    .font(.caption)
            }
        }
        .padding(.horizontal)
        .frame(height: 30)
        .background(Color.gray.opacity(0.1))
    }
}
EOF

cat > "$BASE/Timeline/TimelineInteraction.swift" << 'EOF'
import SwiftUI
import AutoResolveCore

struct TimelineInteraction: ViewModifier {
    @EnvironmentObject var appState: AppState
    @State private var dragStart: CGPoint?
    @State private var selection: CGRect?
    
    func body(content: Content) -> some View {
        content
            .onTapGesture { location in
                handleClick(at: location)
            }
            .gesture(
                DragGesture()
                    .onChanged { value in
                        handleDrag(value)
                    }
                    .onEnded { _ in
                        endDrag()
                    }
            )
    }
    
    private func handleClick(at location: CGPoint) {
        // Update playhead position
        let tick = locationToTick(location)
        appState.currentTime = tick
    }
    
    private func handleDrag(_ value: DragGesture.Value) {
        if dragStart == nil {
            dragStart = value.startLocation
        }
        
        // Update selection rectangle
        selection = CGRect(
            x: min(value.startLocation.x, value.location.x),
            y: min(value.startLocation.y, value.location.y),
            width: abs(value.location.x - value.startLocation.x),
            height: abs(value.location.y - value.startLocation.y)
        )
    }
    
    private func endDrag() {
        dragStart = nil
        selection = nil
    }
    
    private func locationToTick(_ location: CGPoint) -> Tick {
        let seconds = location.x / (50.0 * appState.zoomLevel)
        return Tick.from(seconds: seconds)
    }
}

extension View {
    func timelineInteraction() -> some View {
        modifier(TimelineInteraction())
    }
}
EOF

# PlaybackEngine.swift
cat > "$BASE/Playback/PlaybackEngine.swift" << 'EOF'
import AVFoundation
import AutoResolveCore

public class PlaybackEngine: ObservableObject {
    private var player: AVPlayer?
    private var playerItem: AVPlayerItem?
    private var composition: AVMutableComposition?
    
    @Published var isPlaying = false
    @Published var currentTime = CMTime.zero
    @Published var duration = CMTime.zero
    
    public init() {}
    
    public func loadTimeline(_ timeline: Timeline) {
        composition = AVMutableComposition()
        
        // Build composition from timeline
        buildComposition(from: timeline)
        
        // Create player item
        if let composition = composition {
            playerItem = AVPlayerItem(asset: composition)
            player = AVPlayer(playerItem: playerItem)
            
            // Get duration
            duration = composition.duration
        }
    }
    
    private func buildComposition(from timeline: Timeline) {
        guard let composition = composition else { return }
        
        // Add video tracks
        let videoTrack = composition.addMutableTrack(
            withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid
        )
        
        // Add audio tracks
        let audioTrack = composition.addMutableTrack(
            withMediaType: .audio,
            preferredTrackID: kCMPersistentTrackID_Invalid
        )
        
        // Build tracks from clips
        for track in timeline.tracks {
            for clip in track.clips {
                insertClip(clip, into: composition)
            }
        }
    }
    
    private func insertClip(_ clip: Clip, into composition: AVMutableComposition) {
        // Load asset and insert into composition
        // Simplified for now
    }
    
    public func play() {
        player?.play()
        isPlaying = true
    }
    
    public func pause() {
        player?.pause()
        isPlaying = false
    }
    
    public func seek(to time: CMTime) {
        player?.seek(to: time)
        currentTime = time
    }
}
EOF

# Create remaining files in batch
echo "Creating Media files..."
cat > "$BASE/Media/MediaPoolView.swift" << 'EOF'
import SwiftUI
import AutoResolveCore

struct MediaPoolView: View {
    @EnvironmentObject var appState: AppState
    @State private var searchText = ""
    
    var body: some View {
        VStack {
            SearchBar(text: $searchText)
            
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))]) {
                    ForEach(filteredMedia, id: \.id) { item in
                        MediaThumbnail(item: item)
                    }
                }
            }
        }
        .padding()
    }
    
    private var filteredMedia: [MediaItem] {
        guard let pool = appState.currentProject?.mediaPool else { return [] }
        
        if searchText.isEmpty {
            return pool.items
        }
        
        return pool.items.filter { item in
            item.name.localizedCaseInsensitiveContains(searchText)
        }
    }
}

struct MediaThumbnail: View {
    let item: MediaItem
    
    var body: some View {
        VStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.3))
                .frame(width: 100, height: 75)
                .overlay(
                    Image(systemName: iconName)
                        .font(.largeTitle)
                        .foregroundColor(.gray)
                )
            
            Text(item.name)
                .font(.caption)
                .lineLimit(1)
        }
        .draggable(item.url)
    }
    
    private var iconName: String {
        switch item.type {
        case .video: return "video"
        case .audio: return "waveform"
        case .image: return "photo"
        default: return "doc"
        }
    }
}

struct SearchBar: View {
    @Binding var text: String
    
    var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.gray)
            
            TextField("Search media...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
        }
    }
}
EOF

echo "Creating Inspector files..."
cat > "$BASE/Inspector/InspectorView.swift" << 'EOF'
import SwiftUI
import AutoResolveCore

struct InspectorView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack {
            if !appState.selectedClips.isEmpty {
                ClipInspector()
            } else if let _ = appState.currentProject {
                ProjectInspector()
            } else {
                EmptyInspector()
            }
        }
        .frame(width: 300)
        .background(Color.gray.opacity(0.1))
    }
}

struct ClipInspector: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        Form {
            Section("Clip Properties") {
                // Properties
            }
            
            Section("Transform") {
                // Transform controls
            }
            
            Section("Effects") {
                // Effects list
            }
        }
        .padding()
    }
}

struct ProjectInspector: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        Form {
            Section("Project Settings") {
                if let project = appState.currentProject {
                    LabeledContent("Name", value: project.name)
                    LabeledContent("Resolution", value: "1920x1080")
                    LabeledContent("Frame Rate", value: "30 fps")
                }
            }
        }
        .padding()
    }
}

struct EmptyInspector: View {
    var body: some View {
        VStack {
            Spacer()
            Text("No Selection")
                .foregroundColor(.gray)
            Spacer()
        }
    }
}
EOF

echo "Creating Export files..."
cat > "$BASE/Export/ExportView.swift" << 'EOF'
import SwiftUI
import AutoResolveCore

struct ExportView: View {
    @EnvironmentObject var appState: AppState
    @State private var exportFormat = ExportFormat.mov
    @State private var exportQuality = ExportQuality.high
    @State private var isExporting = false
    
    enum ExportFormat: String, CaseIterable {
        case mov = "QuickTime"
        case mp4 = "MP4"
        case prores = "ProRes"
    }
    
    enum ExportQuality: String, CaseIterable {
        case low = "Low"
        case medium = "Medium"
        case high = "High"
        case maximum = "Maximum"
    }
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Export Settings")
                .font(.title2)
            
            Form {
                Picker("Format", selection: $exportFormat) {
                    ForEach(ExportFormat.allCases, id: \.self) { format in
                        Text(format.rawValue).tag(format)
                    }
                }
                
                Picker("Quality", selection: $exportQuality) {
                    ForEach(ExportQuality.allCases, id: \.self) { quality in
                        Text(quality.rawValue).tag(quality)
                    }
                }
            }
            
            HStack {
                Button("Cancel") {
                    // Dismiss
                }
                
                Button("Export") {
                    startExport()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isExporting)
            }
            
            if isExporting {
                ProgressView("Exporting...")
                    .progressViewStyle(.linear)
            }
        }
        .padding()
        .frame(width: 400, height: 300)
    }
    
    private func startExport() {
        isExporting = true
        
        Task {
            // Perform export
            await performExport()
            isExporting = false
        }
    }
    
    private func performExport() async {
        // Export implementation
    }
}
EOF

echo "Creating Security files..."
cat > "$BASE/Security/Sandbox.swift" << 'EOF'
import Foundation

public class Sandbox {
    public static let shared = Sandbox()
    
    private init() {}
    
    public func requestAccess(to url: URL) -> Bool {
        // Request sandbox access
        return url.startAccessingSecurityScopedResource()
    }
    
    public func releaseAccess(to url: URL) {
        url.stopAccessingSecurityScopedResource()
    }
    
    public func isAccessible(_ url: URL) -> Bool {
        return FileManager.default.isReadableFile(atPath: url.path)
    }
}
EOF

echo "Creating Metal files..."
cat > "$BASE/Metal/TimelineRenderer.swift" << 'EOF'
import Metal
import MetalKit
import AutoResolveCore

public class TimelineRenderer {
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var pipelineState: MTLRenderPipelineState?
    
    public init() {
        setupMetal()
    }
    
    private func setupMetal() {
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device?.makeCommandQueue()
        
        // Load shaders and create pipeline
        setupPipeline()
    }
    
    private func setupPipeline() {
        // Create render pipeline
        // Simplified for now
    }
    
    public func render(timeline: Timeline, in view: MTKView) {
        guard let commandBuffer = commandQueue?.makeCommandBuffer(),
              let descriptor = view.currentRenderPassDescriptor else { return }
        
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)
        encoder?.setRenderPipelineState(pipelineState!)
        
        // Render timeline
        renderTimeline(timeline, with: encoder!)
        
        encoder?.endEncoding()
        
        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }
        
        commandBuffer.commit()
    }
    
    private func renderTimeline(_ timeline: Timeline, with encoder: MTLRenderCommandEncoder) {
        // Render each track
        for (index, track) in timeline.tracks.enumerated() {
            renderTrack(track, at: index, with: encoder)
        }
    }
    
    private func renderTrack(_ track: Track, at index: Int, with encoder: MTLRenderCommandEncoder) {
        // Render track clips
        for clip in track.clips {
            renderClip(clip, with: encoder)
        }
    }
    
    private func renderClip(_ clip: Clip, with encoder: MTLRenderCommandEncoder) {
        // Render individual clip
        // Simplified for now
    }
}
EOF

echo "Creating Utilities files..."
cat > "$BASE/Utilities/Extensions.swift" << 'EOF'
import SwiftUI
import AutoResolveCore

extension View {
    func onHover(perform: @escaping (Bool) -> Void) -> some View {
        self.onHover { isHovering in
            perform(isHovering)
        }
    }
}

extension Color {
    static let timelineBackground = Color(white: 0.1)
    static let trackBackground = Color(white: 0.15)
    static let clipDefault = Color.blue.opacity(0.7)
    static let clipSelected = Color.blue
    static let playhead = Color.red
}

extension Tick {
    var displayString: String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        let frames = Int((seconds - Double(totalSeconds)) * 30)
        
        if hours > 0 {
            return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
        } else {
            return String(format: "%02d:%02d:%02d", minutes, secs, frames)
        }
    }
}
EOF

cat > "$BASE/Utilities/KeyboardShortcuts.swift" << 'EOF'
import SwiftUI

struct KeyboardShortcuts: ViewModifier {
    @EnvironmentObject var appState: AppState
    
    func body(content: Content) -> some View {
        content
            .onKeyPress(.space) {
                appState.playPause()
                return .handled
            }
            .onKeyPress(.leftArrow) {
                appState.stepBackward()
                return .handled
            }
            .onKeyPress(.rightArrow) {
                appState.stepForward()
                return .handled
            }
            .onKeyPress(.delete) {
                if appState.hasSelection {
                    appState.deleteSelection()
                    return .handled
                }
                return .ignored
            }
    }
}

extension View {
    func keyboardShortcuts() -> some View {
        modifier(KeyboardShortcuts())
    }
}
EOF

echo "All UI files created successfully!"
chmod +x "$0"