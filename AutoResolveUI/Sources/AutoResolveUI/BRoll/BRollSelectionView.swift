import AppKit
import SwiftUI
import AVFoundation
import AVKit

// MARK: - B-Roll Selection Main View
public struct BRollSelectionView: View {
    @StateObject private var viewModel = BRollSelectionViewModel()
    @EnvironmentObject var timeline: TimelineModel
    @EnvironmentObject var telemetry: PipelineStatusMonitor
    
    @State private var selectedBRollClips: Set<BRollClip.ID> = []
    @State private var showPreview = false
    @State private var previewClip: BRollClip?
    @State private var searchText = ""
    @State private var selectedCategory = "All"
    @State private var sortOrder = SortOrder.relevance
    @State private var showAdvancedSettings = false
    
    enum SortOrder: String, CaseIterable {
        case relevance = "Relevance"
        case duration = "Duration"
        case recent = "Recent"
        case name = "Name"
    }
    
    public var body: some View {
        HSplitView {
            // Left: B-Roll Library
            brollLibraryPanel
                .frame(minWidth: 300, maxWidth: 500)
            
            // Center: Timeline with B-Roll Suggestions
            timelineSuggestionsPanel
                .frame(minWidth: 400)
            
            // Right: B-Roll Inspector
            if !selectedBRollClips.isEmpty {
                brollInspectorPanel
                    .frame(width: 300)
            }
        }
        .onAppear {
            viewModel.loadBRollLibrary()
            viewModel.analyzeTimelineForBRoll(timeline: timeline)
        }
        .sheet(isPresented: $showPreview) {
            if let clip = previewClip {
                BRollPreviewSheet(clip: clip)
            }
        }
    }
    
    // MARK: - B-Roll Library Panel
    private var brollLibraryPanel: some View {
        VStack(spacing: 0) {
            // Header with Search and Filters
            VStack(spacing: 12) {
                Text("B-Roll Library")
                    .font(.headline)
                
                // Search Bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    
                    TextField("Search B-roll...", text: $searchText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                
                // Category Filter
                Picker("Category", selection: $selectedCategory) {
                    Text("All").tag("All")
                    ForEach(viewModel.categories, id: \.self) { category in
                        Text(category).tag(category)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                
                // Sort Options
                HStack {
                    Text("Sort by:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Picker("Sort", selection: $sortOrder) {
                        ForEach(SortOrder.allCases, id: \.self) { order in
                            Text(order.rawValue).tag(order)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(width: 120)
                    
                    Spacer()
                    
                    Button(action: { viewModel.refreshLibrary() }) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .buttonStyle(BorderlessButtonStyle())
                }
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // B-Roll Grid
            ScrollView {
                LazyVGrid(columns: [
                    GridItem(.adaptive(minimum: 120, maximum: 150))
                ], spacing: 12) {
                    ForEach(filteredBRollClips) { clip in
                        BRollThumbnailView(
                            clip: clip,
                            isSelected: selectedBRollClips.contains(clip.id),
                            onTap: {
                                toggleSelection(clip)
                            },
                            onDoubleTap: {
                                previewClip = clip
                                showPreview = true
                            }
                        )
                        .contextMenu {
                            contextMenuForClip(clip)
                        }
                    }
                }
                .padding()
            }
            
            // Footer with Actions
            HStack {
                Text("\(filteredBRollClips.count) clips")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Button("Import...") {
                    viewModel.importBRollClips()
                }
                .buttonStyle(BorderlessButtonStyle())
                
                Button("Apply Selected") {
                    applySelectedBRoll()
                }
                .disabled(selectedBRollClips.isEmpty)
                .buttonStyle(BorderedProminentButtonStyle())
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
        }
    }
    
    // MARK: - Timeline Suggestions Panel
    private var timelineSuggestionsPanel: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("B-Roll Suggestions")
                    .font(.headline)
                
                Spacer()
                
                Toggle("Auto-Suggest", isOn: $viewModel.autoSuggestEnabled)
                    .toggleStyle(SwitchToggleStyle())
                
                Button("Analyze") {
                    viewModel.analyzeTimelineForBRoll(timeline: timeline)
                }
                .disabled(viewModel.isAnalyzing)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Timeline with Overlay
            ZStack {
                // Existing Timeline View
                TimelineVisualizationView(timeline: timeline)
                
                // B-Roll Suggestion Overlay
                TimelineBRollOverlay(
                    suggestions: viewModel.brollSuggestions,
                    timeline: timeline,
                    onSuggestionTap: { suggestion in
                        viewModel.selectedSuggestion = suggestion
                    }
                )
            }
            .frame(maxHeight: .infinity)
            
            // Suggestion Details
            if let suggestion = viewModel.selectedSuggestion {
                BRollSuggestionDetailView(
                    suggestion: suggestion,
                    onAccept: {
                        acceptSuggestion(suggestion)
                    },
                    onReject: {
                        rejectSuggestion(suggestion)
                    }
                )
                .frame(height: 150)
                .background(Color(NSColor.controlBackgroundColor))
            }
        }
    }
    
    // MARK: - B-Roll Inspector Panel
    private var brollInspectorPanel: some View {
        VStack(spacing: 0) {
            Text("B-Roll Inspector")
                .font(.headline)
                .padding()
            
            Divider()
            
            if let firstSelectedId = selectedBRollClips.first,
               let clip = viewModel.brollClips.first(where: { $0.id == firstSelectedId }) {
                
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        // Clip Preview
                        BRollClipPreview(clip: clip)
                            .frame(height: 150)
                        
                        // Clip Information
                        GroupBox("Information") {
                            VStack(alignment: .leading, spacing: 8) {
                                InfoRow(label: "Name", value: clip.name)
                                InfoRow(label: "Duration", value: formatDuration(clip.duration ?? 0))
                                InfoRow(label: "Category", value: clip.category)
                                InfoRow(label: "Tags", value: clip.tags.joined(separator: ", "))
                                InfoRow(label: "Confidence", value: "\(Int(clip.relevanceScore * 100))%")
                            }
                            .frame(maxWidth: .infinity)
                        }
                        
                        // AI Analysis
                        GroupBox("AI Analysis") {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Scene Type: \(clip.sceneType)")
                                    .font(.caption)
                                
                                Text("Motion: \(clip.motionIntensity)")
                                    .font(.caption)
                                
                                Text("Color Palette: \(clip.dominantColors.joined(separator: ", "))")
                                    .font(.caption)
                                
                                if !clip.detectedObjects.isEmpty {
                                    Text("Objects: \(clip.detectedObjects.joined(separator: ", "))")
                                        .font(.caption)
                                }
                            }
                            .frame(maxWidth: .infinity)
                        }
                        
                        // Suggested Placement
                        GroupBox("Suggested Placement") {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(clip.suggestedPlacements, id: \.self) { placement in
                                    HStack {
                                        Image(systemName: "arrow.right.circle")
                                            .foregroundColor(.blue)
                                        Text(formatTimecode(placement))
                                            .font(.system(.caption, design: .monospaced))
                                        Spacer()
                                        Button("Insert") {
                                            insertBRollAt(clip: clip, time: placement)
                                        }
                                        .buttonStyle(BorderlessButtonStyle())
                                    }
                                }
                            }
                            .frame(maxWidth: .infinity)
                        }
                        
                        // Actions
                        VStack(spacing: 8) {
                            Button("Insert at Playhead") {
                                insertBRollAtPlayhead(clip: clip)
                            }
                            .buttonStyle(BorderedProminentButtonStyle())
                            .frame(maxWidth: .infinity)
                            
                            Button("Replace Selection") {
                                replaceSelectionWithBRoll(clip: clip)
                            }
                            .buttonStyle(BorderedButtonStyle())
                            .frame(maxWidth: .infinity)
                        }
                    }
                    .padding()
                }
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private var filteredBRollClips: [BRollClip] {
        viewModel.brollClips
            .filter { clip in
                (selectedCategory == "All" || clip.category == selectedCategory) &&
                (searchText.isEmpty || clip.matchesSearch(searchText))
            }
            .sorted { lhs, rhs in
                switch sortOrder {
                case .relevance:
                    return lhs.relevanceScore > rhs.relevanceScore
                case .duration:
                    return lhs.duration > rhs.duration
                case .recent:
                    return lhs.dateAdded > rhs.dateAdded
                case .name:
                    return lhs.name < rhs.name
                }
            }
    }
    
    private func toggleSelection(_ clip: BRollClip) {
        if selectedBRollClips.contains(clip.id) {
            selectedBRollClips.remove(clip.id)
        } else {
            selectedBRollClips.insert(clip.id)
        }
    }
    
    private func contextMenuForClip(_ clip: BRollClip) -> some View {
        Group {
            Button("Preview") {
                previewClip = clip
                showPreview = true
            }
            
            Button("Insert at Playhead") {
                insertBRollAtPlayhead(clip: clip)
            }
            
            Divider()
            
            Button("Reveal in Finder") {
                NSWorkspace.shared.selectFile(clip.url.path, inFileViewerRootedAtPath: "")
            }
            
            Button("Remove from Library") {
                viewModel.removeClip(clip)
            }
        }
    }
    
    private func applySelectedBRoll() {
        let clips = viewModel.brollClips.filter { selectedBRollClips.contains($0.id) }
        viewModel.applyBRollToTimeline(clips: clips, timeline: timeline)
        selectedBRollClips.removeAll()
        
        telemetry.addMessage(.info, "Applied \(clips.count) B-roll clips to timeline")
    }
    
    private func acceptSuggestion(_ suggestion: BRollSuggestion) {
        viewModel.acceptSuggestion(suggestion, timeline: timeline)
        telemetry.addMessage(.info, "Accepted B-roll suggestion at \(formatTimecode(suggestion.timeRange.lowerBound))")
    }
    
    private func rejectSuggestion(_ suggestion: BRollSuggestion) {
        viewModel.rejectSuggestion(suggestion)
        telemetry.addMessage(.info, "Rejected B-roll suggestion")
    }
    
    private func insertBRollAt(clip: BRollClip, time: TimeInterval) {
        viewModel.insertBRollClip(clip, at: time, timeline: timeline)
        telemetry.addMessage(.info, "Inserted \(clip.name) at \(formatTimecode(time))")
    }
    
    private func insertBRollAtPlayhead(clip: BRollClip) {
        let currentTime = timeline.playheadPosition
        insertBRollAt(clip: clip, time: currentTime)
    }
    
    private func replaceSelectionWithBRoll(clip: BRollClip) {
        if let selection = timeline.selectedTimeRange {
            if selection.start > 0 || selection.end > 0 {
                let range = selection.start...selection.end
                viewModel.replacetimeRange(range, with: clip, timeline: timeline)
                telemetry.addMessage(.info, "Replaced selection with \(clip.name)")
            }
        }
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.unitsStyle = .positional
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: duration) ?? "0:00"
    }
    
    private func formatTimecode(_ time: TimeInterval) -> String {
        let hours = Int(time) / 3600
        let minutes = (Int(time) % 3600) / 60
        let seconds = Int(time) % 60
        let frames = Int((time - Double(Int(time))) * 30)
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, seconds, frames)
    }
}

// MARK: - Supporting Views

struct BRollThumbnailView: View {
    let clip: BRollClip
    let isSelected: Bool
    let onTap: () -> Void
    let onDoubleTap: () -> Void
    
    @State private var thumbnail: NSImage?
    @State private var isHovering = false
    
    public var body: some View {
        VStack(spacing: 4) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.black)
                    .frame(height: 80)
                
                if let thumbnail = thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(height: 80)
                        .clipped()
                        .cornerRadius(8)
                } else {
                    ProgressView()
                        .scaleEffect(0.5)
                }
                
                // Selection Overlay
                if isSelected {
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.accentColor, lineWidth: 3)
                }
                
                // Hover Overlay
                if isHovering {
                    VStack {
                        Spacer()
                        HStack {
                            Image(systemName: "play.circle.fill")
                                .foregroundColor(.white)
                                .font(.title2)
                            Spacer()
                            Text(formatDuration(clip.duration ?? 0))
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(.horizontal, 4)
                                .background(Color.black.opacity(0.5))
                                .cornerRadius(4)
                        }
                        .padding(8)
                    }
                }
                
                // Confidence Badge
                VStack {
                    HStack {
                        Spacer()
                        Text("\(Int(clip.relevanceScore * 100))%")
                            .font(.caption2)
                            .foregroundColor(.white)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 2)
                            .background(confidenceColor(clip.relevanceScore))
                            .cornerRadius(4)
                    }
                    Spacer()
                }
                .padding(4)
            }
            
            Text(clip.name)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
            
            Text(clip.category)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .onTapGesture(perform: onTap)
        .onTapGesture(count: 2, perform: onDoubleTap)
        .onHover { hovering in
            isHovering = hovering
        }
        .onAppear {
            loadThumbnail()
        }
    }
    
    private func loadThumbnail() {
        Task {
            let asset = AVAsset(url: clip.url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 160, height: 90)
            
            do {
                let cgImage = try generator.copyCGImage(at: .zero, actualTime: nil)
                await MainActor.run {
                    thumbnail = NSImage(cgImage: cgImage, size: NSSize(width: 160, height: 90))
                }
            } catch {
                print("Failed to generate thumbnail: \(error)")
            }
        }
    }
    
    private func confidenceColor(_ score: Double) -> Color {
        if score > 0.8 { return .green }
        if score > 0.6 { return .orange }
        return .red
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

struct TimelineBRollOverlay: View {
    let suggestions: [BRollSuggestion]
    let timeline: TimelineModel
    let onSuggestionTap: (BRollSuggestion) -> Void
    
    public var body: some View {
        GeometryReader { geometry in
            ForEach(suggestions) { suggestion in
                BRollSuggestionMarker(
                    suggestion: suggestion,
                    geometry: geometry,
                    timeline: timeline,
                    onTap: { onSuggestionTap(suggestion) }
                )
            }
        }
    }
}

struct BRollSuggestionMarker: View {
    let suggestion: BRollSuggestion
    let geometry: GeometryProxy
    let timeline: TimelineModel
    let onTap: () -> Void
    
    @State private var isHovering = false
    
    public var body: some View {
        let xPosition = CGFloat(suggestion.timeRange.lowerBound / timeline.duration) * geometry.size.width
        let width = CGFloat((suggestion.timeRange.upperBound - suggestion.timeRange.lowerBound) / timeline.duration) * geometry.size.width
        
        Rectangle()
            .fill(Color.blue.opacity(isHovering ? 0.4 : 0.2))
            .overlay(
                Rectangle()
                    .stroke(Color.blue, lineWidth: 2)
            )
            .frame(width: width, height: 30)
            .position(x: xPosition + width/2, y: geometry.size.height - 15)
            .onTapGesture(perform: onTap)
            .onHover { hovering in
                isHovering = hovering
            }
            .overlay(
                Text(suggestion.clipName)
                    .font(.caption2)
                    .foregroundColor(.white)
                    .lineLimit(1)
                    .padding(.horizontal, 4)
                    .position(x: xPosition + width/2, y: geometry.size.height - 15)
            )
    }
}

struct BRollSuggestionDetailView: View {
    let suggestion: BRollSuggestion
    let onAccept: () -> Void
    let onReject: () -> Void
    
    public var body: some View {
        HStack(spacing: 16) {
            // Thumbnail
            if let thumbnail = suggestion.thumbnail {
                Image(nsImage: thumbnail)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 120, height: 80)
                    .cornerRadius(8)
            }
            
            // Details
            VStack(alignment: .leading, spacing: 8) {
                Text(suggestion.clipName)
                    .font(.headline)
                
                Text("Reason: \(suggestion.reason)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                HStack {
                    Label("\(Int(suggestion.confidence * 100))% match", systemImage: "chart.bar.fill")
                        .font(.caption)
                        .foregroundColor(suggestion.confidence > 0.7 ? .green : .orange)
                    
                    Label(formatTimeRange(suggestion.timeRange), systemImage: "clock")
                        .font(.caption)
                }
            }
            
            Spacer()
            
            // Actions
            VStack(spacing: 8) {
                Button("Accept") {
                    onAccept()
                }
                .buttonStyle(BorderedProminentButtonStyle())
                
                Button("Reject") {
                    onReject()
                }
                .buttonStyle(BorderedButtonStyle())
            }
        }
        .padding()
    }
    
    private func formatTimeRange(_ range: ClosedRange<TimeInterval>) -> String {
        let start = formatTimecode(range.lowerBound)
        let end = formatTimecode(range.upperBound)
        return "\(start) - \(end)"
    }
    
    private func formatTimecode(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
}

struct BRollClipPreview: View {
    let clip: BRollClip
    @State private var player: AVPlayer?
    
    public var body: some View {
        VideoPlayer(player: player)
            .onAppear {
                player = AVPlayer(url: clip.url)
                player?.isMuted = true
                player?.play()
                
                // Loop playback
                NotificationCenter.default.addObserver(
                    forName: .AVPlayerItemDidPlayToEndTime,
                    object: player?.currentItem,
                    queue: .main
                ) { _ in
                    player?.seek(to: .zero)
                    player?.play()
                }
            }
            .onDisappear {
                player?.pause()
                player = nil
            }
    }
}

struct BRollPreviewSheet: View {
    let clip: BRollClip
    @Environment(\.dismiss) var dismiss
    
    public var body: some View {
        VStack {
            Text(clip.name)
                .font(.title2)
                .padding()
            
            BRollClipPreview(clip: clip)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            
            HStack {
                Button("Close") {
                    dismiss()
                }
                .keyboardShortcut(.escape)
            }
            .padding()
        }
        .frame(width: 800, height: 600)
    }
}

struct InfoRow: View {
    let label: String
    let value: String
    
    public var body: some View {
        HStack {
            Text(label + ":")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(width: 80, alignment: .trailing)
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
            Spacer()
        }
    }
}

struct TimelineVisualizationView: View {
    let timeline: TimelineModel
    
    public var body: some View {
        // Simplified timeline visualization
        GeometryReader { geometry in
            VStack(spacing: 2) {
                ForEach(timeline.videoTracks) { track in
                    TrackView(track: track, totalWidth: geometry.size.width)
                }
            }
            .padding()
        }
    }
    
    private func clipWidth(_ duration: TimeInterval, in totalWidth: CGFloat) -> CGFloat {
        CGFloat(duration / timeline.duration) * totalWidth
    }
}

struct TrackView: View {
    let track: UITimelineTrack
    let totalWidth: CGFloat
    
    public var body: some View {
        HStack(spacing: 2) {
            ForEach(track.clips) { clip in
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.blue.opacity(0.3))
                    .frame(width: CGFloat(clip.duration ?? 0) * (totalWidth / 120.0))
            }
            Spacer()
        }
        .frame(height: 20)
        .background(Color.gray.opacity(0.1))
    }
}
