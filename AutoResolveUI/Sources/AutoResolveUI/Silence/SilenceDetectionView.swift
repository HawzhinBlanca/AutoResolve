import SwiftUI
import AVFoundation
import Accelerate

// MARK: - Silence Detection Main View
public struct SilenceDetectionView: View {
    @StateObject private var viewModel = SilenceDetectionViewModel()
    @EnvironmentObject var timeline: TimelineModel
    @EnvironmentObject var telemetry: PipelineStatusMonitor
    
    @State private var selectedSilenceRegion: SilenceRegion?
    @State private var showAdvancedSettings = false
    @State private var isProcessing = false
    @State private var autoRemoveSilence = true
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header Controls
            headerControls
            
            Divider()
            
            // Main Content
            HSplitView {
                // Left: Waveform with Silence Overlay
                waveformPanel
                    .frame(minWidth: 400)
                
                // Right: Silence Regions List
                silenceRegionsPanel
                    .frame(width: 350)
            }
            
            // Bottom: Action Bar
            actionBar
        }
        .onAppear {
            if let firstVideoClip = timeline.videoTracks.first?.clips.first,
               let url = firstVideoClip.sourceURL {
                viewModel.loadAudioFromVideo(url: url)
            }
        }
    }
    
    // MARK: - Header Controls
    private var headerControls: some View {
        HStack {
            Text("Silence Detection")
                .font(.headline)
            
            Spacer()
            
            // Threshold Controls
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Threshold")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    HStack {
                        Slider(value: $viewModel.silenceThresholdDB, in: -60...0)
                            .frame(width: 100)
                        Text("\(Int(viewModel.silenceThresholdDB)) dB")
                            .font(.caption)
                            .monospacedDigit()
                            .frame(width: 50)
                    }
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Min Duration")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    HStack {
                        Slider(value: $viewModel.minSilenceDuration, in: 0.1...2.0)
                            .frame(width: 100)
                        Text("\(viewModel.minSilenceDuration, specifier: "%.1f")s")
                            .font(.caption)
                            .monospacedDigit()
                            .frame(width: 40)
                    }
                }
                
                Toggle("Auto-Remove", isOn: $autoRemoveSilence)
                    .toggleStyle(SwitchToggleStyle())
                
                Button(action: { showAdvancedSettings.toggle() }) {
                    Image(systemName: "gearshape")
                }
                .buttonStyle(BorderlessButtonStyle())
            }
            
            Button("Detect Silence") {
                detectSilence()
            }
            .buttonStyle(BorderedProminentButtonStyle())
            .disabled(isProcessing || viewModel.audioURL == nil)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .sheet(isPresented: $showAdvancedSettings) {
            SilenceDetectionSettingsView(viewModel: viewModel)
        }
    }
    
    // MARK: - Waveform Panel
    private var waveformPanel: some View {
        VStack(spacing: 0) {
            // Waveform Display
            ZStack {
                // Background
                Rectangle()
                    .fill(Color.black.opacity(0.05))
                
                // Waveform Visualization
                SilenceWaveformView(
                    audioURL: viewModel.audioURL,
                    waveformData: viewModel.waveformData,
                    silenceRegions: viewModel.detectedSilenceRegions,
                    selectedRegion: selectedSilenceRegion
                )
                
                // Timeline Overlay
                TimelineRulerOverlay(duration: viewModel.audioDuration)
                
                // Processing Overlay
                if isProcessing {
                    ZStack {
                        Rectangle()
                            .fill(Color.black.opacity(0.5))
                        
                        VStack(spacing: 12) {
                            ProgressView()
                                .scaleEffect(1.5)
                            
                            Text("Analyzing audio...")
                                .foregroundColor(.white)
                            
                            Text("\(Int(viewModel.processingProgress * 100))%")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.8))
                        }
                    }
                }
            }
            .frame(maxHeight: .infinity)
            
            // Zoom Controls
            HStack {
                Button(action: { viewModel.zoomLevel = max(0.5, viewModel.zoomLevel - 0.25) }) {
                    Image(systemName: "minus.magnifyingglass")
                }
                .buttonStyle(BorderlessButtonStyle())
                
                Slider(value: $viewModel.zoomLevel, in: 0.5...4.0)
                    .frame(width: 100)
                
                Button(action: { viewModel.zoomLevel = min(4.0, viewModel.zoomLevel + 0.25) }) {
                    Image(systemName: "plus.magnifyingglass")
                }
                .buttonStyle(BorderlessButtonStyle())
                
                Spacer()
                
                Text("Duration: \(formatDuration(viewModel.audioDuration))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(8)
            .background(Color(NSColor.controlBackgroundColor))
        }
    }
    
    // MARK: - Silence Regions Panel
    private var silenceRegionsPanel: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Detected Silence Regions")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text("\(viewModel.detectedSilenceRegions.count)")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(0.2))
                    .cornerRadius(8)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Regions List
            if viewModel.detectedSilenceRegions.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "waveform")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    
                    Text("No silence detected")
                        .foregroundColor(.secondary)
                    
                    Text("Click 'Detect Silence' to analyze")
                        .font(.caption)
                        .foregroundColor(.secondary.opacity(0.7))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(selection: $selectedSilenceRegion) {
                    ForEach(viewModel.detectedSilenceRegions) { region in
                        SilenceRegionRow(
                            region: region,
                            isSelected: selectedSilenceRegion?.id == region.id,
                            onToggle: {
                                viewModel.toggleRegionSelection(region)
                            },
                            onPreview: {
                                previewRegion(region)
                            }
                        )
                    }
                }
                .listStyle(PlainListStyle())
            }
            
            // Summary
            if !viewModel.detectedSilenceRegions.isEmpty {
                VStack(spacing: 8) {
                    Divider()
                    
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Total Silence")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(formatDuration(viewModel.totalSilenceDuration))
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .trailing, spacing: 4) {
                            Text("Percentage")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(Int(viewModel.silencePercentage))%")
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                }
                .background(Color(NSColor.controlBackgroundColor))
            }
        }
    }
    
    // MARK: - Action Bar
    private var actionBar: some View {
        HStack {
            // Selection Actions
            HStack(spacing: 8) {
                Button("Select All") {
                    viewModel.selectAllRegions()
                }
                .disabled(viewModel.detectedSilenceRegions.isEmpty)
                
                Button("Select None") {
                    viewModel.deselectAllRegions()
                }
                .disabled(viewModel.selectedRegions.isEmpty)
                
                Divider()
                    .frame(height: 20)
                
                Text("\(viewModel.selectedRegions.count) selected")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Main Actions
            HStack(spacing: 12) {
                Button("Export EDL") {
                    exportEDL()
                }
                .disabled(viewModel.detectedSilenceRegions.isEmpty)
                
                Button("Remove Selected") {
                    removeSelectedSilence()
                }
                .disabled(viewModel.selectedRegions.isEmpty)
                .buttonStyle(BorderedButtonStyle())
                
                Button("Apply to Timeline") {
                    applyToTimeline()
                }
                .disabled(viewModel.detectedSilenceRegions.isEmpty)
                .buttonStyle(BorderedProminentButtonStyle())
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    // MARK: - Actions
    
    private func detectSilence() {
        isProcessing = true
        telemetry.currentStatus = .detectingSilence
        
        Task {
            await viewModel.detectSilence()
            
            await MainActor.run {
                isProcessing = false
                telemetry.currentStatus = .idle
                telemetry.addMessage(.info, "Detected \(viewModel.detectedSilenceRegions.count) silence regions")
                
                if autoRemoveSilence && !viewModel.detectedSilenceRegions.isEmpty {
                    removeSelectedSilence()
                }
            }
        }
    }
    
    private func removeSelectedSilence() {
        let selectedCount = viewModel.selectedRegions.count
        viewModel.removeSelectedSilenceFromTimeline(timeline: timeline)
        
        telemetry.addMessage(.info, "Removed \(selectedCount) silence regions from timeline")
    }
    
    private func applyToTimeline() {
        viewModel.applySilenceDetectionToTimeline(timeline: timeline)
        telemetry.addMessage(.info, "Applied silence detection to timeline")
    }
    
    private func previewRegion(_ region: SilenceRegion) {
        // Preview the silence region in the timeline
        timeline.playheadPosition = region.startTime
        timeline.selectedTimeRange = (start: region.startTime, end: region.endTime)
    }
    
    private func exportEDL() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "silence_regions.edl"
        
        if panel.runModal() == .OK, let url = panel.url {
            viewModel.exportEDL(to: url)
            telemetry.addMessage(.info, "Exported EDL to \(url.lastPathComponent)")
        }
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.unitsStyle = .positional
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: duration) ?? "0:00"
    }
}

// MARK: - Waveform View

struct SilenceWaveformView: View {
    let audioURL: URL?
    let waveformData: [Float]
    let silenceRegions: [SilenceRegion]
    let selectedRegion: SilenceRegion?
    
    @State private var hoveredRegion: SilenceRegion?
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Waveform
                if !waveformData.isEmpty {
                    WaveformShape(samples: waveformData)
                        .stroke(Color.blue.opacity(0.8), lineWidth: 1)
                }
                
                // Silence Regions
                ForEach(silenceRegions) { region in
                    SilenceRegionOverlay(
                        region: region,
                        totalDuration: waveformData.count > 0 ? Double(waveformData.count) / 44100.0 : 1.0,
                        geometry: geometry,
                        isSelected: selectedRegion?.id == region.id || region.isSelected,
                        isHovered: hoveredRegion?.id == region.id
                    )
                    .onHover { hovering in
                        hoveredRegion = hovering ? region : nil
                    }
                }
            }
        }
    }
}

struct WaveformShape: Shape {
    let samples: [Float]
    
    func path(in rect: CGRect) -> Path {
        var path = Path()
        
        guard !samples.isEmpty else { return path }
        
        let width = rect.width
        let height = rect.height
        let midY = height / 2
        let sampleCount = samples.count
        let samplesPerPixel = max(1, sampleCount / Int(width))
        
        for x in 0..<Int(width) {
            let sampleIndex = x * samplesPerPixel
            guard sampleIndex < sampleCount else { break }
            
            // Get peak value for this pixel
            let endIndex = min(sampleIndex + samplesPerPixel, sampleCount)
            let slice = samples[sampleIndex..<endIndex]
            let peak = slice.max() ?? 0
            
            let amplitude = CGFloat(peak) * height / 2
            
            if x == 0 {
                path.move(to: CGPoint(x: CGFloat(x), y: midY))
            }
            
            path.addLine(to: CGPoint(x: CGFloat(x), y: midY - amplitude))
            path.addLine(to: CGPoint(x: CGFloat(x), y: midY + amplitude))
            path.move(to: CGPoint(x: CGFloat(x), y: midY))
        }
        
        return path
    }
}

struct SilenceRegionOverlay: View {
    let region: SilenceRegion
    let totalDuration: TimeInterval
    let geometry: GeometryProxy
    let isSelected: Bool
    let isHovered: Bool
    
    public var body: some View {
        let startX = CGFloat(region.startTime / totalDuration) * geometry.size.width
        let width = CGFloat(region.duration / totalDuration) * geometry.size.width
        
        Rectangle()
            .fill(Color.red.opacity(isSelected ? 0.3 : (isHovered ? 0.2 : 0.1)))
            .overlay(
                Rectangle()
                    .stroke(Color.red.opacity(isSelected ? 0.8 : 0.4), lineWidth: isSelected ? 2 : 1)
            )
            .frame(width: width, height: geometry.size.height)
            .position(x: startX + width/2, y: geometry.size.height/2)
    }
}

struct TimelineRulerOverlay: View {
    let duration: TimeInterval
    
    public var body: some View {
        GeometryReader { geometry in
            VStack {
                HStack(alignment: .top, spacing: 0) {
                    ForEach(0..<(duration.isFinite && duration >= 0 ? Int(min(duration, 10000)) + 1 : 1), id: \.self) { second in
                        VStack {
                            Text(formatTime(TimeInterval(second)))
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                            
                            Rectangle()
                                .fill(Color.secondary.opacity(0.3))
                                .frame(width: 1, height: 5)
                        }
                        .frame(width: geometry.size.width / CGFloat(duration), alignment: .leading)
                    }
                }
                Spacer()
            }
        }
    }
    
    private func formatTime(_ time: TimeInterval) -> String {
        guard time.isFinite && time >= 0 else { return "0:00" }
        let safeTime = min(time, 359999.0)
        let minutes = Int(safeTime) / 60
        let seconds = Int(safeTime) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

// MARK: - Silence Region Row

struct SilenceRegionRow: View {
    let region: SilenceRegion
    let isSelected: Bool
    let onToggle: () -> Void
    let onPreview: () -> Void
    
    public var body: some View {
        HStack {
            // Selection Checkbox
            Button(action: onToggle) {
                Image(systemName: region.isSelected ? "checkmark.square.fill" : "square")
                    .foregroundColor(region.isSelected ? .accentColor : .secondary)
            }
            .buttonStyle(BorderlessButtonStyle())
            
            // Region Info
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(formatTimecode(region.startTime))
                        .font(.system(.caption, design: .monospaced))
                    
                    Image(systemName: "arrow.right")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    Text(formatTimecode(region.endTime))
                        .font(.system(.caption, design: .monospaced))
                }
                
                HStack {
                    Text("Duration: \(formatDuration(region.duration))")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    if region.averageLevel < -40 {
                        Label("Very Quiet", systemImage: "speaker.slash.fill")
                            .font(.caption2)
                            .foregroundColor(.orange)
                    }
                }
            }
            
            Spacer()
            
            // Preview Button
            Button(action: onPreview) {
                Image(systemName: "play.circle")
            }
            .buttonStyle(BorderlessButtonStyle())
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .cornerRadius(4)
    }
    
    private func formatTimecode(_ time: TimeInterval) -> String {
        guard time.isFinite && time >= 0 else { return "00:00:00" }
        let safeTime = min(time, 359999.0)
        let minutes = Int(safeTime) / 60
        let seconds = Int(safeTime) % 60
        let frames = Int((safeTime - Double(Int(safeTime))) * 30)
        return String(format: "%02d:%02d:%02d", minutes, seconds, frames)
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        return String(format: "%.2fs", duration)
    }
}

// MARK: - Settings Sheet

struct SilenceDetectionSettingsView: View {
    @ObservedObject var viewModel: SilenceDetectionViewModel
    @Environment(\.dismiss) var dismiss
    
    public var body: some View {
        VStack(spacing: 20) {
            Text("Advanced Silence Detection Settings")
                .font(.headline)
            
            Form {
                Section("Detection Parameters") {
                    HStack {
                        Text("Silence Threshold:")
                        Slider(value: $viewModel.silenceThresholdDB, in: -60...0)
                        Text("\(Int(viewModel.silenceThresholdDB)) dB")
                            .monospacedDigit()
                            .frame(width: 60)
                    }
                    
                    HStack {
                        Text("Minimum Duration:")
                        Slider(value: $viewModel.minSilenceDuration, in: 0.1...5.0)
                        Text("\(viewModel.minSilenceDuration, specifier: "%.1f") sec")
                            .monospacedDigit()
                            .frame(width: 60)
                    }
                    
                    HStack {
                        Text("Padding Before:")
                        Slider(value: $viewModel.paddingBefore, in: 0...1.0)
                        Text("\(viewModel.paddingBefore, specifier: "%.2f") sec")
                            .monospacedDigit()
                            .frame(width: 60)
                    }
                    
                    HStack {
                        Text("Padding After:")
                        Slider(value: $viewModel.paddingAfter, in: 0...1.0)
                        Text("\(viewModel.paddingAfter, specifier: "%.2f") sec")
                            .monospacedDigit()
                            .frame(width: 60)
                    }
                }
                
                Section("Analysis Options") {
                    Toggle("Use RMS Analysis", isOn: $viewModel.useRMSAnalysis)
                    Toggle("Adaptive Threshold", isOn: $viewModel.useAdaptiveThreshold)
                    Toggle("Merge Adjacent Regions", isOn: $viewModel.mergeAdjacentRegions)
                    
                    if viewModel.mergeAdjacentRegions {
                        HStack {
                            Text("Merge Gap:")
                            Slider(value: $viewModel.mergeGapThreshold, in: 0.1...2.0)
                            Text("\(viewModel.mergeGapThreshold, specifier: "%.1f") sec")
                                .monospacedDigit()
                                .frame(width: 60)
                        }
                    }
                }
                
                Section("Processing") {
                    Picker("Quality", selection: $viewModel.processingQuality) {
                        Text("Fast").tag(ProcessingQuality.fast)
                        Text("Balanced").tag(ProcessingQuality.balanced)
                        Text("High").tag(ProcessingQuality.high)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    
                    Toggle("Real-time Preview", isOn: $viewModel.realtimePreview)
                }
            }
            .padding()
            
            HStack {
                Button("Reset Defaults") {
                    viewModel.resetToDefaults()
                }
                
                Spacer()
                
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.escape)
                
                Button("Apply") {
                    dismiss()
                }
                .buttonStyle(BorderedProminentButtonStyle())
                .keyboardShortcut(.return)
            }
        }
        .padding()
        .frame(width: 500, height: 450)
    }
}
