import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

// MARK: - Professional Export Panel

public struct ExportPanel: View {
    @ObservedObject var timeline: TimelineModel
    @State private var selectedFormat: ExportFormat = .h264_mp4
    @State private var exportPreset: ExportPreset = .youtube1080p
    @State private var customSettings = CustomExportSettings()
    @State private var useCustomSettings = false
    @State private var exportRange: ExportRange = .entireTimeline
    @State private var customStartTime: TimeInterval = 0
    @State private var customEndTime: TimeInterval = 0
    @State private var outputDirectory: URL?
    @State private var filename = "AutoResolve_Export"
    @State private var showAdvancedOptions = false
    @State private var isExporting = false
    @State private var exportProgress: Double = 0
    @State private var estimatedTimeRemaining: TimeInterval = 0
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Export Settings")
                    .font(.headline)
                
                Spacer()
                
                Button(action: { showAdvancedOptions.toggle() }) {
                    Label(showAdvancedOptions ? "Hide Advanced" : "Show Advanced",
                          systemImage: "gearshape")
                }
                .buttonStyle(.bordered)
            }
            .padding()
            
            Divider()
            
            if isExporting {
                ExportProgressView(
                    progress: exportProgress,
                    timeRemaining: estimatedTimeRemaining,
                    onCancel: cancelExport
                )
            } else {
                ScrollView {
                    VStack(spacing: 20) {
                        // Format and Preset Selection
                        FormatSelectionSection(
                            selectedFormat: $selectedFormat,
                            exportPreset: $exportPreset,
                            useCustomSettings: $useCustomSettings
                        )
                        
                        Divider()
                        
                        // Custom Settings
                        if useCustomSettings {
                            CustomSettingsSection(settings: $customSettings)
                            Divider()
                        } else {
                            PresetDetailsSection(preset: exportPreset)
                            Divider()
                        }
                        
                        // Export Range
                        ExportRangeSection(
                            exportRange: $exportRange,
                            customStartTime: $customStartTime,
                            customEndTime: $customEndTime,
                            timeline: timeline
                        )
                        
                        Divider()
                        
                        // Output Settings
                        OutputSettingsSection(
                            outputDirectory: $outputDirectory,
                            filename: $filename
                        )
                        
                        if showAdvancedOptions {
                            Divider()
                            AdvancedExportOptions()
                        }
                    }
                    .padding()
                }
                
                Divider()
                
                // Export Controls
                ExportControlsSection(
                    selectedFormat: selectedFormat,
                    exportPreset: exportPreset,
                    customSettings: customSettings,
                    useCustomSettings: useCustomSettings,
                    exportRange: exportRange,
                    outputDirectory: outputDirectory,
                    filename: filename,
                    onStartExport: startExport
                )
            }
        }
        .frame(width: 400)
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private func startExport() {
        isExporting = true
        exportProgress = 0
        
        // Start export process
        Task {
            await performExport()
        }
    }
    
    private func cancelExport() {
        isExporting = false
        exportProgress = 0
    }
    
    private func performExport() async {
        // Simulate export progress
        for i in 0...100 {
            await MainActor.run {
                exportProgress = Double(i) / 100.0
                estimatedTimeRemaining = Double(100 - i) * 0.5 // 0.5 seconds per percent
            }
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }
        
        await MainActor.run {
            isExporting = false
            // Show completion notification
        }
    }
}

// MARK: - Format Selection Section

struct FormatSelectionSection: View {
    @Binding var selectedFormat: ExportFormat
    @Binding var exportPreset: ExportPreset
    @Binding var useCustomSettings: Bool
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Format & Quality")
                .font(.subheadline.bold())
            
            // Format picker
            LabeledField("Format") {
                Picker("", selection: $selectedFormat) {
                    ForEach(ExportFormat.allCases, id: \.self) { format in
                        Label(format.displayName, systemImage: format.icon)
                            .tag(format)
                    }
                }
                .pickerStyle(.menu)
            }
            
            // Preset vs Custom toggle
            Picker("Settings Type", selection: $useCustomSettings) {
                Text("Presets").tag(false)
                Text("Custom").tag(true)
            }
            .pickerStyle(SegmentedPickerStyle())
            
            // Preset selection (when using presets)
            if !useCustomSettings {
                LabeledField("Preset") {
                    Picker("", selection: $exportPreset) {
                        ForEach(ExportPreset.allCases, id: \.self) { preset in
                            Text(preset.displayName)
                                .tag(preset)
                        }
                    }
                    .pickerStyle(.menu)
                }
            }
        }
    }
}

// MARK: - Custom Settings Section

struct CustomSettingsSection: View {
    @Binding var settings: CustomExportSettings
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Custom Settings")
                .font(.subheadline.bold())
            
            // Video Settings
            GroupBox("Video") {
                VStack(spacing: 8) {
                    ResolutionPicker(resolution: $settings.resolution)
                    
                    LabeledField("Frame Rate") {
                        Picker("", selection: $settings.frameRate) {
                            ForEach([23.98, 24, 25, 29.97, 30, 50, 59.94, 60], id: \.self) { rate in
                                Text("\(String(format: "%.2f", rate)) fps")
                                    .tag(rate)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Codec") {
                        Picker("", selection: $settings.videoCodec) {
                            ForEach(VideoCodec.allCases, id: \.self) { codec in
                                Text(codec.displayName).tag(codec)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    BitrateControl(
                        mode: $settings.bitrateMode,
                        targetBitrate: $settings.targetBitrate,
                        maxBitrate: $settings.maxBitrate
                    )
                }
            }
            
            // Audio Settings
            GroupBox("Audio") {
                VStack(spacing: 8) {
                    LabeledField("Codec") {
                        Picker("", selection: $settings.audioCodec) {
                            ForEach(AudioCodec.allCases, id: \.self) { codec in
                                Text(codec.displayName).tag(codec)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Sample Rate") {
                        Picker("", selection: $settings.audioSampleRate) {
                            ForEach([44100, 48000, 96000], id: \.self) { rate in
                                Text("\(rate) Hz").tag(rate)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Channels") {
                        Picker("", selection: $settings.audioChannels) {
                            Text("Mono").tag(1)
                            Text("Stereo").tag(2)
                            Text("5.1 Surround").tag(6)
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Bitrate") {
                        HStack {
                            Slider(
                                value: Binding(
                                    get: { Double(settings.audioBitrate) },
                                    set: { settings.audioBitrate = Int($0) }
                                ),
                                in: 64...320,
                                step: 16
                            )
                            Text("\(settings.audioBitrate) kbps")
                                .monospacedDigit()
                                .frame(width: 70)
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Preset Details Section

struct PresetDetailsSection: View {
    let preset: ExportPreset
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Preset Details")
                .font(.subheadline.bold())
            
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    PresetDetailRow(label: "Resolution", value: preset.resolution)
                    PresetDetailRow(label: "Frame Rate", value: preset.frameRate)
                    PresetDetailRow(label: "Video Codec", value: preset.videoCodec)
                    PresetDetailRow(label: "Video Bitrate", value: preset.videoBitrate)
                    PresetDetailRow(label: "Audio Codec", value: preset.audioCodec)
                    PresetDetailRow(label: "Audio Bitrate", value: preset.audioBitrate)
                    PresetDetailRow(label: "File Size (1 min)", value: preset.estimatedFileSize)
                }
                .font(.caption)
            }
        }
    }
}

struct PresetDetailRow: View {
    let label: String
    let value: String
    
    public var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
        }
    }
}

// MARK: - Export Range Section

struct ExportRangeSection: View {
    @Binding var exportRange: ExportRange
    @Binding var customStartTime: TimeInterval
    @Binding var customEndTime: TimeInterval
    @ObservedObject var timeline: TimelineModel
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Export Range")
                .font(.subheadline.bold())
            
            Picker("Range", selection: $exportRange) {
                ForEach(ExportRange.allCases, id: \.self) { range in
                    Text(range.displayName).tag(range)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            
            switch exportRange {
            case .entireTimeline:
                Text("Duration: \(timeline.timecode(for: timeline.duration))")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
            case .workArea:
                Text("Work Area: \(timeline.timecode(for: timeline.workAreaStart)) - \(timeline.timecode(for: timeline.workAreaEnd))")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
            case .custom:
                VStack(spacing: 8) {
                    LabeledField("Start Time") {
                        TimecodeField(
                            time: $customStartTime,
                            timeline: timeline
                        )
                    }
                    
                    LabeledField("End Time") {
                        TimecodeField(
                            time: $customEndTime,
                            timeline: timeline
                        )
                    }
                    
                    Text("Duration: \(timeline.timecode(for: customEndTime - customStartTime))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }
}

// MARK: - Output Settings Section

struct OutputSettingsSection: View {
    @Binding var outputDirectory: URL?
    @Binding var filename: String
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Output Settings")
                .font(.subheadline.bold())
            
            // Output directory
            VStack(alignment: .leading, spacing: 4) {
                Text("Destination")
                    .font(.caption)
                
                HStack {
                    Text(outputDirectory?.path ?? "Not selected")
                        .lineLimit(1)
                        .foregroundColor(outputDirectory == nil ? .secondary : .primary)
                    
                    Spacer()
                    
                    Button("Browse...") {
                        selectOutputDirectory()
                    }
                    .buttonStyle(.bordered)
                }
                .padding(8)
                .background(Color(NSColor.textBackgroundColor))
                .cornerRadius(4)
            }
            
            // Filename
            LabeledField("Filename") {
                TextField("Export filename", text: $filename)
                    .textFieldStyle(.roundedBorder)
            }
            
            // Full path preview
            if let directory = outputDirectory {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Full Path")
                        .font(.caption)
                    
                    Text(directory.appendingPathComponent(filename).path)
                        .font(.caption.monospacedDigit())
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                        .padding(8)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(4)
                }
            }
        }
    }
    
    private func selectOutputDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        
        if panel.runModal() == .OK {
            outputDirectory = panel.url
        }
    }
}

// MARK: - Advanced Export Options

struct AdvancedExportOptions: View {
    @State private var useHardwareAcceleration = true
    @State private var includeAudio = true
    @State private var addTimecode = false
    @State private var embedMetadata = true
    @State private var colorSpace = "Rec. 709"
    @State private var encodeInBackground = false
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Advanced Options")
                .font(.subheadline.bold())
            
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Hardware Acceleration", isOn: $useHardwareAcceleration)
                Toggle("Include Audio", isOn: $includeAudio)
                Toggle("Add Timecode Track", isOn: $addTimecode)
                Toggle("Embed Metadata", isOn: $embedMetadata)
                Toggle("Encode in Background", isOn: $encodeInBackground)
            }
            
            LabeledField("Color Space") {
                Picker("", selection: $colorSpace) {
                    Text("Rec. 709").tag("Rec. 709")
                    Text("Rec. 2020").tag("Rec. 2020")
                    Text("sRGB").tag("sRGB")
                    Text("P3").tag("P3")
                }
                .pickerStyle(.menu)
            }
        }
    }
}

// MARK: - Export Controls Section

struct ExportControlsSection: View {
    let selectedFormat: ExportFormat
    let exportPreset: ExportPreset
    let customSettings: CustomExportSettings
    let useCustomSettings: Bool
    let exportRange: ExportRange
    let outputDirectory: URL?
    let filename: String
    let onStartExport: () -> Void
    
    public var body: some View {
        HStack {
            // Export summary
            VStack(alignment: .leading, spacing: 2) {
                Text(selectedFormat.displayName)
                    .font(.caption.bold())
                
                if useCustomSettings {
                    Text("\(customSettings.resolution) • \(String(format: "%.0f", customSettings.frameRate)) fps")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                } else {
                    Text(exportPreset.displayName)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            // Estimated file size
            VStack(alignment: .trailing, spacing: 2) {
                Text("Est. Size")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                Text(estimatedFileSize)
                    .font(.caption.bold())
            }
            
            // Export button
            Button("Export") {
                onStartExport()
            }
            .buttonStyle(.borderedProminent)
            .disabled(outputDirectory == nil || filename.isEmpty)
        }
        .padding()
    }
    
    private var estimatedFileSize: String {
        // Calculate estimated file size based on settings
        let durationMinutes = exportRange.duration(timeline: TimelineModel()) / 60
        let bitrateMbps = useCustomSettings ? 
            Double(customSettings.targetBitrate) / 1000 : 
            Double(exportPreset.videoBitrateValue) / 1000
        
        let estimatedMB = durationMinutes * bitrateMbps * 7.5 // Rough estimate
        
        if estimatedMB > 1000 {
            return String(format: "%.1f GB", estimatedMB / 1000)
        } else {
            return String(format: "%.0f MB", estimatedMB)
        }
    }
}

// MARK: - Export Progress View

struct ExportProgressView: View {
    let progress: Double
    let timeRemaining: TimeInterval
    let onCancel: () -> Void
    
    public var body: some View {
        VStack(spacing: 16) {
            Text("Exporting...")
                .font(.headline)
            
            VStack(spacing: 8) {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                
                HStack {
                    Text("\(Int(progress * 100))%")
                        .monospacedDigit()
                    
                    Spacer()
                    
                    Text("Time remaining: \(formatTimeRemaining(timeRemaining))")
                        .foregroundColor(.secondary)
                }
                .font(.caption)
            }
            
            Button("Cancel Export") {
                onCancel()
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }
    
    private func formatTimeRemaining(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return "\(minutes):\(String(format: "%02d", seconds))"
    }
}

// MARK: - Helper Views

struct ResolutionPicker: View {
    @Binding var resolution: String
    
    private let commonResolutions = [
        "3840×2160", "2560×1440", "1920×1080", 
        "1280×720", "854×480", "640×360"
    ]
    
    public var body: some View {
        LabeledField("Resolution") {
            Picker("", selection: $resolution) {
                ForEach(commonResolutions, id: \.self) { res in
                    Text(res).tag(res)
                }
                Text("Custom...").tag("Custom")
            }
            .pickerStyle(.menu)
        }
    }
}

struct BitrateControl: View {
    @Binding var mode: BitrateMode
    @Binding var targetBitrate: Int
    @Binding var maxBitrate: Int
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            LabeledField("Bitrate Mode") {
                Picker("", selection: $mode) {
                    ForEach(BitrateMode.allCases, id: \.self) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .pickerStyle(.menu)
            }
            
            switch mode {
            case .constant:
                LabeledField("Bitrate") {
                    HStack {
                        Slider(
                            value: Binding(
                                get: { Double(targetBitrate) },
                                set: { targetBitrate = Int($0) }
                            ),
                            in: 1000...50000,
                            step: 500
                        )
                        Text("\(targetBitrate) kbps")
                            .monospacedDigit()
                            .frame(width: 80)
                    }
                }
                
            case .variable:
                VStack(spacing: 4) {
                    LabeledField("Target") {
                        HStack {
                            Slider(
                                value: Binding(
                                    get: { Double(targetBitrate) },
                                    set: { targetBitrate = Int($0) }
                                ),
                                in: 1000...50000,
                                step: 500
                            )
                            Text("\(targetBitrate) kbps")
                                .monospacedDigit()
                                .frame(width: 80)
                        }
                    }
                    
                    LabeledField("Maximum") {
                        HStack {
                            Slider(
                                value: Binding(
                                    get: { Double(maxBitrate) },
                                    set: { maxBitrate = Int($0) }
                                ),
                                in: Double(targetBitrate)...100000,
                                step: 1000
                            )
                            Text("\(maxBitrate) kbps")
                                .monospacedDigit()
                                .frame(width: 80)
                        }
                    }
                }
            }
        }
    }
}

struct TimecodeField: View {
    @Binding var time: TimeInterval
    @ObservedObject var timeline: TimelineModel
    
    public var body: some View {
        TextField("00:00:00:00", text: Binding(
            get: { timeline.timecode(for: time) },
            set: { newValue in
                // Parse timecode string back to time
                time = parseTimecode(newValue)
            }
        ))
        .textFieldStyle(.roundedBorder)
        .font(.system(.body, design: .monospaced))
    }
    
    private func parseTimecode(_ timecode: String) -> TimeInterval {
        // Basic timecode parsing (HH:MM:SS:FF)
        let components = timecode.split(separator: ":").compactMap { Double($0) }
        guard components.count >= 3 else { return time }
        
        let hours = components[0]
        let minutes = components[1] 
        let seconds = components[2]
        let frames = components.count > 3 ? components[3] : 0
        
        let hoursInSeconds = TimeInterval(hours * 3600)
        let minutesInSeconds = TimeInterval(minutes * 60)
        let secondsValue = TimeInterval(seconds)
        let framesInSeconds = TimeInterval(frames) / TimeInterval(timeline.frameRate)
        return hoursInSeconds + minutesInSeconds + secondsValue + framesInSeconds
    }
}

// MARK: - Export Models

// Use global ExportFormat defined in Core/swift

public enum ExportPreset: String, CaseIterable {
    case youtube4k = "youtube4k"
    case youtube1080p = "youtube1080p"
    case youtube720p = "youtube720p"
    case tiktok = "tiktok"
    case instagram = "instagram"
    case broadcast = "broadcast"
    case archive = "archive"
    case web = "web"
    
    var displayName: String {
        switch self {
        case .youtube4k: return "YouTube 4K"
        case .youtube1080p: return "YouTube 1080p"
        case .youtube720p: return "YouTube 720p"
        case .tiktok: return "TikTok/Shorts"
        case .instagram: return "Instagram"
        case .broadcast: return "Broadcast"
        case .archive: return "Archive Quality"
        case .web: return "Web Optimized"
        }
    }
    
    var resolution: String {
        switch self {
        case .youtube4k: return "3840×2160"
        case .youtube1080p, .instagram, .broadcast, .archive, .web: return "1920×1080"
        case .youtube720p: return "1280×720"
        case .tiktok: return "1080×1920"
        }
    }
    
    var frameRate: String {
        switch self {
        case .youtube4k, .youtube1080p, .youtube720p, .web: return "30 fps"
        case .tiktok, .instagram: return "30 fps"
        case .broadcast: return "29.97 fps"
        case .archive: return "24 fps"
        }
    }
    
    var videoCodec: String {
        switch self {
        case .youtube4k, .youtube1080p, .youtube720p, .tiktok, .instagram, .web: return "H.264"
        case .broadcast, .archive: return "ProRes 422"
        }
    }
    
    var videoBitrate: String {
        switch self {
        case .youtube4k: return "45 Mbps"
        case .youtube1080p: return "15 Mbps"
        case .youtube720p: return "8 Mbps"
        case .tiktok, .instagram: return "12 Mbps"
        case .web: return "5 Mbps"
        case .broadcast, .archive: return "147 Mbps"
        }
    }
    
    var videoBitrateValue: Int {
        switch self {
        case .youtube4k: return 45000
        case .youtube1080p: return 15000
        case .youtube720p: return 8000
        case .tiktok, .instagram: return 12000
        case .web: return 5000
        case .broadcast, .archive: return 147000
        }
    }
    
    var audioCodec: String {
        switch self {
        case .youtube4k, .youtube1080p, .youtube720p, .tiktok, .instagram, .web: return "AAC"
        case .broadcast, .archive: return "PCM"
        }
    }
    
    var audioBitrate: String {
        switch self {
        case .youtube4k, .youtube1080p, .youtube720p, .tiktok, .instagram, .web: return "128 kbps"
        case .broadcast, .archive: return "1411 kbps"
        }
    }
    
    var estimatedFileSize: String {
        switch self {
        case .youtube4k: return "~337 MB/min"
        case .youtube1080p: return "~112 MB/min"
        case .youtube720p: return "~60 MB/min"
        case .tiktok, .instagram: return "~90 MB/min"
        case .web: return "~37 MB/min"
        case .broadcast, .archive: return "~1.1 GB/min"
        }
    }
}

public enum ExportRange: String, CaseIterable {
    case entireTimeline = "entire"
    case workArea = "work_area"
    case custom = "custom"
    
    var displayName: String {
        switch self {
        case .entireTimeline: return "Entire Timeline"
        case .workArea: return "Work Area"
        case .custom: return "Custom Range"
        }
    }
    
    func duration(timeline: TimelineModel) -> TimeInterval {
        switch self {
        case .entireTimeline: return timeline.duration
        case .workArea: return timeline.workAreaEnd - timeline.workAreaStart
        case .custom: return 0 // Will be calculated from custom times
        }
    }
}

public struct CustomExportSettings {
    var resolution = "1920×1080"
    var frameRate: Double = 30
    var videoCodec = VideoCodec.h264
    var bitrateMode = BitrateMode.variable
    var targetBitrate = 15000
    var maxBitrate = 30000
    var audioCodec = AudioCodec.aac
    var audioSampleRate = 48000
    var audioChannels = 2
    var audioBitrate = 128
}

public enum VideoCodec: String, CaseIterable {
    case h264 = "h264"
    case h265 = "h265"
    case prores422 = "prores422"
    case prores4444 = "prores4444"
    case dnxhd = "dnxhd"
    case dnxhr = "dnxhr"
    
    var displayName: String {
        switch self {
        case .h264: return "H.264"
        case .h265: return "H.265 (HEVC)"
        case .prores422: return "ProRes 422"
        case .prores4444: return "ProRes 4444"
        case .dnxhd: return "DNxHD"
        case .dnxhr: return "DNxHR"
        }
    }
}

public enum AudioCodec: String, CaseIterable {
    case aac = "aac"
    case pcm = "pcm"
    case mp3 = "mp3"
    case flac = "flac"
    
    var displayName: String {
        switch self {
        case .aac: return "AAC"
        case .pcm: return "PCM (Uncompressed)"
        case .mp3: return "MP3"
        case .flac: return "FLAC"
        }
    }
}

public enum BitrateMode: String, CaseIterable {
    case constant = "constant"
    case variable = "variable"
    
    var displayName: String {
        switch self {
        case .constant: return "Constant (CBR)"
        case .variable: return "Variable (VBR)"
        }
    }
}
