import SwiftUI
import AVFoundation

// MARK: - Project Settings Inspector

public struct ProjectInspector: View {
    @ObservedObject var project: VideoProject
    @State private var selectedTab: ProjectTab = .general
    @State private var showAdvancedSettings = false
    
    enum ProjectTab: String, CaseIterable {
        case general = "General"
        case video = "Video"
        case audio = "Audio"
        case performance = "Performance"
        case collaboration = "Collaboration"
        
        var icon: String {
            switch self {
            case .general: return "gearshape"
            case .video: return "video"
            case .audio: return "speaker.wave.3"
            case .performance: return "speedometer"
            case .collaboration: return "person.2"
            }
        }
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Project Settings")
                    .font(.headline)
                
                Spacer()
                
                Button(action: { showAdvancedSettings.toggle() }) {
                    Label(showAdvancedSettings ? "Hide Advanced" : "Show Advanced",
                          systemImage: "slider.horizontal.3")
                }
                .buttonStyle(.bordered)
            }
            .padding()
            
            Divider()
            
            // Tab selector
            Picker("Settings", selection: $selectedTab) {
                ForEach(ProjectTab.allCases, id: \.self) { tab in
                    Label(tab.rawValue, systemImage: tab.icon)
                        .tag(tab)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding(.horizontal)
            .padding(.vertical, 8)
            
            Divider()
            
            // Content
            ScrollView {
                VStack(spacing: 16) {
                    switch selectedTab {
                    case .general:
                        GeneralSettingsView(project: project)
                    case .video:
                        VideoSettingsView(project: project, showAdvanced: showAdvancedSettings)
                    case .audio:
                        AudioSettingsView(project: project, showAdvanced: showAdvancedSettings)
                    case .performance:
                        PerformanceSettingsView(project: project, showAdvanced: showAdvancedSettings)
                    case .collaboration:
                        CollaborationSettingsView(project: project)
                    }
                }
                .padding()
            }
        }
        .frame(width: 400)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

// MARK: - General Settings View

struct GeneralSettingsView: View {
    @ObservedObject var project: VideoProject
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Project Info
            GroupBox("Project Information") {
                VStack(spacing: 12) {
                    LabeledField("Name") {
                        TextField("Project Name", text: $project.name)
                            .textFieldStyle(.roundedBorder)
                    }
                    
                    LabeledField("Description") {
                        TextEditor(text: Binding(
                            get: { project.metadata.notes },
                            set: { project.metadata.notes = $0 }
                        ))
                        .frame(height: 60)
                        .scrollContentBackground(.hidden)
                        .background(Color(NSColor.textBackgroundColor))
                        .cornerRadius(4)
                    }
                    
                    LabeledField("Creator") {
                        TextField("Creator Name", text: Binding(
                            get: { project.metadata.creator },
                            set: { project.metadata.creator = $0 }
                        ))
                        .textFieldStyle(.roundedBorder)
                    }
                    
                    LabeledField("Keywords") {
                        TextField("Comma-separated keywords", text: $project.metadata.keywords)
                        .textFieldStyle(.roundedBorder)
                    }
                }
            }
            
            // Project Statistics
            GroupBox("Statistics") {
                VStack(alignment: .leading, spacing: 8) {
                    ProjectStatRow(label: "Created", value: formatDate(project.createdAt))
                    ProjectStatRow(label: "Last Modified", value: formatDate(project.modifiedAt))
                    ProjectStatRow(label: "Duration", value: formatDuration(project.timeline.duration))
                    ProjectStatRow(label: "Video Tracks", value: "\(project.timeline.videoTracks.count)")
                    ProjectStatRow(label: "Audio Tracks", value: "\(project.timeline.audioTracks.count)")
                    ProjectStatRow(label: "Total Clips", value: "\(totalClips)")
                }
                .font(.caption)
            }
            
            // Project Location
            GroupBox("Location") {
                VStack(spacing: 8) {
                    LabeledField("Project File") {
                        HStack {
                            Text(project.name + ".autoresolve")
                                .lineLimit(1)
                                .foregroundColor(.primary)
                            
                            Spacer()
                            
                            Button("Reveal") {
                                // Project file location not available in current model
                                print("Project: \(project.name)")
                            }
                            .buttonStyle(.bordered)
                        }
                        .padding(8)
                        .background(Color(NSColor.textBackgroundColor))
                        .cornerRadius(4)
                    }
                    
                    LabeledField("Cache Location") {
                        HStack {
                            Text(project.cacheDirectory?.path ?? "Default")
                                .lineLimit(1)
                            
                            Spacer()
                            
                            Button("Change...") {
                                changeCacheLocation()
                            }
                            .buttonStyle(.bordered)
                        }
                        .padding(8)
                        .background(Color(NSColor.textBackgroundColor))
                        .cornerRadius(4)
                    }
                }
            }
        }
    }
    
    private var totalClips: Int {
        project.timeline.videoTracks.reduce(0) { $0 + $1.clips.count } +
        project.timeline.audioTracks.reduce(0) { $0 + $1.clips.count }
    }
    
    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let hours = Int(duration) / 3600
        let minutes = Int(duration) % 3600 / 60
        let seconds = Int(duration) % 60
        return String(format: "%02d:%02d:%02d", hours, minutes, seconds)
    }
    
    private func changeCacheLocation() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        
        if panel.runModal() == .OK {
            project.cacheDirectory = panel.url
        }
    }
}

// MARK: - Video Settings View

struct VideoSettingsView: View {
    @ObservedObject var project: VideoProject
    let showAdvanced: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Timeline Settings
            GroupBox("Timeline Settings") {
                VStack(spacing: 12) {
                    LabeledField("Format") {
                        Picker("", selection: $project.settings.videoFormat) {
                            ForEach(VideoFormat.allCases, id: \.self) { format in
                                Text(format.displayName).tag(format)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Resolution") {
                        HStack {
                            TextField("Width", value: $project.settings.width, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                            
                            Text("×")
                            
                            TextField("Height", value: $project.settings.height, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                            
                            Spacer()
                            
                            Menu("Presets") {
                                Button("4K (3840×2160)") { setResolution(3840, 2160) }
                                Button("1080p (1920×1080)") { setResolution(1920, 1080) }
                                Button("720p (1280×720)") { setResolution(1280, 720) }
                                Button("480p (854×480)") { setResolution(854, 480) }
                            }
                            .menuStyle(.borderlessButton)
                        }
                    }
                    
                    LabeledField("Frame Rate") {
                        Picker("", selection: $project.settings.frameRate) {
                            ForEach([23.98, 24, 25, 29.97, 30, 50, 59.94, 60], id: \.self) { rate in
                                Text("\(String(format: "%.2f", rate)) fps").tag(rate)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Pixel Aspect") {
                        Picker("", selection: $project.settings.pixelAspectRatio) {
                            Text("Square (1.0)").tag(1.0)
                            Text("NTSC DV (0.9)").tag(0.9)
                            Text("PAL DV (1.07)").tag(1.07)
                            Text("Anamorphic (1.33)").tag(1.33)
                        }
                        .pickerStyle(.menu)
                    }
                }
            }
            
            // Color Settings
            GroupBox("Color Management") {
                VStack(spacing: 12) {
                    LabeledField("Color Space") {
                        Picker("", selection: $project.settings.colorSpace) {
                            ForEach(ColorSpace.allCases, id: \.self) { space in
                                Text(space.displayName).tag(space)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Working Space") {
                        Picker("", selection: $project.settings.workingColorSpace) {
                            Text("Rec. 709").tag("Rec. 709")
                            Text("Rec. 2020").tag("Rec. 2020")
                            Text("sRGB").tag("sRGB")
                            Text("Adobe RGB").tag("Adobe RGB")
                        }
                        .pickerStyle(.menu)
                    }
                    
                    if showAdvanced {
                        LabeledField("Gamma") {
                            Picker("", selection: $project.settings.gamma) {
                                Text("2.2").tag(2.2)
                                Text("2.4").tag(2.4)
                                Text("sRGB").tag(2.2) // Simplified
                            }
                            .pickerStyle(.menu)
                        }
                        
                        Toggle("Use Color Management", isOn: $project.settings.useColorManagement)
                    }
                }
            }
            
            // Render Settings
            if showAdvanced {
                GroupBox("Render Settings") {
                    VStack(spacing: 12) {
                        LabeledField("Quality") {
                            Picker("", selection: $project.settings.renderQuality) {
                                Text("Draft").tag(RenderQuality.draft)
                                Text("Good").tag(RenderQuality.good)
                                Text("Better").tag(RenderQuality.better)
                                Text("Best").tag(RenderQuality.best)
                            }
                            .pickerStyle(.menu)
                        }
                        
                        LabeledField("Motion Blur") {
                            Slider(
                                value: $project.settings.motionBlurAmount,
                                in: 0...1,
                                step: 0.1
                            )
                            
                            Text("\(Int(project.settings.motionBlurAmount * 100))%")
                                .monospacedDigit()
                                .frame(width: 40)
                        }
                        
                        // Toggle("Field Rendering", isOn: $project.settings.fieldRendering) // fieldRendering is String not Bool
                        // Toggle("3:2 Pulldown Removal", isOn: $project.settings.pulldownRemoval) // Property doesn't exist
                    }
                }
            }
        }
    }
    
    private func setResolution(_ width: Int, _ height: Int) {
        project.settings.width = width
        project.settings.height = height
    }
}

// MARK: - Audio Settings View

struct AudioSettingsView: View {
    @ObservedObject var project: VideoProject
    let showAdvanced: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Audio Format
            GroupBox("Audio Format") {
                VStack(spacing: 12) {
                    LabeledField("Sample Rate") {
                        Picker("", selection: $project.settings.audioSampleRate) {
                            ForEach([44100, 48000, 96000, 192000], id: \.self) { rate in
                                Text("\(rate) Hz").tag(rate)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Bit Depth") {
                        Picker("", selection: $project.settings.audioBitDepth) {
                            Text("16-bit").tag(16)
                            Text("24-bit").tag(24)
                            Text("32-bit").tag(32)
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Channels") {
                        Picker("", selection: $project.settings.audioChannels) {
                            Text("Mono").tag(1)
                            Text("Stereo").tag(2)
                            Text("5.1 Surround").tag(6)
                            Text("7.1 Surround").tag(8)
                        }
                        .pickerStyle(.menu)
                    }
                }
            }
            
            // Monitoring
            GroupBox("Monitoring") {
                VStack(spacing: 12) {
                    LabeledField("Output Device") {
                        Picker("", selection: $project.settings.audioOutputDevice) {
                            ForEach(AudioOutputDevice.availableDevices, id: \.id) { device in
                                Text(device.name).tag(device.id)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Master Volume") {
                        HStack {
                            Slider(value: $project.settings.masterVolume, in: 0...1)
                            Text("\(Int(project.settings.masterVolume * 100))%")
                                .monospacedDigit()
                                .frame(width: 40)
                        }
                    }
                    
                    if showAdvanced {
                        LabeledField("Buffer Size") {
                            Picker("", selection: $project.settings.audioBufferSize) {
                                Text("64 samples").tag(64)
                                Text("128 samples").tag(128)
                                Text("256 samples").tag(256)
                                Text("512 samples").tag(512)
                                Text("1024 samples").tag(1024)
                            }
                            .pickerStyle(.menu)
                        }
                        
                        Toggle("Real-time Processing", isOn: $project.settings.realtimeAudio)
                    }
                }
            }
            
            // Levels and Meters
            if showAdvanced {
                GroupBox("Levels & Metering") {
                    VStack(spacing: 12) {
                        LabeledField("Reference Level") {
                            Picker("", selection: $project.settings.referenceLevelDB) {
                                Text("-18 dBFS").tag(-18)
                                Text("-20 dBFS").tag(-20)
                                Text("-23 dBFS").tag(-23)
                            }
                            .pickerStyle(.menu)
                        }
                        
                        // Peak Hold Time not available in ProjectSettings
                        /*LabeledField("Peak Hold") {
                            HStack {
                                Slider(
                                    value: $project.settings.peakHoldTime,
                                    in: 0...5,
                                    step: 0.5
                                )
                                Text("\(String(format: "%.1f", project.settings.peakHoldTime))s")
                                    .monospacedDigit()
                                    .frame(width: 40)
                            }
                        }*/
                        
                        // These properties don't exist in ProjectSettings
                        // Toggle("Show Phase Meter", isOn: $project.settings.showPhaseMeter)
                        // Toggle("Loudness Metering (LUFS)", isOn: $project.settings.loudnessMetering)
                    }
                }
            }
        }
    }
}

// MARK: - Performance Settings View

struct PerformanceSettingsView: View {
    @ObservedObject var project: VideoProject
    let showAdvanced: Bool
    
    @State private var availableMemory = ProcessInfo.processInfo.physicalMemory
    @State private var cpuCores = ProcessInfo.processInfo.processorCount
    
    var body: some View {
        let memoryString = ByteCountFormatter().string(fromByteCount: Int64(availableMemory))
        let gpuInfo = getGPUInfo()
        
        return VStack(alignment: .leading, spacing: 16) {
            // System Info
            GroupBox("System Information") {
                VStack(alignment: .leading, spacing: 8) {
                    ProjectStatRow(label: "CPU Cores", value: "\(cpuCores)")
                    ProjectStatRow(label: "Memory", value: memoryString)
                    ProjectStatRow(label: "GPU", value: gpuInfo)
                }
                .font(.caption)
            }
            
            // Playback Performance
            GroupBox("Playback Performance") {
                VStack(spacing: 12) {
                    LabeledField("Preview Quality") {
                        Picker("", selection: $project.settings.previewQuality) {
                            Text("Quarter (1/4)").tag(PreviewQuality.quarter)
                            Text("Half (1/2)").tag(PreviewQuality.half)
                            Text("Full").tag(PreviewQuality.full)
                            Text("Auto").tag(PreviewQuality.auto)
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Render Threads") {
                        Picker("", selection: $project.settings.renderThreads) {
                            Text("Auto").tag(0)
                            ForEach(1...cpuCores, id: \.self) { count in
                                Text("\(count) thread\(count == 1 ? "" : "s")").tag(count)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    Toggle("GPU Acceleration", isOn: $project.settings.useGPUAcceleration)
                    Toggle("Background Rendering", isOn: $project.settings.backgroundRendering)
                }
            }
            
            // Memory Management
            GroupBox("Memory Management") {
                VStack(spacing: 12) {
                    LabeledField("RAM Usage") {
                        HStack {
                            // maxRAMUsageGB doesn't exist in ProjectSettings
                            /*Slider(
                                value: Binding(
                                    get: { Double(project.settings.maxRAMUsageGB) },
                                    set: { project.settings.maxRAMUsageGB = Int($0) }
                                ),
                                in: 1...Double(availableMemory / 1_000_000_000),
                                step: 1
                            )
                            Text("\(project.settings.maxRAMUsageGB) GB")
                                .monospacedDigit()
                                .frame(width: 50)*/
                        }
                    }
                    
                    LabeledField("Cache Size") {
                        HStack {
                            // cacheSize doesn't exist in ProjectSettings
                            /*Slider(
                                value: Binding(
                                    get: { Double(project.settings.cacheSize) },
                                    set: { project.settings.cacheSize = Int($0) }
                                ),
                                in: 1...100,
                                step: 1
                            )
                            Text("\(project.settings.cacheSize) GB")
                                .monospacedDigit()
                                .frame(width: 50)*/
                        }
                    }
                    
                    if showAdvanced {
                        // Toggle("Purge Cache on Exit", isOn: $project.settings.purgeCacheOnExit) // Property doesn't exist
                        // Toggle("Pre-render Audio", isOn: $project.settings.preRenderAudio) // Property doesn't exist
                        
                        LabeledField("Disk Cache Location") {
                            HStack {
                                Text(project.settings.diskCacheLocation?.path ?? "Default")
                                    .lineLimit(1)
                                
                                Spacer()
                                
                                Button("Change...") {
                                    selectDiskCacheLocation()
                                }
                                .buttonStyle(.bordered)
                            }
                            .padding(8)
                            .background(Color(NSColor.textBackgroundColor))
                            .cornerRadius(4)
                        }
                    }
                }
            }
            
            // Performance Monitoring
            if showAdvanced {
                GroupBox("Performance Monitoring") {
                    VStack(spacing: 12) {
                        // Toggle("Show Performance Stats", isOn: $project.settings.showPerformanceStats) // Property doesn't exist
                        // Toggle("Memory Usage Warnings", isOn: $project.settings.memoryWarnings) // Property doesn't exist
                        // Toggle("GPU Memory Monitoring", isOn: $project.settings.gpuMemoryMonitoring) // Property doesn't exist
                        
                        // LabeledField("Warning Threshold") {
                        //     HStack {
                        //         Slider(
                        //             value: $project.settings.memoryWarningThreshold,
                        //             in: 50...95,
                        //             step: 5
                        //         )
                        //         Text("\(Int(project.settings.memoryWarningThreshold))%")
                        //             .monospacedDigit()
                        //             .frame(width: 40)
                        //     }
                        // } // Property doesn't exist
                    }
                }
            }
        }
    }
    
    private func getGPUInfo() -> String {
        // Simplified GPU detection
        "Metal Compatible"
    }
    
    private func selectDiskCacheLocation() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        
        if panel.runModal() == .OK {
            project.settings.diskCacheLocation = panel.url
        }
    }
}

// MARK: - Collaboration Settings View

struct CollaborationSettingsView: View {
    @ObservedObject var project: VideoProject
    
    @State private var shareEnabled = false
    @State private var shareURL = ""
    @State private var permissions = CollaborationPermissions()
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Sharing
            GroupBox("Project Sharing") {
                VStack(spacing: 12) {
                    Toggle("Enable Collaboration", isOn: $shareEnabled)
                    
                    if shareEnabled {
                        LabeledField("Share URL") {
                            HStack {
                                TextField("https://...", text: $shareURL)
                                    .textFieldStyle(.roundedBorder)
                                
                                Button("Copy") {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(shareURL, forType: .string)
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        
                        LabeledField("Access") {
                            Picker("", selection: $permissions.defaultAccess) {
                                Text("View Only").tag("view")
                                Text("Comment").tag("comment")
                                Text("Edit").tag("edit")
                            }
                            .pickerStyle(.menu)
                        }
                    }
                }
            }
            
            // Version Control
            GroupBox("Version Control") {
                VStack(spacing: 12) {
                    Toggle("Auto-save Versions", isOn: $permissions.autoSaveVersions)
                    
                    LabeledField("Save Interval") {
                        Picker("", selection: $permissions.saveInterval) {
                            Text("1 minute").tag(60)
                            Text("5 minutes").tag(300)
                            Text("10 minutes").tag(600)
                            Text("30 minutes").tag(1800)
                        }
                        .pickerStyle(.menu)
                        .disabled(!permissions.autoSaveVersions)
                    }
                    
                    LabeledField("Max Versions") {
                        Picker("", selection: $permissions.maxVersions) {
                            Text("5").tag(5)
                            Text("10").tag(10)
                            Text("25").tag(25)
                            Text("50").tag(50)
                            Text("Unlimited").tag(0)
                        }
                        .pickerStyle(.menu)
                    }
                }
            }
            
            // Comments and Notes
            GroupBox("Comments & Annotations") {
                VStack(spacing: 12) {
                    Toggle("Allow Comments", isOn: $permissions.allowComments)
                    Toggle("Allow Annotations", isOn: $permissions.allowAnnotations)
                    Toggle("Require Approval", isOn: $permissions.requireApproval)
                    
                    LabeledField("Notification") {
                        Picker("", selection: $permissions.notificationLevel) {
                            Text("None").tag("none")
                            Text("Mentions Only").tag("mentions")
                            Text("All Activity").tag("all")
                        }
                        .pickerStyle(.menu)
                    }
                }
            }
            
            // Export Permissions
            GroupBox("Export & Download") {
                VStack(spacing: 12) {
                    Toggle("Allow Export", isOn: $permissions.allowExport)
                    Toggle("Allow Download", isOn: $permissions.allowDownload)
                    Toggle("Watermark Exports", isOn: $permissions.watermarkExports)
                    
                    if permissions.watermarkExports {
                        LabeledField("Watermark Text") {
                            TextField("DRAFT - DO NOT DISTRIBUTE", text: $permissions.watermarkText)
                                .textFieldStyle(.roundedBorder)
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Helper Views

struct ProjectStatRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
        }
    }
}

// MARK: - Supporting Types

public enum VideoFormat: String, CaseIterable {
    case cinema4k = "cinema4k"
    case uhd4k = "uhd4k"
    case hd1080p = "hd1080p"
    case hd720p = "hd720p"
    case sd480p = "sd480p"
    case custom = "custom"
    
    var displayName: String {
        switch self {
        case .cinema4k: return "Cinema 4K (4096×2160)"
        case .uhd4k: return "UHD 4K (3840×2160)"
        case .hd1080p: return "HD 1080p (1920×1080)"
        case .hd720p: return "HD 720p (1280×720)"
        case .sd480p: return "SD 480p (854×480)"
        case .custom: return "Custom"
        }
    }
}

public enum ColorSpace: String, CaseIterable {
    case rec709 = "rec709"
    case rec2020 = "rec2020"
    case srgb = "srgb"
    case adobergb = "adobergb"
    case p3 = "p3"
    
    var displayName: String {
        switch self {
        case .rec709: return "Rec. 709"
        case .rec2020: return "Rec. 2020"
        case .srgb: return "sRGB"
        case .adobergb: return "Adobe RGB"
        case .p3: return "Display P3"
        }
    }
}

public enum RenderQuality: String, CaseIterable {
    case draft = "draft"
    case good = "good"
    case better = "better"
    case best = "best"
}

public enum PreviewQuality: String, CaseIterable {
    case quarter = "quarter"
    case half = "half"
    case full = "full"
    case auto = "auto"
}

public struct AudioOutputDevice {
    let id: String
    let name: String
    
    static let availableDevices: [AudioOutputDevice] = [
        AudioOutputDevice(id: "default", name: "Default Output"),
        AudioOutputDevice(id: "builtin", name: "Built-in Output"),
        // Add more devices as detected
    ]
}

public struct CollaborationPermissions {
    var defaultAccess = "view"
    var autoSaveVersions = true
    var saveInterval = 300 // 5 minutes
    var maxVersions = 10
    var allowComments = true
    var allowAnnotations = true
    var requireApproval = false
    var notificationLevel = "mentions"
    var allowExport = false
    var allowDownload = false
    var watermarkExports = true
    var watermarkText = "DRAFT - DO NOT DISTRIBUTE"
}