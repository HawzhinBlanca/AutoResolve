import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

// MARK: - Clip Properties Inspector

public struct ClipInspector: View {
    @ObservedObject var timeline: TimelineModel
    @State private var selectedTab: InspectorTab = .properties
    @State private var showAdvancedOptions = false
    
    enum InspectorTab: String, CaseIterable {
        case properties = "Properties"
        case transform = "Transform"
        case crop = "Crop"
        case audio = "Audio"
        case metadata = "Metadata"
        
        var icon: String {
            switch self {
            case .properties: return "info.circle"
            case .transform: return "move.3d"
            case .crop: return "crop"
            case .audio: return "waveform"
            case .metadata: return "doc.text"
            }
        }
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            InspectorHeader(title: selectedClipName)
            
            // Tab selector
            Picker("Tab", selection: $selectedTab) {
                ForEach(InspectorTab.allCases, id: \.self) { tab in
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
                    case .properties:
                        PropertiesPanel(clip: selectedClip, timeline: timeline)
                    case .transform:
                        TransformPanel(clip: selectedClip)
                    case .crop:
                        CropPanel(clip: selectedClip)
                    case .audio:
                        AudioPanel(clip: selectedClip)
                    case .metadata:
                        MetadataPanel(clip: selectedClip)
                    }
                    
                    if showAdvancedOptions {
                        AdvancedOptionsPanel(clip: selectedClip)
                    }
                }
                .padding()
            }
            
            Divider()
            
            // Footer
            HStack {
                Button(action: { showAdvancedOptions.toggle() }) {
                    Label(showAdvancedOptions ? "Hide Advanced" : "Show Advanced",
                          systemImage: "gearshape")
                        .font(.caption)
                }
                
                Spacer()
                
                Button("Reset All") {
                    resetClipProperties()
                }
                .buttonStyle(.plain)
                .foregroundColor(.red)
                .disabled(selectedClip == nil)
            }
            .padding(12)
        }
        .frame(width: 320)
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private var selectedClip: SimpleTimelineClip? {
        guard let clipId = timeline.selectedClips.first else { return nil }
        return timeline.tracks.flatMap { $0.clips }.first { $0.id == clipId }
    }
    
    private var selectedClipName: String {
        selectedClip?.name ?? "No Selection"
    }
    
    private func resetClipProperties() {
        // Reset UI fields for the selected clip; if none selected, no-op
        guard let _ = selectedClip else { return }
        // These are UI-only resets unless model binding is added later
        // Intentionally minimal to avoid misleading persistence
    }
}

// MARK: - Inspector Header

struct InspectorHeader: View {
    let title: String
    
    public var body: some View {
        HStack {
            Text(title)
                .font(.headline)
                .lineLimit(1)
            
            Spacer()
            
            Button(action: {}) {
                Image(systemName: "ellipsis.circle")
            }
            .buttonStyle(.plain)
        }
        .padding()
        .background(Color(NSColor.windowBackgroundColor))
    }
}

// MARK: - Properties Panel

struct PropertiesPanel: View {
    var clip: SimpleTimelineClip?
    @ObservedObject var timeline: TimelineModel
    
    @State private var clipName: String = ""
    @State private var startTime: TimeInterval = 0.0
    @State private var duration: TimeInterval = 10.0
    @State private var inPoint: TimeInterval = 0.0
    @State private var outPoint: TimeInterval = 10.0
    @State private var speed: Double = 1.0
    @State private var reverse: Bool = false
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Name
            LabeledField("Name") {
                TextField("Clip Name", text: $clipName)
                    .textFieldStyle(.roundedBorder)
            }
            
            Divider()
            
            // Timing
            GroupBox("Timing") {
                VStack(spacing: 8) {
                    LabeledField("Start") {
                        TimecodeField(time: Binding(
                            get: { startTime },
                            set: { startTime = max(0, $0) }
                        ), timeline: timeline)
                    }
                    
                    LabeledField("Duration") {
                        TimecodeField(time: Binding(
                            get: { duration },
                            set: { duration = max(0.01, $0) }
                        ), timeline: timeline)
                    }
                    
                    LabeledField("In Point") {
                        TimecodeField(time: Binding(
                            get: { inPoint },
                            set: { newVal in
                                inPoint = max(0, min(newVal, outPoint))
                            }
                        ), timeline: timeline)
                    }
                    
                    LabeledField("Out Point") {
                        TimecodeField(time: Binding(
                            get: { outPoint },
                            set: { newVal in
                                outPoint = max(inPoint, min(newVal, startTime + duration))
                            }
                        ), timeline: timeline)
                    }
                }
            }
            
            Divider()
            
            // Speed
            GroupBox("Speed") {
                VStack(spacing: 8) {
                    HStack {
                        Text("Playback Speed")
                        Spacer()
                        Text("\(String(format: "%.1f", max(0.1, min(4.0, speed))))x")
                            .monospacedDigit()
                    }
                    
                    Slider(value: $speed, in: 0.1...4.0, step: 0.1)
                    
                    HStack {
                        Button("0.5x") { speed = 0.5 }
                        Button("1x") { speed = 1.0 }
                        Button("2x") { speed = 2.0 }
                        Button("4x") { speed = 4.0 }
                    }
                    .buttonStyle(.bordered)
                    
                    Toggle("Reverse", isOn: $reverse)
                }
            }
            
            Divider()
            
            // Blend Mode
            GroupBox("Compositing") {
                VStack(spacing: 8) {
                    LabeledField("Blend Mode") {
                        Picker("", selection: .constant("Normal")) {
                            Text("Normal").tag("Normal")
                            Text("Add").tag("Add")
                            Text("Multiply").tag("Multiply")
                            Text("Screen").tag("Screen")
                            Text("Overlay").tag("Overlay")
                        }
                        .pickerStyle(.menu)
                    }
                    
                    LabeledField("Opacity") {
                        HStack {
                            Slider(value: .constant(1.0), in: 0...1)
                            Text("100%")
                                .monospacedDigit()
                                .frame(width: 45)
                        }
                    }
                }
            }
        }
        .onAppear { syncFromClip() }
        .onChange(of: clip?.id) { _, _ in syncFromClip() }
    }
}

private extension PropertiesPanel {
    func syncFromClip() {
        guard let clip = clip else { return }
        clipName = clip.name
        startTime = clip.startTime
        duration = clip.duration ?? 0
        inPoint = clip.inPoint
        outPoint = clip.outPoint
    }
}

// MARK: - Transform Panel

struct TransformPanel: View {
    var clip: SimpleTimelineClip?
    
    @State private var positionX: Double = 0
    @State private var positionY: Double = 0
    @State private var scaleX: Double = 100
    @State private var scaleY: Double = 100
    @State private var rotation: Double = 0
    @State private var anchorX: Double = 0.5
    @State private var anchorY: Double = 0.5
    @State private var uniformScale = true
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Position
            GroupBox("Position") {
                VStack(spacing: 8) {
                    NumberField(label: "X", value: $positionX, suffix: "px")
                    NumberField(label: "Y", value: $positionY, suffix: "px")
                    
                    HStack {
                        Button("Center") {
                            positionX = 0
                            positionY = 0
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            
            // Scale
            GroupBox("Scale") {
                VStack(spacing: 8) {
                    Toggle("Uniform Scale", isOn: $uniformScale)
                    
                    NumberField(label: "X", value: $scaleX, suffix: "%")
                        .onChange(of: scaleX) { _, newValue in
                            if uniformScale {
                                scaleY = newValue
                            }
                        }
                    
                    NumberField(label: "Y", value: $scaleY, suffix: "%")
                        .disabled(uniformScale)
                    
                    HStack {
                        Button("50%") { setScale(50) }
                        Button("100%") { setScale(100) }
                        Button("150%") { setScale(150) }
                        Button("200%") { setScale(200) }
                    }
                    .buttonStyle(.bordered)
                }
            }
            
            // Rotation
            GroupBox("Rotation") {
                VStack(spacing: 8) {
                    NumberField(label: "Angle", value: $rotation, suffix: "°")
                    
                    Slider(value: $rotation, in: -180...180)
                    
                    HStack {
                        Button("-90°") { rotation = -90 }
                        Button("0°") { rotation = 0 }
                        Button("90°") { rotation = 90 }
                        Button("180°") { rotation = 180 }
                    }
                    .buttonStyle(.bordered)
                }
            }
            
            // Anchor Point
            GroupBox("Anchor Point") {
                VStack(spacing: 8) {
                    NumberField(label: "X", value: $anchorX, suffix: "")
                    NumberField(label: "Y", value: $anchorY, suffix: "")
                    
                    AnchorPointSelector(x: $anchorX, y: $anchorY)
                }
            }
        }
    }
    
    private func setScale(_ value: Double) {
        scaleX = value
        if uniformScale {
            scaleY = value
        }
    }
}

// MARK: - Crop Panel

struct CropPanel: View {
    var clip: SimpleTimelineClip?
    
    @State private var cropLeft: Double = 0
    @State private var cropRight: Double = 0
    @State private var cropTop: Double = 0
    @State private var cropBottom: Double = 0
    @State private var feather: Double = 0
    @State private var maintainAspect = true
    @State private var sourceSize: CGSize = CGSize(width: 1920, height: 1080)
    @State private var sourceSizeTask: Task<Void, Never>? = nil
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            GroupBox("Crop Values") {
                VStack(spacing: 8) {
                    NumberField(label: "Left", value: Binding(
                        get: { cropLeft },
                        set: { cropLeft = max(0, min($0, Double(sourceSize.width))) }
                    ), suffix: "px")
                    NumberField(label: "Right", value: Binding(
                        get: { cropRight },
                        set: { cropRight = max(0, min($0, Double(sourceSize.width))) }
                    ), suffix: "px")
                    NumberField(label: "Top", value: Binding(
                        get: { cropTop },
                        set: { cropTop = max(0, min($0, Double(sourceSize.height))) }
                    ), suffix: "px")
                    NumberField(label: "Bottom", value: Binding(
                        get: { cropBottom },
                        set: { cropBottom = max(0, min($0, Double(sourceSize.height))) }
                    ), suffix: "px")
                }
            }
            
            GroupBox("Options") {
                VStack(spacing: 8) {
                    NumberField(label: "Feather", value: Binding(
                        get: { feather },
                        set: { feather = max(0, $0) }
                    ), suffix: "px")
                    Toggle("Maintain Aspect Ratio", isOn: $maintainAspect)
                }
            }
            
            // Visual crop preview
            CropPreview(
                left: cropLeft,
                right: cropRight,
                top: cropTop,
                bottom: cropBottom,
                sourceSize: sourceSize
            )
            .frame(height: 150)
            
            HStack {
                Button("Reset Crop") {
                    cropLeft = 0
                    cropRight = 0
                    cropTop = 0
                    cropBottom = 0
                    feather = 0
                }
                .buttonStyle(.bordered)
                
                Spacer()
                
                Button("Auto Crop") {
                    // Detect and remove black bars
                }
                .buttonStyle(.bordered)
            }
        }
        .onAppear { updateSourceSize() }
        .onChange(of: clip?.id) { _, _ in updateSourceSize() }
        .onDisappear { sourceSizeTask?.cancel() }
    }
    
    private func updateSourceSize() {
        guard let url = clip?.sourceURL else { return }
        sourceSizeTask?.cancel()
        let asset = AVURLAsset(url: url)
        sourceSizeTask = Task {
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                if let track = tracks.first {
                    let nat = try await track.load(.naturalSize)
                    let tx = try await track.load(.preferredTransform)
                    let transformed = nat.applying(tx)
                    let width = abs(transformed.width)
                    let height = abs(transformed.height)
                    if !Task.isCancelled, width > 0 && height > 0 {
                        await MainActor.run {
                            sourceSize = CGSize(width: width, height: height)
                        }
                    }
                }
            } catch {
                // Keep default sourceSize on error
            }
        }
    }
}

// MARK: - Audio Panel

struct AudioPanel: View {
    var clip: SimpleTimelineClip?
    
    @State private var volume: Double = 0
    @State private var pan: Double = 0
    @State private var mute = false
    @State private var solo = false
    @State private var fadeInDuration: Double = 0
    @State private var fadeOutDuration: Double = 0
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Volume
            GroupBox("Volume") {
                VStack(spacing: 8) {
                    HStack {
                        Text("Level")
                        Spacer()
                        Text("\(String(format: "%.1f", volume)) dB")
                            .monospacedDigit()
                    }
                    
                    Slider(value: $volume, in: -60...12)
                    
                    HStack {
                        Toggle("Mute", isOn: $mute)
                        Toggle("Solo", isOn: $solo)
                    }
                }
            }
            
            // Pan
            GroupBox("Pan") {
                VStack(spacing: 8) {
                    HStack {
                        Text("L")
                        Slider(value: $pan, in: -100...100)
                        Text("R")
                    }
                    
                    HStack {
                        Spacer()
                        Text(panDescription)
                            .font(.caption)
                            .monospacedDigit()
                        Spacer()
                    }
                }
            }
            
            // Fades
            GroupBox("Fades") {
                VStack(spacing: 8) {
                    NumberField(label: "Fade In", value: Binding(
                        get: { fadeInDuration },
                        set: { fadeInDuration = max(0, $0) }
                    ), suffix: "s")
                    NumberField(label: "Fade Out", value: Binding(
                        get: { fadeOutDuration },
                        set: { fadeOutDuration = max(0, $0) }
                    ), suffix: "s")
                    
                    // Fade curve selector
                    LabeledField("Curve") {
                        Picker("", selection: .constant("Linear")) {
                            Text("Linear").tag("Linear")
                            Text("S-Curve").tag("S-Curve")
                            Text("Exponential").tag("Exponential")
                        }
                        .pickerStyle(.menu)
                    }
                }
            }
            
            // Audio meters
            AudioMetersView()
                .frame(height: 100)
        }
    }
    
    private var panDescription: String {
        if pan == 0 {
            return "Center"
        } else if pan < 0 {
            return "L\(Int(abs(pan)))"
        } else {
            return "R\(Int(pan))"
        }
    }
}

// MARK: - Metadata Panel

struct MetadataPanel: View {
    var clip: SimpleTimelineClip?
