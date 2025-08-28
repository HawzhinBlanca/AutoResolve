import AppKit
import SwiftUI
import AVFoundation

// MARK: - Effects Inspector

public struct EffectsInspector: View {
    @ObservedObject var timeline: TimelineModel
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    @State private var selectedEffect: VideoProcessorEffect?
    @State private var showEffectBrowser = false
    @State private var searchText = ""
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            InspectorHeader(title: "Effects")
            
            // Applied Effects List
            VStack(spacing: 0) {
                HStack {
                    Text("Applied Effects")
                        .font(.subheadline.bold())
                    
                    Spacer()
                    
                    Button(action: { showEffectBrowser = true }) {
                        Image(systemName: "plus")
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                
                Divider()
                
                ScrollView {
                    LazyVStack(spacing: 2) {
                        ForEach(effectsProcessor.currentEffects) { effect in
                            EffectRow(
                                effect: effect,
                                isSelected: selectedEffect?.id == effect.id,
                                onSelect: { selectedEffect = effect },
                                onRemove: { effectsProcessor.removeEffect(effect) },
                                onToggle: { toggleEffect(effect) }
                            )
                        }
                        
                        if effectsProcessor.currentEffects.isEmpty {
                            Text("No effects applied")
                                .foregroundColor(.secondary)
                                .padding()
                        }
                    }
                }
                .frame(maxHeight: 200)
            }
            
            Divider()
            
            // Effect Controls
            if let selectedEffect = selectedEffect {
                EffectControls(
                    effect: selectedEffect,
                    effectsProcessor: effectsProcessor
                )
            } else {
                VStack {
                    Spacer()
                    Text("Select an effect to edit")
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
        }
        .frame(width: 320)
        .background(Color(NSColor.controlBackgroundColor))
        .sheet(isPresented: $showEffectBrowser) {
            EffectBrowser(
                effectsProcessor: effectsProcessor,
                onDismiss: { showEffectBrowser = false }
            )
        }
    }
    
    private func toggleEffect(_ effect: VideoProcessorEffect) {
        var updatedEffect = effect
        updatedEffect.enabled.toggle()
        effectsProcessor.updateEffect(updatedEffect)
    }
}

// MARK: - Effect Row

struct EffectRow: View {
    let effect: VideoProcessorEffect
    let isSelected: Bool
    let onSelect: () -> Void
    let onRemove: () -> Void
    let onToggle: () -> Void
    
    public var body: some View {
        HStack(spacing: 8) {
            // Enable/Disable toggle
            Button(action: onToggle) {
                Image(systemName: effect.enabled ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(effect.enabled ? .green : .secondary)
            }
            .buttonStyle(.plain)
            
            // Effect icon
            Image(systemName: iconForEffect(effect))
                .foregroundColor(effect.enabled ? .primary : .secondary)
                .frame(width: 20)
            
            // Effect name and intensity
            VStack(alignment: .leading, spacing: 2) {
                Text(effect.name)
                    .font(.caption)
                    .foregroundColor(effect.enabled ? .primary : .secondary)
                
                Text("\(Int(effect.intensity * 100))%")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Remove button
            Button(action: onRemove) {
                Image(systemName: "trash")
                    .foregroundColor(.red)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
        .onTapGesture {
            onSelect()
        }
    }
    
    private func iconForEffect(_ effect: VideoProcessorEffect) -> String {
        switch effect.type {
        case .colorCorrection: return "slider.horizontal.3"
        case .blur: return "drop.circle"
        case .brightness: return "sun.max"
        case .contrast: return "circle.lefthalf.filled"
        case .saturation: return "paintpalette"
        case .vignette: return "camera.filters"
        case .sharpen: return "sparkles"
        case .lut: return "square.grid.3x3"
        }
    }
}

// MARK: - Effect Controls

struct EffectControls: View {
    let effect: VideoProcessorEffect
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    public var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Effect header
                HStack {
                    Text(effect.name)
                        .font(.headline)
                    
                    Spacer()
                    
                    Button("Reset") {
                        resetEffect()
                    }
                    .buttonStyle(.bordered)
                }
                .padding(.horizontal)
                
                Divider()
                
                // Effect-specific controls
                switch effect.type {
                case .colorCorrection(let settings):
                    ColorCorrectionControls(
                        settings: settings,
                        effect: effect,
                        effectsProcessor: effectsProcessor
                    )
                    
                case .blur(let radius):
                    BlurControls(
                        radius: radius,
                        effect: effect,
                        effectsProcessor: effectsProcessor
                    )
                    
                case .brightness(let amount):
                    BrightnessControls(
                        amount: amount,
                        effect: effect,
                        effectsProcessor: effectsProcessor
                    )
                    
                case .contrast(let amount):
                    ContrastControls(
                        amount: amount,
                        effect: effect,
                        effectsProcessor: effectsProcessor
                    )
                    
                case .saturation(let amount):
                    SaturationControls(
                        amount: amount,
                        effect: effect,
                        effectsProcessor: effectsProcessor
                    )
                    
                case .vignette(let intensity):
                    VignetteControls(
                        intensity: intensity,
                        effect: effect,
                        effectsProcessor: effectsProcessor
                    )
                    
                case .sharpen(let intensity):
                    SharpenControls(
                        intensity: intensity,
                        effect: effect,
                        effectsProcessor: effectsProcessor
                    )
                    
                case .lut(let lutImage):
                    LUTControls(
                        lutImage: lutImage,
                        effect: effect,
                        effectsProcessor: effectsProcessor
                    )
                }
                
                Divider()
                
                // Keyframe controls
                KeyframeSection(effect: effect)
            }
            .padding()
        }
    }
    
    private func resetEffect() {
        // Reset effect parameters to defaults
    }
}

// MARK: - Color Correction Controls

struct ColorCorrectionControls: View {
    let settings: ColorCorrectionSettings
    let effect: VideoProcessorEffect
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var temperature: Double
    @State private var tint: Double
    @State private var exposure: Double
    @State private var highlights: Double
    @State private var shadows: Double
    @State private var whites: Double
    @State private var blacks: Double
    @State private var vibrance: Double
    
    init(settings: ColorCorrectionSettings, effect: VideoProcessorEffect, effectsProcessor: VideoEffectsProcessor) {
        self.settings = settings
        self.effect = effect
        self.effectsProcessor = effectsProcessor
        
        _temperature = State(initialValue: settings.temperature)
        _tint = State(initialValue: settings.tint)
        _exposure = State(initialValue: settings.exposure)
        _highlights = State(initialValue: settings.highlights)
        _shadows = State(initialValue: settings.shadows)
        _whites = State(initialValue: settings.whites)
        _blacks = State(initialValue: settings.blacks)
        _vibrance = State(initialValue: settings.vibrance)
    }
    
    public var body: some View {
        VStack(spacing: 12) {
            GroupBox("Basic") {
                VStack(spacing: 8) {
                    SliderControl("Temperature", value: $temperature, range: 2000...10000, format: "K")
                    SliderControl("Tint", value: $tint, range: -100...100)
                    SliderControl("Exposure", value: $exposure, range: -3...3, format: "EV")
                }
            }
            
            GroupBox("Tone") {
                VStack(spacing: 8) {
                    SliderControl("Highlights", value: $highlights, range: -100...100)
                    SliderControl("Shadows", value: $shadows, range: -100...100)
                    SliderControl("Whites", value: $whites, range: -100...100)
                    SliderControl("Blacks", value: $blacks, range: -100...100)
                }
            }
            
            GroupBox("Presence") {
                VStack(spacing: 8) {
                    SliderControl("Vibrance", value: $vibrance, range: -100...100)
                }
            }
        }
        .onChange(of: temperature) { _ in updateEffect() }
        .onChange(of: tint) { _ in updateEffect() }
        .onChange(of: exposure) { _ in updateEffect() }
        .onChange(of: highlights) { _ in updateEffect() }
        .onChange(of: shadows) { _ in updateEffect() }
        .onChange(of: whites) { _ in updateEffect() }
        .onChange(of: blacks) { _ in updateEffect() }
        .onChange(of: vibrance) { _ in updateEffect() }
    }
    
    private func updateEffect() {
        var newSettings = settings
        newSettings.temperature = temperature
        newSettings.tint = tint
        newSettings.exposure = exposure
        newSettings.highlights = highlights
        newSettings.shadows = shadows
        newSettings.whites = whites
        newSettings.blacks = blacks
        newSettings.vibrance = vibrance
        
        var updatedEffect = effect
        updatedEffect.type = .colorCorrection(newSettings)
        effectsProcessor.updateEffect(updatedEffect)
    }
}

// MARK: - Basic Effect Controls

struct BlurControls: View {
    let radius: Double
    let effect: VideoProcessorEffect
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var blurRadius: Double
    
    init(radius: Double, effect: VideoProcessorEffect, effectsProcessor: VideoEffectsProcessor) {
        self.radius = radius
        self.effect = effect
        self.effectsProcessor = effectsProcessor
        _blurRadius = State(initialValue: radius)
    }
    
    public var body: some View {
        GroupBox("Blur") {
            SliderControl("Radius", value: $blurRadius, range: 0...100, format: "px")
                .onChange(of: blurRadius) { newValue in
                    var updatedEffect = effect
                    updatedEffect.type = .blur(radius: newValue)
                    effectsProcessor.updateEffect(updatedEffect)
                }
        }
    }
}

struct BrightnessControls: View {
    let amount: Double
    let effect: VideoProcessorEffect
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var brightness: Double
    
    init(amount: Double, effect: VideoProcessorEffect, effectsProcessor: VideoEffectsProcessor) {
        self.amount = amount
        self.effect = effect
        self.effectsProcessor = effectsProcessor
        _brightness = State(initialValue: amount)
    }
    
    public var body: some View {
        GroupBox("Brightness") {
            SliderControl("Amount", value: $brightness, range: -1...1)
                .onChange(of: brightness) { newValue in
                    var updatedEffect = effect
                    updatedEffect.type = .brightness(amount: newValue)
                    effectsProcessor.updateEffect(updatedEffect)
                }
        }
    }
}

struct ContrastControls: View {
    let amount: Double
    let effect: VideoProcessorEffect
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var contrast: Double
    
    init(amount: Double, effect: VideoProcessorEffect, effectsProcessor: VideoEffectsProcessor) {
        self.amount = amount
        self.effect = effect
        self.effectsProcessor = effectsProcessor
        _contrast = State(initialValue: amount)
    }
    
    public var body: some View {
        GroupBox("Contrast") {
            SliderControl("Amount", value: $contrast, range: 0...2)
                .onChange(of: contrast) { newValue in
                    var updatedEffect = effect
                    updatedEffect.type = .contrast(amount: newValue)
                    effectsProcessor.updateEffect(updatedEffect)
                }
        }
    }
}

struct SaturationControls: View {
    let amount: Double
    let effect: VideoProcessorEffect
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var saturation: Double
    
    init(amount: Double, effect: VideoProcessorEffect, effectsProcessor: VideoEffectsProcessor) {
        self.amount = amount
        self.effect = effect
        self.effectsProcessor = effectsProcessor
        _saturation = State(initialValue: amount)
    }
    
    public var body: some View {
        GroupBox("Saturation") {
            SliderControl("Amount", value: $saturation, range: 0...2)
                .onChange(of: saturation) { newValue in
                    var updatedEffect = effect
                    updatedEffect.type = .saturation(amount: newValue)
                    effectsProcessor.updateEffect(updatedEffect)
                }
        }
    }
}

struct VignetteControls: View {
    let intensity: Double
    let effect: VideoProcessorEffect
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var vignetteIntensity: Double
    
    init(intensity: Double, effect: VideoProcessorEffect, effectsProcessor: VideoEffectsProcessor) {
        self.intensity = intensity
        self.effect = effect
        self.effectsProcessor = effectsProcessor
        _vignetteIntensity = State(initialValue: intensity)
    }
    
    public var body: some View {
        GroupBox("Vignette") {
            SliderControl("Intensity", value: $vignetteIntensity, range: 0...3)
                .onChange(of: vignetteIntensity) { newValue in
                    var updatedEffect = effect
                    updatedEffect.type = .vignette(intensity: newValue)
                    effectsProcessor.updateEffect(updatedEffect)
                }
        }
    }
}

struct SharpenControls: View {
    let intensity: Double
    let effect: VideoProcessorEffect
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var sharpenIntensity: Double
    
    init(intensity: Double, effect: VideoProcessorEffect, effectsProcessor: VideoEffectsProcessor) {
        self.intensity = intensity
        self.effect = effect
        self.effectsProcessor = effectsProcessor
        _sharpenIntensity = State(initialValue: intensity)
    }
    
    public var body: some View {
        GroupBox("Sharpen") {
            SliderControl("Intensity", value: $sharpenIntensity, range: 0...2)
                .onChange(of: sharpenIntensity) { newValue in
                    var updatedEffect = effect
                    updatedEffect.type = .sharpen(intensity: newValue)
                    effectsProcessor.updateEffect(updatedEffect)
                }
        }
    }
}

struct LUTControls: View {
    let lutImage: NSImage
    let effect: VideoProcessorEffect
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var intensity: Double = 1.0
    @State private var showLUTBrowser = false
    
    public var body: some View {
        GroupBox("LUT") {
            VStack(spacing: 8) {
                HStack {
                    Text("Current LUT")
                    Spacer()
                    Button("Browse...") {
                        showLUTBrowser = true
                    }
                    .buttonStyle(.bordered)
                }
                
                SliderControl("Intensity", value: $intensity, range: 0...1)
            }
        }
        .fileImporter(
            isPresented: $showLUTBrowser,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            // Handle LUT file selection
        }
    }
}

// MARK: - Keyframe Section

struct KeyframeSection: View {
    let effect: VideoProcessorEffect
    
    @State private var showKeyframes = false
    @State private var keyframes: [EffectKeyframe] = []
    
    public var body: some View {
        GroupBox("Animation") {
            VStack(spacing: 8) {
                HStack {
                    Button(action: { showKeyframes.toggle() }) {
                        Label("Keyframes", systemImage: showKeyframes ? "chevron.down" : "chevron.right")
                    }
                    .buttonStyle(.plain)
                    
                    Spacer()
                    
                    Button("Add Keyframe") {
                        addKeyframe()
                    }
                    .buttonStyle(.bordered)
                }
                
                if showKeyframes {
                    KeyframeTimeline(keyframes: $keyframes)
                        .frame(height: 60)
                }
            }
        }
    }
    
    private func addKeyframe() {
        let newKeyframe = EffectKeyframe(
            time: 0, // Current timeline position
            value: effect.intensity
        )
        keyframes.append(newKeyframe)
    }
}

// MARK: - Effect Browser

struct EffectBrowser: View {
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    let onDismiss: () -> Void
    
    @State private var searchText = ""
    @State private var selectedCategory: EffectCategory = .all
    
    enum EffectCategory: String, CaseIterable {
        case all = "All"
        case color = "Color"
        case blur = "Blur & Sharpen"
        case stylize = "Stylize"
        case distortion = "Distortion"
        case generators = "Generators"
        
        var effects: [EffectDefinition] {
            switch self {
            case .all:
                return EffectDefinition.allEffects
            case .color:
                return EffectDefinition.colorEffects
            case .blur:
                return EffectDefinition.blurEffects
            case .stylize:
                return EffectDefinition.stylizeEffects
            case .distortion:
                return EffectDefinition.distortionEffects
            case .generators:
                return EffectDefinition.generatorEffects
            }
        }
    }
    
    public var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Search bar
                SearchBar(text: $searchText)
                    .padding()
                
                // Category selector
                Picker("Category", selection: $selectedCategory) {
                    ForEach(EffectCategory.allCases, id: \.self) { category in
                        Text(category.rawValue).tag(category)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)
                
                Divider()
                
                // Effects grid
                ScrollView {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3)) {
                        ForEach(filteredEffects) { effectDef in
                            EffectCard(effectDefinition: effectDef) {
                                addEffect(effectDef)
                            }
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Effects")
            // .navigationBarTitleDisplayMode(.inline) // Not available on macOS
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        onDismiss()
                    }
                }
            }
        }
        .frame(width: 600, height: 500)
    }
    
    private var filteredEffects: [EffectDefinition] {
        let categoryEffects = selectedCategory.effects
        
        if searchText.isEmpty {
            return categoryEffects
        } else {
            return categoryEffects.filter { effect in
                effect.name.localizedCaseInsensitiveContains(searchText) ||
                effect.description.localizedCaseInsensitiveContains(searchText)
            }
        }
    }
    
    private func addEffect(_ effectDef: EffectDefinition) {
        let effect = effectDef.createEffect()
        effectsProcessor.addEffect(effect)
        onDismiss()
    }
}

// MARK: - Helper Views

struct SliderControl: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String
    
    init(_ label: String, value: Binding<Double>, range: ClosedRange<Double>, format: String = "") {
        self.label = label
        self._value = value
        self.range = range
        self.format = format
    }
    
    public var body: some View {
        HStack {
            Text(label)
                .frame(width: 80, alignment: .leading)
            
            Slider(value: $value, in: range)
            
            Text("\(String(format: "%.1f", value))\(format)")
                .monospacedDigit()
                .frame(width: 50, alignment: .trailing)
        }
    }
}

struct SearchBar: View {
    @Binding var text: String
    
    public var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.secondary)
            
            TextField("Search effects...", text: $text)
            
            if !text.isEmpty {
                Button(action: { text = "" }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(8)
        .background(Color(NSColor.textBackgroundColor))
        .cornerRadius(8)
    }
}

struct EffectCard: View {
    let effectDefinition: EffectDefinition
    let onAdd: () -> Void
    
    public var body: some View {
        VStack(spacing: 8) {
            // Effect preview thumbnail
            Rectangle()
                .fill(LinearGradient(colors: [.blue, .purple], startPoint: .topLeading, endPoint: .bottomTrailing))
                .frame(height: 80)
                .cornerRadius(8)
                .overlay(
                    Image(systemName: effectDefinition.icon)
                        .font(.title)
                        .foregroundColor(.white)
                )
            
            // Effect name
            Text(effectDefinition.name)
                .font(.caption.bold())
                .lineLimit(1)
            
            // Add button
            Button("Add") {
                onAdd()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(8)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}

// MARK: - Effect Definitions

struct EffectDefinition: Identifiable {
    public let id = UUID()
    let name: String
    let description: String
    let icon: String
    let createEffect: () -> VideoProcessorEffect
    
    static let allEffects: [EffectDefinition] = colorEffects + blurEffects + stylizeEffects + distortionEffects + generatorEffects
    
    static let colorEffects: [EffectDefinition] = [
        EffectDefinition(
            name: "Color Correction",
            description: "Adjust color temperature, tint, and exposure",
            icon: "slider.horizontal.3"
        ) {
            VideoProcessorEffect(
                name: "Color Correction",
                type: .colorCorrection(ColorCorrectionSettings())
            )
        },
        EffectDefinition(
            name: "Brightness",
            description: "Adjust image brightness",
            icon: "sun.max"
        ) {
            VideoProcessorEffect(
                name: "Brightness",
                type: .brightness(amount: 0.0)
            )
        },
        EffectDefinition(
            name: "Contrast",
            description: "Adjust image contrast",
            icon: "circle.lefthalf.filled"
        ) {
            VideoProcessorEffect(
                name: "Contrast",
                type: .contrast(amount: 1.0)
            )
        },
        EffectDefinition(
            name: "Saturation",
            description: "Adjust color saturation",
            icon: "paintpalette"
        ) {
            VideoProcessorEffect(
                name: "Saturation",
                type: .saturation(amount: 1.0)
            )
        }
    ]
    
    static let blurEffects: [EffectDefinition] = [
        EffectDefinition(
            name: "Gaussian Blur",
            description: "Apply blur effect",
            icon: "drop.circle"
        ) {
            VideoProcessorEffect(
                name: "Gaussian Blur",
                type: .blur(radius: 5.0)
            )
        },
        EffectDefinition(
            name: "Sharpen",
            description: "Enhance image sharpness",
            icon: "sparkles"
        ) {
            VideoProcessorEffect(
                name: "Sharpen",
                type: .sharpen(intensity: 0.5)
            )
        }
    ]
    
    static let stylizeEffects: [EffectDefinition] = [
        EffectDefinition(
            name: "Vignette",
            description: "Add dark edges to image",
            icon: "camera.filters"
        ) {
            VideoProcessorEffect(
                name: "Vignette",
                type: .vignette(intensity: 1.0)
            )
        }
    ]
    
    static let distortionEffects: [EffectDefinition] = []
    static let generatorEffects: [EffectDefinition] = []
}

// MARK: - Keyframe Models

struct EffectKeyframe: Identifiable {
    public let id = UUID()
    var time: TimeInterval
    var value: Double
}

struct KeyframeTimeline: View {
    @Binding var keyframes: [EffectKeyframe]
    
    public var body: some View {
        // Simplified keyframe timeline view
        Rectangle()
            .fill(Color.black.opacity(0.1))
            .overlay(
                HStack {
                    ForEach(keyframes) { keyframe in
                        Circle()
                            .fill(Color.accentColor)
                            .frame(width: 8, height: 8)
                    }
                    Spacer()
                }
            )
    }
}
