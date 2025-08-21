// AUTORESOLVE V3.0 - MACOS SEQUOIA NATIVE
// APPLE SILICON EXCLUSIVE - NEURAL ENGINE + METAL 3

import SwiftUI
import UniformTypeIdentifiers
import AVKit
import Vision
import NaturalLanguage
import CoreML
import Metal
import Combine

// MARK: - Navigation Items
enum NavigationItem: String, CaseIterable, Identifiable {
    case importMedia, timeline, director, transcription
    
    var id: String { rawValue }
    
    var label: String {
        switch self {
        case .importMedia: return "Import Media"
        case .timeline: return "Neural Timeline"
        case .director: return "Director's Brain"
        case .transcription: return "Transcription"
        }
    }
    
    var icon: String {
        switch self {
        case .importMedia: return "plus.rectangle.on.folder.fill"
        case .timeline: return "timeline.play"
        case .director: return "brain.fill"
        case .transcription: return "captions.bubble"
        }
    }
}

// MARK: - PRIMARY INTERFACE - NEURAL TIMELINE
struct NeuralTimelineView: View {
    @EnvironmentObject var store: UnifiedStore
    @State private var timelineScale: CGFloat = 100
    @State private var inspectorWidth: CGFloat = 340
    @State private var sidebarSelection: NavigationItem? = .timeline
    @State private var isProcessing = false
    @Namespace private var animation
    
    var body: some View {
        NavigationSplitView(
            columnVisibility: .constant(.all),
            preferredCompactColumn: .constant(.sidebar)
        ) {
            // INTELLIGENT SIDEBAR
            IntelligentSidebar(selection: $sidebarSelection)
                .navigationSplitViewColumnWidth(280)
                .background(.ultraThinMaterial)
        } content: {
            // MAIN CANVAS
            GeometryReader { geometry in
                ZStack {
                    // MULTI-LAYER TIMELINE
                    TimelineCanvas(scale: $timelineScale)
                        .background(VisualEffectBlur())
                    
                    // FLOATING DIRECTOR OVERLAY
                    if store.director.isAnalyzing {
                        DirectorOverlay()
                            .transition(.asymmetric(
                                insertion: .scale.combined(with: .opacity),
                                removal: .identity
                            ))
                            .zIndex(100)
                    }
                    
                    // MAGNETIC TIMELINE RULER
                    VStack {
                        MagneticTimelineRuler(scale: timelineScale)
                            .frame(height: 60)
                            .background(.regularMaterial)
                        Spacer()
                    }
                }
            }
            .toolbar {
                NeuralToolbar()
            }
        } detail: {
            // ADAPTIVE INSPECTOR
            AdaptiveInspector()
                .frame(width: inspectorWidth)
                .background(.ultraThinMaterial)
        }
        .navigationTitle("")
        .navigationSubtitle(store.project.name)
    }
}

// MARK: - INTELLIGENT SIDEBAR
struct IntelligentSidebar: View {
    @Binding var selection: NavigationItem?
    @EnvironmentObject var store: UnifiedStore
    @State private var searchText = ""
    @State private var isHovering = false
    
    var body: some View {
        VStack(spacing: 0) {
            // NEURAL SEARCH
            HStack {
                Image(systemName: "brain")
                    .foregroundStyle(.tertiary)
                    .symbolEffect(.pulse, options: .repeating)
                
                TextField("Ask Director AI...", text: $searchText)
                    .textFieldStyle(.plain)
                    .onSubmit {
                        store.director.processNaturalQuery(searchText)
                    }
            }
            .padding()
            .background(.quaternary.opacity(0.5))
            
            // PROJECT SECTION
            Section("Project") {
                NavigationLink(value: NavigationItem.timeline) {
                    Label("Neural Timeline", systemImage: "timeline.play")
                }
                NavigationLink(value: NavigationItem.director) {
                    Label {
                        Text("Director's Brain")
                        if store.director.hasInsights {
                            Badge(store.director.insightCount)
                        }
                    } icon: {
                        Image(systemName: "brain.fill")
                            .symbolRenderingMode(.multicolor)
                    }
                }
                NavigationLink(value: NavigationItem.transcription) {
                    Label("Transcription", systemImage: "captions.bubble")
                }
            }
            
            // INTELLIGENT CUTS
            Section("Intelligent Cuts") {
                ForEach(store.cuts.suggestions) { cut in
                    CutSuggestionRow(cut: cut)
                        .badge(cut.confidence)
                }
            }
            
            // STORY BEATS
            Section("Story Structure") {
                StoryBeatsVisualizer(beats: store.director.beats)
                    .frame(height: 120)
                    .padding(.horizontal)
            }
            
            // SHORTS GENERATOR
            Section("Viral Moments") {
                ShortsGeneratorPanel()
            }
            
            Spacer()
            
            // PROCESSING STATUS
            if store.isProcessing {
                ProcessingStatusBar()
                    .padding()
            }
        }
        .listStyle(.sidebar)
        .scrollContentBackground(.hidden)
    }
}

// MARK: - TIMELINE CANVAS - CORE INNOVATION
struct TimelineCanvas: View {
    @Binding var scale: CGFloat
    @EnvironmentObject var store: UnifiedStore
    @State private var selection = Set<ClipID>()
    @State private var currentTime: TimeInterval = 0
    @State private var isDragging = false
    @GestureState private var magnification: CGFloat = 1.0
    
    var body: some View {
        ScrollViewReader { proxy in
            ScrollView([.horizontal, .vertical]) {
                ZStack(alignment: .topLeading) {
                    // BASE TIMELINE
                    TimelineBackground()
                    
                    // VIDEO TRACK V1
                    TimelineTrack(
                        track: store.timeline.videoTracks[0],
                        height: 100,
                        offset: CGPoint(x: 0, y: 100)
                    )
                    
                    // DIRECTOR ANNOTATIONS LAYER
                    DirectorAnnotationsLayer(
                        beats: store.director.beats,
                        emphasis: store.director.emphasis,
                        height: 40,
                        offset: CGPoint(x: 0, y: 60)
                    )
                    
                    // AUDIO WAVEFORM
                    WaveformTrack(
                        audio: store.timeline.audioTracks[0],
                        height: 80,
                        offset: CGPoint(x: 0, y: 220)
                    )
                    
                    // TRANSCRIPTION TRACK
                    TranscriptionTrack(
                        segments: store.transcription.segments,
                        height: 60,
                        offset: CGPoint(x: 0, y: 320)
                    )
                    
                    // PLAYHEAD
                    PlayheadView(currentTime: $currentTime)
                        .zIndex(1000)
                    
                    // SELECTION RECTANGLE
                    if isDragging {
                        SelectionRectangle()
                    }
                }
                .frame(
                    width: max(3000, store.timeline.duration * scale),
                    height: 600
                )
                .scaleEffect(magnification)
                .gesture(
                    MagnificationGesture()
                        .updating($magnification) { value, state, _ in
                            state = value
                        }
                        .onEnded { value in
                            scale *= value
                            scale = min(max(10, scale), 500)
                        }
                )
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
        .onDrop(of: [.movie, .audio], isTargeted: nil) { providers in
            handleDrop(providers)
        }
    }
    
    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        // Handle video/audio file drops
        return true
    }
}

// MARK: - DIRECTOR'S BRAIN VIEW
struct DirectorBrainView: View {
    @EnvironmentObject var store: UnifiedStore
    @State private var selectedInsight: DirectorInsight?
    @State private var visualizationMode: VisualizationMode = .energy
    
    var body: some View {
        HSplitView {
            // INSIGHTS LIST
            insightsList
            
            // VISUALIZATION
            visualizationPanel
        }
        .frame(width: 1000, height: 700)
    }
    
    private var insightsList: some View {
        List(selection: $selectedInsight) {
            narrativeSection
            emotionalSection
            emphasisSection
        }
        .listStyle(.sidebar)
        .frame(minWidth: 300)
    }
    
    private var narrativeSection: some View {
        Section("Narrative Structure") {
            ForEach(store.director.beats.all) { beat in
                beatRow(beat)
            }
        }
    }
    
    private var emotionalSection: some View {
        Section("Emotional Dynamics") {
            TensionGraph(data: store.director.tensionCurve)
                .frame(height: 100)
                .listRowInsets(EdgeInsets())
            
            ForEach(store.director.tensionPeaks) { peak in
                tensionRow(peak)
            }
        }
    }
    
    private var emphasisSection: some View {
        Section("Emphasis Moments") {
            ForEach(store.director.emphasisMoments) { moment in
                EmphasisRow(moment: moment)
                    .tag(DirectorInsight.emphasis(moment))
            }
        }
    }
    
    private func beatRow(_ beat: StoryBeat) -> some View {
        HStack {
            Circle()
                .fill(beat.color)
                .frame(width: 8)
            
            VStack(alignment: .leading) {
                Text(beat.type.label)
                    .font(.headline)
                Text("Time Range")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
            
            Text("\(beat.confidence)%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
        .tag(DirectorInsight.beat(beat))
    }
    
    private func tensionRow(_ peak: TensionPeak) -> some View {
        HStack {
            Image(systemName: "flame.fill")
                .foregroundStyle(.orange.gradient)
            
            Text("High Tension")
            Text("Range")
                .foregroundStyle(.secondary)
            
            Spacer()
            
            ProgressView(value: peak.intensity)
                .frame(width: 60)
        }
        .tag(DirectorInsight.tension(peak))
    }
    
    private var visualizationPanel: some View {
        VStack {
            modePicker
            mainVisualization
            if let selectedInsight {
                insightDetailPanel(selectedInsight)
            }
        }
        .frame(minWidth: 600)
    }
    
    private var modePicker: some View {
        Picker("Visualization", selection: $visualizationMode) {
            Label("Energy", systemImage: "waveform.path.ecg").tag(VisualizationMode.energy)
            Label("Motion", systemImage: "figure.run").tag(VisualizationMode.motion)
            Label("Complexity", systemImage: "brain").tag(VisualizationMode.complexity)
            Label("Continuity", systemImage: "arrow.triangle.branch").tag(VisualizationMode.continuity)
        }
        .pickerStyle(.segmented)
        .padding()
    }
    
    private var mainVisualization: some View {
        GeometryReader { geometry in
            visualizationContent(geometry.size)
        }
        .background(.black.opacity(0.9))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding()
    }
    
    private func visualizationContent(_ size: CGSize) -> some View {
        Group {
            switch visualizationMode {
            case .energy:
                EnergyVisualization(
                    data: store.director.energyCurve,
                    beats: store.director.beats,
                    size: size
                )
            case .motion:
                MotionFlowVisualization(
                    vectors: store.director.motionVectors,
                    size: size
                )
                .drawingGroup(opaque: false, colorMode: .linear)
            case .complexity:
                ComplexityHeatmap(
                    data: store.director.complexityMap,
                    size: size
                )
            case .continuity:
                ContinuityFlowChart(
                    connections: store.director.continuityScores,
                    size: size
                )
            }
        }
    }
    
    private func insightDetailPanel(_ insight: DirectorInsight) -> some View {
        InsightDetailPanel(insight: insight)
            .frame(height: 200)
            .padding()
            .background(.regularMaterial)
    }
}

// MARK: - NEURAL TOOLBAR
struct NeuralToolbar: ToolbarContent {
    @EnvironmentObject var store: UnifiedStore
    @State private var showingExportOptions = false
    
    var body: some ToolbarContent {
        ToolbarItemGroup(placement: .principal) {
            // PLAYBACK CONTROLS
            PlaybackControls()
            
            Divider()
            
            // AI ACTIONS
            Menu {
                Button {
                    store.director.analyze()
                } label: {
                    Label("Analyze Story", systemImage: "brain")
                }
                
                Button {
                    store.cuts.generateSmart()
                } label: {
                    Label("Smart Cut", systemImage: "scissors")
                }
                
                Button {
                    store.shorts.generate()
                } label: {
                    Label("Generate Shorts", systemImage: "rectangle.stack")
                }
                
                Divider()
                
                Button {
                    store.transcribe()
                } label: {
                    Label("Transcribe", systemImage: "captions.bubble")
                }
                
                Button {
                    store.silence.detect()
                } label: {
                    Label("Remove Silence", systemImage: "waveform.badge.minus")
                }
            } label: {
                Label("AI Actions", systemImage: "sparkle")
                    .symbolEffect(.bounce, value: store.director.hasNewInsights)
            }
            .menuStyle(.borderlessButton)
            
            Divider()
            
            // RESOLVE INTEGRATION
            Button {
                store.resolve.sync()
            } label: {
                Label("Sync to Resolve", systemImage: "arrow.triangle.2.circlepath")
            }
            .disabled(!store.resolve.isConnected)
            
            // EXPORT
            Button {
                showingExportOptions.toggle()
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .popover(isPresented: $showingExportOptions) {
                ExportOptionsPanel()
            }
        }
        
        ToolbarItemGroup(placement: .automatic) {
            // VIEW OPTIONS
            TimelineViewOptions()
        }
    }
}

// MARK: - ADAPTIVE INSPECTOR
struct AdaptiveInspector: View {
    @EnvironmentObject var store: UnifiedStore
    @State private var selectedTab: InspectorTab = .properties
    
    var body: some View {
        VStack(spacing: 0) {
            // TAB SELECTOR
            Picker("Inspector", selection: $selectedTab) {
                ForEach(InspectorTab.allCases) { tab in
                    Label(tab.label, systemImage: tab.icon)
                        .tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .padding()
            
            Divider()
            
            // CONTENT
            ScrollView {
                switch selectedTab {
                case .properties:
                    PropertiesInspector()
                case .effects:
                    EffectsInspector()
                case .director:
                    DirectorInsightsPanel()
                case .transcription:
                    TranscriptionInspector()
                case .metadata:
                    MetadataInspector()
                }
            }
            .scrollIndicators(.hidden)
        }
    }
}

// MARK: - Supporting Types and Views (Mock implementations)
typealias ClipID = UUID

enum VisualizationMode: CaseIterable {
    case energy, motion, complexity, continuity
}

enum InspectorTab: String, CaseIterable, Identifiable {
    case properties, effects, director, transcription, metadata
    
    var id: String { rawValue }
    
    var label: String {
        switch self {
        case .properties: return "Properties"
        case .effects: return "Effects"
        case .director: return "Director"
        case .transcription: return "Transcription"
        case .metadata: return "Metadata"
        }
    }
    
    var icon: String {
        switch self {
        case .properties: return "slider.horizontal.3"
        case .effects: return "sparkle"
        case .director: return "brain"
        case .transcription: return "captions.bubble"
        case .metadata: return "info.circle"
        }
    }
}

enum DirectorInsight: Hashable {
    case beat(StoryBeat)
    case tension(TensionPeak)
    case emphasis(EmphasisMoment)
}

// MARK: - Mock View Components
struct VisualEffectBlur: View {
    var body: some View {
        Rectangle()
            .fill(.regularMaterial)
    }
}

struct DirectorOverlay: View {
    var body: some View {
        Text("AI Director Analyzing...")
            .padding()
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct MagneticTimelineRuler: View {
    let scale: CGFloat
    
    var body: some View {
        Rectangle()
            .fill(.regularMaterial)
            .overlay(
                Text("Timeline Ruler")
                    .font(.caption)
            )
    }
}

struct Badge: View {
    let content: String
    
    init(_ count: Int) {
        self.content = "\(count)"
    }
    
    init(_ text: String) {
        self.content = text
    }
    
    var body: some View {
        Text(content)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.red)
            .foregroundColor(.white)
            .clipShape(Capsule())
    }
}

struct CutSuggestionRow: View {
    let cut: CutSuggestion
    
    var body: some View {
        HStack {
            Text("Cut at \(cut.time, format: .number)")
            Spacer()
        }
        .badge(cut.confidence)
    }
}

struct StoryBeatsVisualizer: View {
    let beats: StoryBeats
    
    var body: some View {
        Rectangle()
            .fill(.blue.opacity(0.3))
            .overlay(
                Text("Story Beats")
                    .font(.caption)
            )
    }
}

struct ShortsGeneratorPanel: View {
    var body: some View {
        VStack {
            Text("Shorts Generator")
                .font(.headline)
            Button("Generate") { }
                .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

struct ProcessingStatusBar: View {
    var body: some View {
        HStack {
            ProgressView()
                .scaleEffect(0.8)
            Text("Processing...")
        }
    }
}

struct TimelineBackground: View {
    var body: some View {
        Rectangle()
            .fill(.black.opacity(0.05))
    }
}

struct TimelineTrack: View {
    let track: VideoTrack
    let height: CGFloat
    let offset: CGPoint
    
    var body: some View {
        Rectangle()
            .fill(.blue.opacity(0.6))
            .frame(height: height)
            .offset(x: offset.x, y: offset.y)
    }
}

struct DirectorAnnotationsLayer: View {
    let beats: StoryBeats
    let emphasis: [EmphasisMoment]
    let height: CGFloat
    let offset: CGPoint
    
    var body: some View {
        Rectangle()
            .fill(.purple.opacity(0.3))
            .frame(height: height)
            .offset(x: offset.x, y: offset.y)
    }
}

struct WaveformTrack: View {
    let audio: AudioTrack
    let height: CGFloat
    let offset: CGPoint
    
    var body: some View {
        Rectangle()
            .fill(.green.opacity(0.6))
            .frame(height: height)
            .offset(x: offset.x, y: offset.y)
    }
}

struct TranscriptionTrack: View {
    let segments: [TranscriptionSegment]
    let height: CGFloat
    let offset: CGPoint
    
    var body: some View {
        Rectangle()
            .fill(.orange.opacity(0.6))
            .frame(height: height)
            .offset(x: offset.x, y: offset.y)
    }
}

struct PlayheadView: View {
    @Binding var currentTime: TimeInterval
    
    var body: some View {
        Rectangle()
            .fill(.red)
            .frame(width: 2)
    }
}

struct SelectionRectangle: View {
    var body: some View {
        Rectangle()
            .stroke(.blue, lineWidth: 2)
            .fill(.blue.opacity(0.1))
    }
}

struct PlaybackControls: View {
    var body: some View {
        HStack {
            Button("◀◀") { }
            Button("▶") { }
            Button("▶▶") { }
        }
    }
}

struct TimelineViewOptions: View {
    var body: some View {
        Menu("View") {
            Button("Zoom In") { }
            Button("Zoom Out") { }
        }
    }
}

struct ExportOptionsPanel: View {
    var body: some View {
        VStack {
            Text("Export Options")
            Button("Export") { }
        }
        .padding()
    }
}

struct TensionGraph: View {
    let data: [Double]
    
    var body: some View {
        Rectangle()
            .fill(.orange.opacity(0.3))
    }
}

struct EmphasisRow: View {
    let moment: EmphasisMoment
    
    var body: some View {
        HStack {
            Text("Emphasis")
            Spacer()
            Text("\(moment.strength, format: .percent)")
        }
    }
}

struct EnergyVisualization: View {
    let data: [Double]
    let beats: StoryBeats
    let size: CGSize
    
    var body: some View {
        Rectangle()
            .fill(.blue.opacity(0.5))
    }
}

struct MotionFlowVisualization: View {
    let vectors: [MotionVector]
    let size: CGSize
    
    var body: some View {
        Rectangle()
            .fill(.green.opacity(0.5))
    }
}

struct ComplexityHeatmap: View {
    let data: [[Double]]
    let size: CGSize
    
    var body: some View {
        Rectangle()
            .fill(.red.opacity(0.5))
    }
}

struct ContinuityFlowChart: View {
    let connections: [ContinuityScore]
    let size: CGSize
    @State private var hoveredConnection: ContinuityScore?
    
    var body: some View {
        Canvas { context, size in
            // Draw motion flow vectors between shots
            for connection in connections {
                let continuityScore = connection.score
                
                // Draw connection arc
                var path = Path()
                let start = CGPoint(x: CGFloat(connection.fromShot) * 80, y: size.height/2)
                let end = CGPoint(x: CGFloat(connection.toShot) * 80, y: size.height/2)
                let control = CGPoint(x: (start.x + end.x)/2, y: size.height/2 - 50)
                
                path.move(to: start)
                path.addQuadCurve(to: end, control: control)
                
                let strokeColor = continuityGradientColor(score: continuityScore)
                context.stroke(path, with: .color(strokeColor), lineWidth: 3)
                
                // Draw score badge
                let badgePosition = CGPoint(x: (start.x + end.x)/2, y: control.y)
                context.draw(
                    Text("\(Int(continuityScore * 100))%")
                        .font(.caption.bold())
                        .foregroundColor(.white),
                    at: badgePosition
                )
            }
        }
        .background(.black.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
    
    private func continuityGradientColor(score: Double) -> Color {
        if score > 0.8 { return .green }
        if score > 0.5 { return .orange }
        return .red
    }
}

struct InsightDetailPanel: View {
    let insight: DirectorInsight
    
    var body: some View {
        Text("Insight Details")
            .padding()
    }
}

struct PropertiesInspector: View {
    var body: some View {
        Text("Properties Inspector")
            .padding()
    }
}

struct EffectsInspector: View {
    var body: some View {
        Text("Effects Inspector")
            .padding()
    }
}

struct DirectorInsightsPanel: View {
    @EnvironmentObject var store: UnifiedStore
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // CURRENT SEGMENT ANALYSIS
            if let current = store.timeline.currentSegment {
                GroupBox {
                    VStack(alignment: .leading) {
                        Label("Current Segment", systemImage: "play.square")
                            .font(.headline)
                        
                        Divider()
                        
                        LabeledContent("Story Phase") {
                            Badge(current.storyPhase.label)
                                .foregroundStyle(current.storyPhase.color)
                        }
                        
                        LabeledContent("Energy Level") {
                            EnergyMeter(value: current.energy)
                        }
                        
                        LabeledContent("Tension") {
                            TensionIndicator(value: current.tension)
                        }
                        
                        if let continuity = current.continuityScore {
                            LabeledContent("Continuity") {
                                Text("\(continuity, format: .percent)")
                                    .foregroundStyle(continuity > 0.8 ? .green : .orange)
                            }
                        }
                    }
                }
                
                // AI SUGGESTIONS
                GroupBox {
                    VStack(alignment: .leading) {
                        Label("AI Suggestions", systemImage: "sparkle")
                            .font(.headline)
                        
                        Divider()
                        
                        ForEach(current.suggestions) { suggestion in
                            HStack {
                                Image(systemName: suggestion.icon)
                                    .foregroundStyle(suggestion.priority.color)
                                
                                VStack(alignment: .leading) {
                                    Text(suggestion.title)
                                        .font(.subheadline)
                                    Text(suggestion.description)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                
                                Spacer()
                                
                                Button("Apply") {
                                    store.apply(suggestion)
                                }
                                .buttonStyle(.accessoryBar)
                            }
                        }
                    }
                }
            }
            
            // NEXT SUGGESTED CUT
            if let nextCut = store.director.nextSuggestedCut {
                GroupBox {
                    VStack(alignment: .leading) {
                        Label("Next Cut Point", systemImage: "scissors")
                            .font(.headline)
                        
                        Divider()
                        
                        HStack {
                            Text(nextCut.timecode)
                                .font(.system(.body, design: .monospaced))
                            
                            Spacer()
                            
                            Text(nextCut.reason)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        
                        Button {
                            store.timeline.jumpTo(nextCut.time)
                        } label: {
                            Label("Jump to Cut", systemImage: "arrow.right.square")
                        }
                        .controlSize(.small)
                    }
                }
            }
        }
        .padding()
    }
}

struct TranscriptionInspector: View {
    var body: some View {
        Text("Transcription Inspector")
            .padding()
    }
}

struct MetadataInspector: View {
    var body: some View {
        Text("Metadata Inspector")
            .padding()
    }
}

// MARK: - Advanced Meter Components
struct EnergyMeter: View {
    let value: Double
    
    var body: some View {
        HStack {
            ProgressView(value: value)
                .progressViewStyle(LinearProgressViewStyle(tint: energyColor))
                .frame(width: 100)
            Text("\(Int(value * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(energyColor)
        }
    }
    
    private var energyColor: Color {
        if value > 0.7 { return .red }
        if value > 0.4 { return .orange }
        return .green
    }
}

struct TensionIndicator: View {
    let value: Double
    
    var body: some View {
        HStack {
            // Simplified tension indicator
            HStack(spacing: 2) {
                ForEach(0..<5) { i in
                    RoundedRectangle(cornerRadius: 1)
                        .fill(i < Int(value * 5) ? tensionColor : .gray.opacity(0.3))
                        .frame(width: 6, height: CGFloat(8 + i * 2))
                }
            }
            .frame(width: 40, height: 16)
            
            Text("\(Int(value * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(tensionColor)
        }
    }
    
    private var tensionColor: Color {
        if value > 0.8 { return .red }
        if value > 0.5 { return .orange }
        return .blue
    }
}