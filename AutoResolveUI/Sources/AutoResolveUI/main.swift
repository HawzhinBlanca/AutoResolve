// AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
// Professional Video Editor with AI Integration

import SwiftUI
import AppKit
import UniformTypeIdentifiers

@main
struct AutoResolveApp: App {
    @StateObject private var projectStore = VideoProjectStore()
    @StateObject private var timelineViewModel = TimelineViewModel()
    @StateObject private var videoPlayerViewModel = VideoPlayerViewModel()
    @StateObject private var undoManager = ProfessionalUndoManager()
    @StateObject private var backendService = BackendService()
    @StateObject private var unifiedStore = UnifiedStore.shared
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    
    init() {
        // Configure app for professional video editing
        NSApplication.shared.appearance = NSAppearance(named: .darkAqua)
    }
    
    var body: some SwiftUI.Scene {
        WindowGroup(id: "main") {
            CompleteProfessionalTimeline()
                .frame(minWidth: 1400, minHeight: 900)
                .environmentObject(projectStore)
                .environmentObject(timelineViewModel)
                .environmentObject(videoPlayerViewModel)
                .environmentObject(undoManager)
                .environmentObject(backendService)
                .environmentObject(unifiedStore)
                .preferredColorScheme(.dark)
                .onAppear {
                    // Connect timeline to project store
                    timelineViewModel.project = projectStore.currentProject
                }
                .onChange(of: projectStore.currentProject) { project in
                    timelineViewModel.project = project
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unifiedCompact)
        .commands {
            ProfessionalMenuBarCommands(
                projectStore: projectStore,
                timelineViewModel: timelineViewModel,
                undoManager: undoManager
            )
        }
        
        // Timeline Window
        WindowGroup("Timeline", id: "timeline") {
            TimelineWindow()
                .environmentObject(projectStore)
                .environmentObject(timelineViewModel)
                .environmentObject(undoManager)
                .frame(minWidth: 800, minHeight: 400)
        }
        
        // Viewer Window
        WindowGroup("Viewer", id: "viewer") {
            ViewerWindow()
                .environmentObject(projectStore)
                .environmentObject(videoPlayerViewModel)
                .frame(width: 640, height: 360)
        }
        
        // Inspector Window
        WindowGroup("Inspector", id: "inspector") {
            InspectorWindow()
                .environmentObject(projectStore)
                .environmentObject(timelineViewModel)
                .frame(width: 300, height: 600)
        }
        
        // AI Director Brain Window
        Window("AI Director Brain", id: "director") {
            DirectorBrainView()
                .environmentObject(projectStore)
                .environmentObject(backendService)
                .environmentObject(unifiedStore)
                .frame(width: 800, height: 600)
        }
        .windowResizability(.contentSize)
        
        // Settings Window
        Settings {
            NeuralSettingsView()
                .environmentObject(projectStore)
                .environmentObject(backendService)
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Configure app for professional video editing
        configureAppearance()
        configurePerformance()
    }
    
    private func configureAppearance() {
        // Set up professional dark theme
        if #available(macOS 10.14, *) {
            NSApplication.shared.appearance = NSAppearance(named: .darkAqua)
        }
    }
    
    private func configurePerformance() {
        // Configure for video editing performance
        // Enable metal rendering, disable unnecessary animations
    }
}

// MARK: - Supporting Window Views
struct TimelineWindow: View {
    @EnvironmentObject private var projectStore: VideoProjectStore
    @EnvironmentObject private var timelineViewModel: TimelineViewModel
    @EnvironmentObject private var undoManager: ProfessionalUndoManager
    
    var body: some View {
        VStack {
            if let project = projectStore.currentProject {
                // Professional timeline view will be implemented in Phase 2
                VStack {
                    Text("Professional Timeline")
                        .font(.title2)
                        .foregroundColor(.white)
                    
                    Text("Video Tracks: \(project.timeline.videoTracks.count)")
                    Text("Audio Tracks: \(project.timeline.audioTracks.count)")
                    Text("Duration: \(project.timeline.duration, specifier: "%.1f")s")
                    
                    HStack {
                        Button("Play") {
                            timelineViewModel.isPlaying.toggle()
                        }
                        .keyboardShortcut(.space, modifiers: [])
                        
                        Button("Zoom Fit") {
                            timelineViewModel.zoomToFit()
                        }
                        
                        Button("Add Video Track") {
                            // Will be implemented
                        }
                    }
                    .padding()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black.opacity(0.1))
            } else {
                VStack {
                    Text("No Project Open")
                        .font(.title2)
                        .foregroundColor(.secondary)
                    
                    Button("Create New Project") {
                        projectStore.createNewProject()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Timeline")
        .toolbar {
            ToolbarItemGroup(placement: .automatic) {
                Button(timelineViewModel.isPlaying ? "⏸" : "▶️") {
                    timelineViewModel.isPlaying.toggle()
                }
                
                Slider(value: $timelineViewModel.playhead, in: 0...(projectStore.currentProject?.timeline.duration ?? 1))
                    .frame(width: 200)
                
                Text("\(timelineViewModel.playhead, specifier: "%.1f")s")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
}

struct ViewerWindow: View {
    @EnvironmentObject private var projectStore: VideoProjectStore
    @EnvironmentObject private var videoPlayerViewModel: VideoPlayerViewModel
    
    var body: some View {
        VStack {
            // Video player view - will be implemented with AVPlayerView in Phase 2
            Rectangle()
                .fill(Color.black)
                .aspectRatio(16/9, contentMode: .fit)
                .overlay(
                    VStack {
                        Image(systemName: "play.circle")
                            .font(.system(size: 64))
                            .foregroundColor(.white.opacity(0.8))
                        
                        Text("Video Viewer")
                            .foregroundColor(.white)
                            .font(.title2)
                        
                        if let project = projectStore.currentProject {
                            Text(project.name)
                                .foregroundColor(.white.opacity(0.7))
                                .font(.caption)
                        }
                    }
                )
                .cornerRadius(8)
            
            // Playback controls
            HStack {
                Button("⏪") { videoPlayerViewModel.seek(to: max(0, videoPlayerViewModel.currentTime - 10)) }
                Button(videoPlayerViewModel.isPlaying ? "⏸" : "▶️") {
                    videoPlayerViewModel.togglePlayPause()
                }
                Button("⏩") { videoPlayerViewModel.seek(to: min(videoPlayerViewModel.duration, videoPlayerViewModel.currentTime + 10)) }
                
                Spacer()
                
                Slider(value: $videoPlayerViewModel.currentTime, in: 0...videoPlayerViewModel.duration)
                    .frame(width: 200)
                
                Button(videoPlayerViewModel.isMuted ? "🔇" : "🔊") {
                    videoPlayerViewModel.toggleMute()
                }
            }
            .padding()
        }
        .navigationTitle("Viewer")
        .toolbar {
            ToolbarItemGroup(placement: .automatic) {
                Button("Fit") {
                    // Fit to window
                }
                
                Button("Full Screen") {
                    // Enter full screen
                }
            }
        }
    }
}

struct InspectorWindow: View {
    @EnvironmentObject private var projectStore: VideoProjectStore
    @EnvironmentObject private var timelineViewModel: TimelineViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if timelineViewModel.selectedClips.isEmpty {
                VStack {
                    Image(systemName: "info.circle")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    
                    Text("No Selection")
                        .font(.title2)
                        .foregroundColor(.secondary)
                    
                    Text("Select clips in the timeline to view properties")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding()
            } else {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Selected Clips: \(timelineViewModel.selectedClips.count)")
                        .font(.headline)
                        .padding(.horizontal)
                        .padding(.top)
                    
                    // Inspector panels - will be implemented in Phase 3
                    TabView {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Video Properties")
                                .font(.headline)
                            
                            Group {
                                HStack {
                                    Text("Transform")
                                    Spacer()
                                    Button("Reset") { }
                                }
                                
                                VStack(alignment: .leading) {
                                    Text("Position")
                                    HStack {
                                        TextField("X", value: .constant(0.0), format: .number)
                                            .textFieldStyle(.roundedBorder)
                                        TextField("Y", value: .constant(0.0), format: .number)
                                            .textFieldStyle(.roundedBorder)
                                    }
                                }
                                
                                VStack(alignment: .leading) {
                                    Text("Scale")
                                    HStack {
                                        TextField("W", value: .constant(100.0), format: .number)
                                            .textFieldStyle(.roundedBorder)
                                        TextField("H", value: .constant(100.0), format: .number)
                                            .textFieldStyle(.roundedBorder)
                                    }
                                }
                            }
                            
                            Spacer()
                        }
                        .padding()
                        .tabItem {
                            Image(systemName: "video")
                            Text("Video")
                        }
                        
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Audio Properties")
                                .font(.headline)
                            
                            Group {
                                VStack(alignment: .leading) {
                                    Text("Volume")
                                    Slider(value: .constant(1.0), in: 0...2)
                                }
                                
                                VStack(alignment: .leading) {
                                    Text("Pan")
                                    Slider(value: .constant(0.0), in: -1...1)
                                }
                                
                                Toggle("Mute", isOn: .constant(false))
                            }
                            
                            Spacer()
                        }
                        .padding()
                        .tabItem {
                            Image(systemName: "speaker.wave.2")
                            Text("Audio")
                        }
                        
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Effects")
                                .font(.headline)
                            
                            Text("No effects applied")
                                .foregroundColor(.secondary)
                            
                            Button("Add Effect") {
                                // Add effect
                            }
                            .buttonStyle(.borderedProminent)
                            
                            Spacer()
                        }
                        .padding()
                        .tabItem {
                            Image(systemName: "sparkles")
                            Text("Effects")
                        }
                    }
                }
            }
        }
        .navigationTitle("Inspector")
        .frame(minWidth: 300)
    }
}

// MARK: - Notification Names
extension Notification.Name {
    static let importMediaRequested = Notification.Name("importMediaRequested")
    static let exportProjectRequested = Notification.Name("exportProjectRequested")
}


struct NeuralSettingsView: View {
    @EnvironmentObject private var projectStore: VideoProjectStore
    @EnvironmentObject private var backendService: BackendService
    @State private var useVJEPA: Bool = true
    @State private var enableAutoAnalysis: Bool = true
    @State private var showAdvancedSettings: Bool = false
    
    var body: some View {
        VStack(spacing: 20) {
            Text("AI Director Settings")
                .font(.title)
                .fontWeight(.semibold)
            
            VStack(alignment: .leading, spacing: 16) {
                Text("Neural Processing")
                        .font(.headline)
                    
                    Toggle("Use V-JEPA Model", isOn: $useVJEPA)
                        .help("V-JEPA provides more accurate video understanding")
                    
                    Toggle("Auto-analyze new footage", isOn: $enableAutoAnalysis)
                        .help("Automatically analyze footage when imported")
                    
                    Divider()
                    
                    Text("Backend Connection")
                        .font(.headline)
                    
                    HStack {
                        Circle()
                            .fill(backendService.isConnected ? .green : .red)
                            .frame(width: 12, height: 12)
                        
                        Text(backendService.isConnected ? "Connected" : "Disconnected")
                            .font(.caption)
                        
                        Spacer()
                        
                        Button("Reconnect") {
                            Task {
                                await backendService.checkHealth()
                            }
                        }
                        .disabled(backendService.isConnected)
                    }
                    
                    if !backendService.isConnected {
                        Text("Backend service not available. Some AI features may be limited.")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                    
                    Divider()
                    
                    Text("Project Settings")
                        .font(.headline)
                    
                    if let project = projectStore.currentProject {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Project:")
                                Text(project.name)
                                    .fontWeight(.medium)
                                Spacer()
                            }
                            
                            HStack {
                                Text("Resolution:")
                                Text(project.settings.resolution.rawValue)
                                Spacer()
                            }
                            
                            HStack {
                                Text("Frame Rate:")
                                Text("\(project.settings.frameRate.rawValue, specifier: "%.3f") fps")
                                Spacer()
                            }
                            
                            HStack {
                                Text("Auto-save:")
                                Toggle("", isOn: .constant(project.settings.autoSave))
                                    .labelsHidden()
                                Spacer()
                            }
                        }
                        .font(.caption)
                        .padding(.leading)
                    } else {
                        Text("No project open")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.leading)
                    }
                }
                
                Divider()
                
                DisclosureGroup("Advanced Settings", isExpanded: $showAdvancedSettings) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Performance")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Memory Usage Limit")
                                Slider(value: .constant(8.0), in: 4.0...32.0, step: 1.0) {
                                    Text("Memory Limit")
                                } minimumValueLabel: {
                                    Text("4GB")
                                } maximumValueLabel: {
                                    Text("32GB")
                                }
                                Text("Current: 8.0 GB")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Toggle("Use GPU acceleration", isOn: .constant(true))
                            Toggle("Enable proxy media", isOn: .constant(false))
                            
                            Text("Quality Settings")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .padding(.top)
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Analysis Quality")
                                Picker("Analysis Quality", selection: .constant("High")) {
                                    Text("Low").tag("Low")
                                    Text("Medium").tag("Medium")
                                    Text("High").tag("High")
                                    Text("Ultra").tag("Ultra")
                                }
                                .pickerStyle(.segmented)
                            }
                            
                            Toggle("High-precision timestamps", isOn: .constant(true))
                            Toggle("Export with metadata", isOn: .constant(true))
                    }
                    .padding(.top, 8)
                }
            
            Spacer()
            
            HStack {
                Button("Reset to Defaults") {
                    resetToDefaults()
                }
                .buttonStyle(.bordered)
                
                Spacer()
                
                Button("Apply Settings") {
                    applySettings()
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }
    
    private func resetToDefaults() {
        // Reset settings to defaults
    }
    
    private func applySettings() {
        // Apply settings to project and backend
    }
}
