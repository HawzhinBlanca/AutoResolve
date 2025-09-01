import SwiftUI
import AutoResolveCore
import AIDirector

@main
struct AutoResolveApp: App {
    @StateObject private var appState = AppState()
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 1200, minHeight: 800)
                .onAppear {
                    appState.initialize()
                }
        }
        .windowToolbarStyle(.unified)
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Project") {
                    appState.createNewProject()
                }
                .keyboardShortcut("n", modifiers: .command)
                
                Button("Open Project...") {
                    appState.openProject()
                }
                .keyboardShortcut("o", modifiers: .command)
            }
            
            CommandGroup(before: .sidebar) {
                Button("Import Media...") {
                    appState.importMedia()
                }
                .keyboardShortcut("i", modifiers: .command)
                .disabled(appState.currentProject == nil)
            }
            
            CommandMenu("Timeline") {
                Button("Blade") {
                    appState.blade()
                }
                .keyboardShortcut("b", modifiers: .command)
                .disabled(!appState.canBlade)
                
                Button("Delete") {
                    appState.deleteSelection()
                }
                .keyboardShortcut(.delete)
                .disabled(!appState.hasSelection)
                
                Divider()
                
                Button("Zoom In") {
                    appState.zoomIn()
                }
                .keyboardShortcut("+", modifiers: .command)
                
                Button("Zoom Out") {
                    appState.zoomOut()
                }
                .keyboardShortcut("-", modifiers: .command)
            }
            
            CommandMenu("AIDirector") {
                Button("Analyze Timeline") {
                    Task {
                        await appState.runAIAnalysis()
                    }
                }
                .keyboardShortcut("a", modifiers: [.command, .shift])
                .disabled(appState.timeline == nil)
                
                Toggle("Show Suggestions", isOn: $appState.showAISuggestions)
                    .keyboardShortcut("s", modifiers: [.command, .shift])
                
                Divider()
                
                Button("AI Settings...") {
                    appState.showAISettings = true
                }
            }
        }
        
        Settings {
            SettingsView()
                .environmentObject(appState)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HSplitView {
            // Media Pool
            MediaPoolView()
                .frame(minWidth: 200, idealWidth: 250, maxWidth: 300)
            
            VSplitView {
                // Viewer
                ViewerView()
                    .frame(minHeight: 300)
                
                // Timeline
                TimelineView()
                    .frame(minHeight: 200, idealHeight: 400)
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button(action: { appState.playPause() }) {
                    Image(systemName: appState.isPlaying ? "pause.fill" : "play.fill")
                }
                .keyboardShortcut(.space, modifiers: [])
                
                Button(action: { appState.stepBackward() }) {
                    Image(systemName: "backward.frame.fill")
                }
                .keyboardShortcut(.leftArrow, modifiers: [])
                
                Button(action: { appState.stepForward() }) {
                    Image(systemName: "forward.frame.fill")
                }
                .keyboardShortcut(.rightArrow, modifiers: [])
            }
            
            ToolbarItemGroup(placement: .principal) {
                if let project = appState.currentProject {
                    Text(project.name)
                        .font(.headline)
                }
            }
            
            ToolbarItemGroup(placement: .automatic) {
                if appState.aiAnalyzing {
                    ProgressView()
                        .scaleEffect(0.7)
                }
                
                Button(action: { appState.showInspector.toggle() }) {
                    Image(systemName: "sidebar.right")
                }
            }
        }
        .sheet(isPresented: $appState.showAISettings) {
            AISettingsView()
                .environmentObject(appState)
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Set up menu bar
        NSApp.mainMenu?.autoenablesItems = true
        
        // Configure appearance
        NSApp.appearance = NSAppearance(named: .darkAqua)
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
    
    func applicationWillTerminate(_ notification: Notification) {
        // Save state
        AppState.shared?.saveState()
    }
}

// Placeholder views
struct MediaPoolView: View {
    var body: some View {
        Text("Media Pool")
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.gray.opacity(0.1))
    }
}

struct ViewerView: View {
    var body: some View {
        Text("Viewer")
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black)
    }
}

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        TabView {
            GeneralSettingsView()
                .tabItem {
                    Label("General", systemImage: "gear")
                }
            
            PerformanceSettingsView()
                .tabItem {
                    Label("Performance", systemImage: "speedometer")
                }
            
            AISettingsView()
                .tabItem {
                    Label("AI Director", systemImage: "brain")
                }
        }
        .frame(width: 500, height: 400)
    }
}

struct GeneralSettingsView: View {
    var body: some View {
        Form {
            Text("General Settings")
        }
        .padding()
    }
}

struct PerformanceSettingsView: View {
    var body: some View {
        Form {
            Text("Performance Settings")
        }
        .padding()
    }
}

struct AISettingsView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        Form {
            Section("AI Analysis") {
                Toggle("Enable AI suggestions", isOn: $appState.aiEnabled)
                Toggle("Auto-apply safe edits", isOn: $appState.aiAutoApply)
                
                Picker("Confidence threshold", selection: $appState.aiConfidenceThreshold) {
                    Text("Low (0.5)").tag(0.5)
                    Text("Medium (0.7)").tag(0.7)
                    Text("High (0.9)").tag(0.9)
                }
            }
            
            Section("Learning") {
                Toggle("Adaptive learning", isOn: $appState.aiLearningEnabled)
                
                if let stats = appState.aiStats {
                    LabeledContent("Keep rate") {
                        Text("\(Int(stats.keepRate * 100))%")
                    }
                    LabeledContent("Suggestions") {
                        Text("\(stats.totalSuggestions)")
                    }
                }
            }
        }
        .padding()
        .frame(width: 450, height: 350)
    }
}