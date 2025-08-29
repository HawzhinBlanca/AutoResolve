import SwiftUI
import AVKit

struct AutoResolveApp: App {
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ShellView()
                .environmentObject(appState)
                .environmentObject(appState.transport)
                .environmentObject(appState.backendClient)
                .frame(minWidth: 1400, minHeight: 900)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentSize)
        .commands {
            // File menu
            CommandGroup(replacing: .newItem) {
                Button("Import Video...") {
                    appState.showImporter = true
                }
                .keyboardShortcut("i", modifiers: [.command])
                
                Divider()
                
                Button("Export FCPXML...") {
                    Task { await appState.exportFCPXML() }
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                
                Button("Export EDL...") {
                    Task { await appState.exportEDL() }
                }
            }
            
            // Edit menu
            CommandGroup(after: .pasteboard) {
                Divider()
                Button("Blade") {
                    appState.activateBladeTool()
                }
                .keyboardShortcut("b", modifiers: [])
                
                Button("Select") {
                    appState.activateSelectTool()
                }
                .keyboardShortcut("a", modifiers: [])
            }
            
            // Playback menu
            CommandMenu("Playback") {
                Button("Play/Pause") {
                    appState.transport.togglePlayPause()
                }
                .keyboardShortcut(.space, modifiers: [])
                
                Button("Play Forward") {
                    appState.transport.lPressed()
                }
                .keyboardShortcut("l", modifiers: [])
                
                Button("Pause") {
                    appState.transport.kPressed()
                }
                .keyboardShortcut("k", modifiers: [])
                
                Button("Play Reverse") {
                    appState.transport.jPressed()
                }
                .keyboardShortcut("j", modifiers: [])
                
                Divider()
                
                Button("Previous Frame") {
                    appState.transport.seekByFrames(-1)
                }
                .keyboardShortcut(.leftArrow, modifiers: [])
                
                Button("Next Frame") {
                    appState.transport.seekByFrames(1)
                }
                .keyboardShortcut(.rightArrow, modifiers: [])
                
                Divider()
                
                Button("Mark In") {
                    appState.transport.setLoopIn()
                }
                .keyboardShortcut("i", modifiers: [])
                
                Button("Mark Out") {
                    appState.transport.setLoopOut()
                }
                .keyboardShortcut("o", modifiers: [])
            }
            
            // Timeline menu
            CommandMenu("Timeline") {
                Button("Zoom In") {
                    appState.zoomIn()
                }
                .keyboardShortcut("+", modifiers: [.command])
                
                Button("Zoom Out") {
                    appState.zoomOut()
                }
                .keyboardShortcut("-", modifiers: [.command])
                
                Button("Zoom to Fit") {
                    appState.zoomToFit()
                }
                .keyboardShortcut("0", modifiers: [.command])
                
                Divider()
                
                Button("Snap to Frames") {
                    appState.snapSettings.snapToFrames.toggle()
                }
                
                Button("Snap to Clips") {
                    appState.snapSettings.snapToClips.toggle()
                }
                
                Button("Snap to Markers") {
                    appState.snapSettings.snapToMarkers.toggle()
                }
            }
            
            // AI menu
            CommandMenu("AI") {
                Button("Detect Silence") {
                    Task { await appState.runSilenceDetection() }
                }
                
                Button("Transcribe") {
                    Task { await appState.runTranscription() }
                }
                
                Button("Analyze Story Beats") {
                    Task { await appState.runStoryBeats() }
                }
                
                Button("Select B-Roll") {
                    Task { await appState.runBRollSelection() }
                }
                
                Divider()
                
                Button("Run Full Pipeline") {
                    Task { await appState.runFullPipeline() }
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            }
        }
    }
}