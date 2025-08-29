import SwiftUI
import AVKit

struct AutoResolveAppMain: App {
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
        }
    }
}

// Main entry point
AutoResolveAppMain.main()