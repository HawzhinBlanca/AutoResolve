import SwiftUI
import AVKit

/// Dual viewer dock for Cut page
public struct ViewerDock: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    public var body: some View {
        HStack(spacing: 1) {
            // Source viewer
            SourceViewer()
            
            // Program viewer  
            ProgramViewerDock()
        }
        .background(UITheme.Colors.background)
    }
}

struct SourceViewer: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Source")
                    .font(UITheme.Typography.headline)
                    .foregroundColor(UITheme.Colors.textSecondary)
                
                Spacer()
                
                Button(action: markIn) {
                    Image(systemName: "arrowtriangle.right.fill")
                }
                .buttonStyle(.bordered)
                
                Button(action: markOut) {
                    Image(systemName: "arrowtriangle.left.fill")
                }
                .buttonStyle(.bordered)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(UITheme.Colors.surface)
            
            // Video area - connect to main player
            if let player = appState.player {
                VideoPlayer(player: player)
                    .background(Color.black)
                    .onAppear {
                        // Sync transport with this player
                        transport.setPlayer(player)
                    }
            } else {
                Rectangle()
                    .fill(Color.black)
                    .overlay(
                        VStack {
                            Image(systemName: "video.badge.plus")
                                .font(.largeTitle)
                                .foregroundColor(.gray)
                            Text("Import Video")
                                .font(.headline)
                                .foregroundColor(.gray)
                            Text("Drag video here or use Cmd+I")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    )
                    .onDrop(of: [.fileURL, .movie, .video], isTargeted: nil) { providers in
                        handleSourceDrop(providers: providers)
                    }
            }
        }
    }
    
    func handleSourceDrop(providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            if provider.canLoadObject(ofClass: URL.self) {
                provider.loadObject(ofClass: URL.self) { url, error in
                    guard let url = url, error == nil else { return }
                    
                    Task { @MainActor in
                        appState.importVideo(url: url)
                    }
                }
                return true
            }
        }
        return false
    }
    
    func markIn() {
        transport.setLoopIn()
        appState.statusMessage = "In point set at \(appState.timebase.timecodeFromTime(transport.currentTime))"
    }
    
    func markOut() {
        transport.setLoopOut()
        appState.statusMessage = "Out point set at \(appState.timebase.timecodeFromTime(transport.currentTime))"
    }
}

struct ProgramViewerDock: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Program")
                    .font(UITheme.Typography.headline)
                    .foregroundColor(UITheme.Colors.textSecondary)
                
                Spacer()
                
                Text(appState.timebase.timecodeFromTime(transport.currentTime))
                    .font(UITheme.Typography.timecode)
                    .foregroundColor(UITheme.Colors.textPrimary)
            }
            .padding(.horizontal, UITheme.Sizes.spacingM)
            .padding(.vertical, UITheme.Sizes.spacingXS)
            .background(UITheme.Colors.surface)
            
            // Video area
            if let player = appState.player {
                VideoPlayer(player: player)
                    .background(Color.black)
            } else {
                Rectangle()
                    .fill(Color.black)
                    .overlay(
                        VStack {
                            Image(systemName: "video")
                                .font(.system(size: 48))
                                .foregroundColor(UITheme.Colors.textDisabled)
                            Text("Import video to begin")
                                .font(UITheme.Typography.body)
                                .foregroundColor(UITheme.Colors.textDisabled)
                            
                            Button("Import Video") {
                                appState.showImporter = true
                            }
                            .buttonStyle(.borderedProminent)
                            .padding(.top)
                        }
                    )
            }
        }
    }
}