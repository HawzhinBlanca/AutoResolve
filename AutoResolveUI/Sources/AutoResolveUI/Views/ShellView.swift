import SwiftUI
import AVKit

/// Main shell view with page tabs
public struct ShellView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    public var body: some View {
        VStack(spacing: 0) {
            // Top toolbar with page tabs
            ToolbarView()
                .frame(height: UITheme.Sizes.toolbarHeight)
            
            // Main content area
            ZStack {
                switch appState.currentPage {
                case .cut:
                    CutPage()
                case .edit:
                    EditPage()
                case .deliver:
                    DeliverPage()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            
            // Status bar
            StatusBar()
                .frame(height: UITheme.Sizes.statusBarHeight)
        }
        .background(UITheme.Colors.background)
        .focusable()
        .onKeyPress { keyPress in
            switch keyPress.key {
            case .space:
                appState.statusMessage = "Space pressed - toggling playback"
                appState.transport.togglePlayPause()
                return .handled
            case .rightArrow:
                let frames = keyPress.modifiers.contains(.option) ? 5 : 1
                appState.statusMessage = "→ pressed - forward \(frames) frame\(frames > 1 ? "s" : "")"
                appState.transport.seekByFrames(frames)
                return .handled
            case .leftArrow:
                let frames = keyPress.modifiers.contains(.option) ? -5 : -1
                appState.statusMessage = "← pressed - back \(abs(frames)) frame\(abs(frames) > 1 ? "s" : "")"
                appState.transport.seekByFrames(frames)
                return .handled
            case .delete:
                if keyPress.modifiers.contains(.shift) {
                    appState.statusMessage = "⇧⌫ pressed - ripple delete"
                    appState.rippleDeleteSelected()
                } else {
                    appState.statusMessage = "⌫ pressed - delete selected"
                    appState.deleteSelected()
                }
                return .handled
            default:
                break
            }
            
            switch keyPress.characters {
            case "j":
                appState.statusMessage = "J pressed - reverse play"
                appState.transport.jPressed()
                return .handled
            case "k":
                appState.statusMessage = "K pressed - pause"
                appState.transport.kPressed()  
                return .handled
            case "l":
                appState.statusMessage = "L pressed - forward play"
                appState.transport.lPressed()
                return .handled
            case "v":
                appState.statusMessage = "V pressed - select tool"
                appState.currentTool = .select
                return .handled
            case "b":
                appState.statusMessage = "B pressed - blade tool"
                appState.activateBladeTool()
                return .handled
            case "t":
                appState.statusMessage = "T pressed - trim tool"
                appState.currentTool = .trim
                return .handled
            case "y":
                appState.statusMessage = "Y pressed - slip tool"  
                appState.currentTool = .slip
                return .handled
            case "s":
                appState.statusMessage = "S pressed - slide tool"
                appState.currentTool = .slide
                return .handled
            case "n":
                appState.statusMessage = "N pressed - toggle snap"
                appState.snapSettings.snapEnabled.toggle()
                return .handled
            default:
                return .ignored
            }
        }
        .fileImporter(
            isPresented: $appState.showImporter,
            allowedContentTypes: [.movie, .video, .mpeg4Movie, .quickTimeMovie],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                for url in urls {
                    appState.importVideo(url: url)
                }
            }
        }
    }
}

/// Cut page with dual viewers and timeline
struct CutPage: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VSplitView {
            // Top: Dual viewers
            ViewerDock()
                .frame(minHeight: 300)
            
            // Bottom: Timeline
            TimelinePage()
                .frame(minHeight: 200)
        }
    }
}

/// Edit page with timeline focus
struct EditPage: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HSplitView {
            // Left: Media browser (simplified)
            MediaBrowser()
                .frame(width: UITheme.Sizes.sidebarWidth)
            
            VSplitView {
                // Top: Single program viewer
                ProgramViewer()
                    .frame(minHeight: 200)
                
                // Bottom: Timeline
                TimelinePage()
                    .frame(minHeight: 300)
            }
            
            // Right: Inspector
            InspectorView()
                .frame(width: UITheme.Sizes.inspectorWidth)
        }
    }
}

/// Deliver page with export options
struct DeliverPage: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        DeliverView()
    }
}

/// Program viewer
struct ProgramViewer: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    var body: some View {
        VStack(spacing: 0) {
            // Viewer header
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
            
            // Video player
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
                            Text("No video loaded")
                                .font(UITheme.Typography.body)
                                .foregroundColor(UITheme.Colors.textDisabled)
                        }
                    )
            }
            
            // Transport controls
            SimpleTransportControls()
                .padding(UITheme.Sizes.spacingS)
                .background(UITheme.Colors.surface)
        }
    }
}

/// Media browser (simplified)
struct MediaBrowser: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Media")
                .font(UITheme.Typography.headline)
                .foregroundColor(UITheme.Colors.textPrimary)
                .padding(UITheme.Sizes.spacingM)
            
            ScrollView {
                VStack(alignment: .leading, spacing: UITheme.Sizes.spacingS) {
                    if let url = appState.videoURL {
                        MediaItem(url: url)
                    }
                    
                    Button(action: { appState.showImporter = true }) {
                        Label("Import Media", systemImage: "plus.circle")
                            .font(UITheme.Typography.body)
                            .foregroundColor(UITheme.Colors.textSecondary)
                            .frame(maxWidth: .infinity)
                            .padding(UITheme.Sizes.spacingM)
                            .background(UITheme.Colors.surfaceLight)
                            .cornerRadius(UITheme.Sizes.cornerRadiusM)
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                .padding(UITheme.Sizes.spacingM)
            }
        }
        .background(UITheme.Colors.surface)
    }
}

struct MediaItem: View {
    let url: URL
    
    var body: some View {
        HStack {
            Image(systemName: "video.fill")
                .foregroundColor(UITheme.Colors.clipVideo)
            
            VStack(alignment: .leading) {
                Text(url.lastPathComponent)
                    .font(UITheme.Typography.body)
                    .foregroundColor(UITheme.Colors.textPrimary)
                    .lineLimit(1)
                
                Text(url.path)
                    .font(UITheme.Typography.caption)
                    .foregroundColor(UITheme.Colors.textSecondary)
                    .lineLimit(1)
            }
            
            Spacer()
        }
        .padding(UITheme.Sizes.spacingS)
        .background(UITheme.Colors.surfaceLight)
        .cornerRadius(UITheme.Sizes.cornerRadiusS)
    }
}

/// Simple transport controls for Shell view
struct SimpleTransportControls: View {
    @EnvironmentObject var transport: Transport
    
    var body: some View {
        HStack(spacing: UITheme.Sizes.spacingM) {
            // Skip backward
            Button(action: transport.jumpToStart) {
                Image(systemName: Icons.skipBackward)
            }
            .buttonStyle(ToolButtonStyle())
            
            // Play reverse (J)
            Button(action: transport.jPressed) {
                Image(systemName: Icons.backward)
            }
            .buttonStyle(ToolButtonStyle())
            
            // Pause (K)
            Button(action: transport.kPressed) {
                Image(systemName: transport.isPlaying ? Icons.pause : Icons.stop)
            }
            .buttonStyle(ToolButtonStyle())
            
            // Play forward (L)
            Button(action: transport.lPressed) {
                Image(systemName: Icons.forward)
            }
            .buttonStyle(ToolButtonStyle())
            
            // Skip forward
            Button(action: transport.jumpToEnd) {
                Image(systemName: Icons.skipForward)
            }
            .buttonStyle(ToolButtonStyle())
            
            Spacer()
            
            // Speed indicator
            Text("\(transport.playRate, specifier: "%.1f")×")
                .font(UITheme.Typography.mono)
                .foregroundColor(UITheme.Colors.textSecondary)
            
            // Loop indicator
            if transport.isLooping {
                Image(systemName: "repeat")
                    .foregroundColor(UITheme.Colors.accent)
            }
        }
    }
}