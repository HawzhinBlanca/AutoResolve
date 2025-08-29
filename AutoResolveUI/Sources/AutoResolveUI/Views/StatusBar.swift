import SwiftUI

public struct StatusBar: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    @EnvironmentObject var backendClient: BackendClient
    
    public var body: some View {
        HStack(spacing: UITheme.Sizes.spacingM) {
            // Timecode display
            HStack(spacing: UITheme.Sizes.spacingS) {
                Text("TC:")
                    .font(UITheme.Typography.caption)
                    .foregroundColor(UITheme.Colors.textSecondary)
                
                Text(appState.timebase.timecodeFromTime(transport.currentTime))
                    .font(UITheme.Typography.timecode)
                    .foregroundColor(UITheme.Colors.textPrimary)
                
                Text("/")
                    .foregroundColor(UITheme.Colors.textDisabled)
                
                Text(appState.timebase.timecodeFromTime(transport.duration))
                    .font(UITheme.Typography.timecode)
                    .foregroundColor(UITheme.Colors.textSecondary)
            }
            
            Divider()
                .frame(height: 12)
            
            // FPS indicator
            Text("\(Int(appState.timebase.fps)) fps")
                .font(UITheme.Typography.caption)
                .foregroundColor(UITheme.Colors.textSecondary)
            
            Divider()
                .frame(height: 12)
            
            // Status message
            HStack(spacing: UITheme.Sizes.spacingXS) {
                if appState.isProcessing {
                    ProgressView()
                        .scaleEffect(0.5)
                        .frame(width: 12, height: 12)
                }
                
                Text(appState.statusMessage)
                    .font(UITheme.Typography.caption)
                    .foregroundColor(UITheme.Colors.textSecondary)
                    .lineLimit(1)
            }
            
            Spacer()
            
            // Transport controls for testing
            HStack(spacing: UITheme.Sizes.spacingS) {
                Button(action: transport.togglePlayPause) {
                    Image(systemName: transport.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 12))
                }
                .buttonStyle(.bordered)
                .help("Play/Pause (Space)")
                
                Text("\(transport.playRate, specifier: "%.1f")×")
                    .font(UITheme.Typography.caption)
                    .foregroundColor(UITheme.Colors.textSecondary)
            }
            
            Divider()
                .frame(height: 12)
            
            // Backend status
            HStack(spacing: UITheme.Sizes.spacingXS) {
                Circle()
                    .fill(backendClient.isConnected ? UITheme.Colors.success : UITheme.Colors.error)
                    .frame(width: 6, height: 6)
                
                Text(backendClient.isConnected ? "Backend Connected" : "Backend Offline")
                    .font(UITheme.Typography.caption)
                    .foregroundColor(UITheme.Colors.textSecondary)
            }
        }
        .padding(.horizontal, UITheme.Sizes.spacingM)
        .frame(height: UITheme.Sizes.statusBarHeight)
        .background(UITheme.Colors.surface)
    }
}