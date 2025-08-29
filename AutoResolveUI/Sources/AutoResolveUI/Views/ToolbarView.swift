import SwiftUI

public struct ToolbarView: View {
    @EnvironmentObject var appState: AppState
    
    public var body: some View {
        HStack(spacing: 0) {
            // Page tabs
            HStack(spacing: 0) {
                ForEach(AppState.Page.allCases, id: \.self) { page in
                    PageTab(page: page, isSelected: appState.currentPage == page)
                        .onTapGesture {
                            appState.currentPage = page
                        }
                }
            }
            
            Spacer()
            
            // Edit tools
            if appState.currentPage == .edit || appState.currentPage == .cut {
                HStack(spacing: UITheme.Sizes.spacingXS) {
                    ForEach(AppState.EditTool.allCases, id: \.self) { tool in
                        ToolButton(
                            tool: tool,
                            isSelected: appState.currentTool == tool
                        ) {
                            appState.currentTool = tool
                        }
                    }
                }
                .padding(.horizontal, UITheme.Sizes.spacingM)
                
                Divider()
                    .frame(height: 20)
                    .padding(.horizontal, UITheme.Sizes.spacingM)
            }
            
            // AI controls
            HStack(spacing: UITheme.Sizes.spacingS) {
                Toggle("", isOn: $appState.showSilence)
                    .toggleStyle(IconToggleStyle(icon: Icons.silence, color: UITheme.Colors.silence))
                
                Toggle("", isOn: $appState.showTranscription)
                    .toggleStyle(IconToggleStyle(icon: Icons.transcription, color: UITheme.Colors.transcription))
                
                Toggle("", isOn: $appState.showStoryBeats)
                    .toggleStyle(IconToggleStyle(icon: Icons.storyBeat, color: UITheme.Colors.storyBeat))
                
                Toggle("", isOn: $appState.showBRoll)
                    .toggleStyle(IconToggleStyle(icon: Icons.broll, color: UITheme.Colors.broll))
            }
            .padding(.horizontal, UITheme.Sizes.spacingM)
            
            // Zoom controls
            HStack(spacing: UITheme.Sizes.spacingXS) {
                Button(action: appState.zoomOut) {
                    Image(systemName: Icons.zoomOut)
                }
                .buttonStyle(ToolButtonStyle())
                
                Text("\(Int(appState.zoomLevel * 100))%")
                    .font(UITheme.Typography.mono)
                    .foregroundColor(UITheme.Colors.textSecondary)
                    .frame(width: 50)
                
                Button(action: appState.zoomIn) {
                    Image(systemName: Icons.zoomIn)
                }
                .buttonStyle(ToolButtonStyle())
                
                Button(action: appState.zoomToFit) {
                    Image(systemName: Icons.zoomFit)
                }
                .buttonStyle(ToolButtonStyle())
            }
            .padding(.trailing, UITheme.Sizes.spacingM)
        }
        .frame(height: UITheme.Sizes.toolbarHeight)
        .background(UITheme.Colors.surface)
    }
}

struct PageTab: View {
    let page: AppState.Page
    let isSelected: Bool
    
    var body: some View {
        Text(page.rawValue)
            .font(UITheme.Typography.headline)
            .foregroundColor(isSelected ? UITheme.Colors.textPrimary : UITheme.Colors.textSecondary)
            .padding(.horizontal, UITheme.Sizes.spacingL)
            .frame(height: UITheme.Sizes.toolbarHeight)
            .background(isSelected ? UITheme.Colors.surfaceLight : Color.clear)
            .overlay(
                Rectangle()
                    .fill(isSelected ? UITheme.Colors.accent : Color.clear)
                    .frame(height: 2),
                alignment: .bottom
            )
    }
}

struct ToolButton: View {
    let tool: AppState.EditTool
    let isSelected: Bool
    let action: () -> Void
    
    var iconName: String {
        switch tool {
        case .select: return Icons.select
        case .blade: return Icons.blade
        case .trim: return Icons.trim
        case .slip: return Icons.slip
        case .slide: return Icons.slide
        case .ripple: return Icons.ripple
        }
    }
    
    var body: some View {
        Button(action: action) {
            Image(systemName: iconName)
                .font(UITheme.Typography.body)
                .frame(width: 28, height: 28)
        }
        .buttonStyle(ToolButtonStyle(isSelected: isSelected))
        .help(tool.rawValue)
    }
}

struct IconToggleStyle: ToggleStyle {
    let icon: String
    let color: Color
    
    func makeBody(configuration: Configuration) -> some View {
        Button(action: { configuration.isOn.toggle() }) {
            Image(systemName: icon)
                .font(UITheme.Typography.body)
                .foregroundColor(configuration.isOn ? color : UITheme.Colors.textDisabled)
                .frame(width: 28, height: 28)
        }
        .buttonStyle(ToolButtonStyle(isSelected: configuration.isOn))
    }
}