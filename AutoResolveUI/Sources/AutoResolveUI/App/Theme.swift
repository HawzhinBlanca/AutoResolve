import SwiftUI

/// Design system matching DaVinci Resolve aesthetics
public struct UITheme {
    // MARK: - Colors
    
    public struct Colors {
        // Base palette
        public static let background = Color(red: 0.11, green: 0.11, blue: 0.12)  // #1C1C1F
        public static let surface = Color(red: 0.16, green: 0.16, blue: 0.18)     // #292A2D
        public static let surfaceLight = Color(red: 0.22, green: 0.22, blue: 0.24) // #383A3D
        
        // Text
        public static let textPrimary = Color(red: 0.92, green: 0.92, blue: 0.92)  // #EBEBEB
        public static let textSecondary = Color(red: 0.60, green: 0.60, blue: 0.62) // #999A9E
        public static let textDisabled = Color(red: 0.40, green: 0.40, blue: 0.42)  // #666869
        
        // Accent
        public static let accent = Color(red: 0.95, green: 0.26, blue: 0.21)  // #F44336 (red)
        public static let accentHover = Color(red: 1.0, green: 0.34, blue: 0.29)
        public static let selection = Color(red: 0.25, green: 0.47, blue: 0.85) // #4078D8
        
        // Timeline specific
        public static let playhead = Color(red: 1.0, green: 0.87, blue: 0.34)  // #FFDD56
        public static let clipVideo = Color(red: 0.40, green: 0.62, blue: 0.83) // #6699D3
        public static let clipAudio = Color(red: 0.48, green: 0.78, blue: 0.64) // #7BC8A4
        public static let clipTitle = Color(red: 0.69, green: 0.55, blue: 0.86) // #B08CDC
        
        // AI Annotations
        public static let silence = Color(red: 0.85, green: 0.33, blue: 0.33).opacity(0.5)  // Red
        public static let transcription = Color(red: 0.33, green: 0.65, blue: 0.85).opacity(0.5) // Blue
        public static let storyBeat = Color(red: 0.85, green: 0.65, blue: 0.33).opacity(0.5) // Orange
        public static let broll = Color(red: 0.33, green: 0.85, blue: 0.55).opacity(0.5) // Green
        
        // Status
        public static let success = Color(red: 0.30, green: 0.69, blue: 0.31) // #4CAF50
        public static let warning = Color(red: 1.0, green: 0.76, blue: 0.03)  // #FFC107
        public static let error = Color(red: 0.96, green: 0.26, blue: 0.21)   // #F44336
    }
    
    // MARK: - Sizes
    
    public struct Sizes {
        // Timeline
        public static let timelineHeaderHeight: CGFloat = 30
        public static let timelineTrackHeight: CGFloat = 60
        public static let timelineRulerHeight: CGFloat = 30
        public static let timelineScrollbarHeight: CGFloat = 15
        public static let playheadWidth: CGFloat = 2
        
        // Panels
        public static let sidebarWidth: CGFloat = 280
        public static let inspectorWidth: CGFloat = 320
        public static let toolbarHeight: CGFloat = 40
        public static let statusBarHeight: CGFloat = 24
        
        // Viewers
        public static let viewerMinWidth: CGFloat = 320
        public static let viewerAspectRatio: CGFloat = 16.0 / 9.0
        
        // Spacing
        public static let spacingXS: CGFloat = 4
        public static let spacingS: CGFloat = 8
        public static let spacingM: CGFloat = 12
        public static let spacingL: CGFloat = 16
        public static let spacingXL: CGFloat = 24
        
        // Corner radius
        public static let cornerRadiusS: CGFloat = 2
        public static let cornerRadiusM: CGFloat = 4
        public static let cornerRadiusL: CGFloat = 8
    }
    
    // MARK: - Typography
    
    public struct Typography {
        public static let largeTitle = Font.system(size: 24, weight: .semibold)
        public static let title = Font.system(size: 18, weight: .medium)
        public static let headline = Font.system(size: 14, weight: .medium)
        public static let body = Font.system(size: 12, weight: .regular)
        public static let caption = Font.system(size: 10, weight: .regular)
        public static let mono = Font.system(size: 11, weight: .regular, design: .monospaced)
        public static let timecode = Font.system(size: 14, weight: .medium, design: .monospaced)
    }
    
    // MARK: - Shadows
    
    public struct Shadows {
        public static let small = Shadow(
            color: Color.black.opacity(0.2),
            radius: 2,
            x: 0,
            y: 1
        )
        
        public static let medium = Shadow(
            color: Color.black.opacity(0.3),
            radius: 4,
            x: 0,
            y: 2
        )
        
        public static let large = Shadow(
            color: Color.black.opacity(0.4),
            radius: 8,
            x: 0,
            y: 4
        )
    }
    
    // MARK: - Animations
    
    public struct Animations {
        public static let fast = Animation.easeInOut(duration: 0.15)
        public static let normal = Animation.easeInOut(duration: 0.25)
        public static let slow = Animation.easeInOut(duration: 0.35)
        public static let spring = Animation.spring(response: 0.3, dampingFraction: 0.7)
    }
}

// MARK: - View Modifiers

public struct PanelStyle: ViewModifier {
    let padding: CGFloat
    
    public init(padding: CGFloat = UITheme.Sizes.spacingM) {
        self.padding = padding
    }
    
    public func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(UITheme.Colors.surface)
            .cornerRadius(UITheme.Sizes.cornerRadiusM)
    }
}

public struct ToolButtonStyle: ButtonStyle {
    let isSelected: Bool
    
    public init(isSelected: Bool = false) {
        self.isSelected = isSelected
    }
    
    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(UITheme.Typography.body)
            .foregroundColor(isSelected ? UITheme.Colors.textPrimary : UITheme.Colors.textSecondary)
            .padding(.horizontal, UITheme.Sizes.spacingS)
            .padding(.vertical, UITheme.Sizes.spacingXS)
            .background(
                RoundedRectangle(cornerRadius: UITheme.Sizes.cornerRadiusS)
                    .fill(isSelected ? UITheme.Colors.surfaceLight : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: UITheme.Sizes.cornerRadiusS)
                    .stroke(configuration.isPressed ? UITheme.Colors.accent : Color.clear, lineWidth: 1)
            )
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(UITheme.Animations.fast, value: configuration.isPressed)
    }
}

public extension View {
    func panelStyle(padding: CGFloat = UITheme.Sizes.spacingM) -> some View {
        modifier(PanelStyle(padding: padding))
    }
}

// MARK: - Icons

public struct Icons {
    public static let play = "play.fill"
    public static let pause = "pause.fill"
    public static let stop = "stop.fill"
    public static let forward = "forward.fill"
    public static let backward = "backward.fill"
    public static let skipForward = "forward.end.fill"
    public static let skipBackward = "backward.end.fill"
    
    public static let select = "cursorarrow"
    public static let blade = "scissors"
    public static let trim = "arrow.left.and.right"
    public static let slip = "arrow.left.arrow.right"
    public static let slide = "arrow.up.arrow.down"
    public static let ripple = "arrow.left.and.right.righttriangle.left.righttriangle.right"
    
    public static let zoomIn = "plus.magnifyingglass"
    public static let zoomOut = "minus.magnifyingglass"
    public static let zoomFit = "arrow.up.left.and.arrow.down.right"
    
    public static let markIn = "arrow.down.to.line"
    public static let markOut = "arrow.up.to.line"
    
    public static let silence = "speaker.slash.fill"
    public static let transcription = "text.bubble.fill"
    public static let storyBeat = "waveform"
    public static let broll = "photo.fill"
}