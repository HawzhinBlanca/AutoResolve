import SwiftUI
import Foundation

public struct AccessibilitySupport {
    public static func configureForTimeline(_ view: some View) -> some View {
        view
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Timeline view with video clips")
            .accessibilityHint("Use arrow keys to navigate, Space to play/pause")
    }
    
    public static func configureForClip(_ clip: SimpleTimelineClip) -> some View {
        Rectangle()
            .accessibilityLabel("Video clip: \(clip.name)")
            .accessibilityHint("Duration: \(String(format: "%.1f", clip.duration)) seconds")
            .accessibilityValue("Start time: \(String(format: "%.1f", clip.startTime))")
            .accessibilityAddTraits(.isButton)
    }
    
    public static func configureForTransport(_ isPlaying: Bool) -> some View {
        EmptyView()
            .accessibilityLabel(isPlaying ? "Pause video" : "Play video")
            .accessibilityHint("Press Space to toggle playback")
            .accessibilityAddTraits(.isButton)
    }
    
    public static func configureForTool(_ tool: AppState.EditTool, isSelected: Bool) -> some View {
        EmptyView()
            .accessibilityLabel("\(tool.rawValue) tool")
            .accessibilityHint(isSelected ? "Currently selected" : "Tap to select")
            .accessibilityValue(isSelected ? "Selected" : "Not selected")
            .accessibilityAddTraits(.isButton)
    }
}

public struct HighContrastColors {
    public static let background = Color.black
    public static let surface = Color(red: 0.1, green: 0.1, blue: 0.1)
    public static let textPrimary = Color.white
    public static let textSecondary = Color(red: 0.8, green: 0.8, blue: 0.8)
    public static let accent = Color.yellow
    public static let selection = Color.cyan
    public static let playhead = Color.red
    public static let success = Color.green
    public static let error = Color.red
    public static let warning = Color.orange
}

public struct AccessibilitySettings: ObservableObject {
    @Published public var highContrastMode = false
    @Published public var largerText = false
    @Published public var reduceMotion = false
    @Published public var announceChanges = true
    
    public init() {
        // Check system preferences
        updateFromSystemSettings()
    }
    
    public func updateFromSystemSettings() {
        #if os(macOS)
        highContrastMode = NSWorkspace.shared.accessibilityDisplayShouldIncreaseContrast
        reduceMotion = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        #endif
    }
}

public extension View {
    func accessibleTimeline() -> some View {
        self
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Video timeline")
            .accessibilityHint("Use keyboard shortcuts: V for select, B for blade, Space for play/pause")
    }
    
    func accessibleClip(_ clip: SimpleTimelineClip) -> some View {
        self
            .accessibilityLabel("Clip: \(clip.name)")
            .accessibilityValue("Duration: \(String(format: "%.1f", clip.duration))s at \(String(format: "%.1f", clip.startTime))s")
            .accessibilityAddTraits(.isButton)
            .accessibilityHint("Double-click to select, drag to move")
    }
    
    func accessibleButton(_ label: String, hint: String? = nil) -> some View {
        self
            .accessibilityLabel(label)
            .accessibilityHint(hint ?? "")
            .accessibilityAddTraits(.isButton)
    }
}