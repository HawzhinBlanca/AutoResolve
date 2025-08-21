// AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
// Comprehensive Keyboard Shortcuts and Accessibility System

import SwiftUI
import AppKit
import Combine

// MARK: - Keyboard Shortcut Manager
@MainActor
class KeyboardShortcutManager: ObservableObject {
    static let shared = KeyboardShortcutManager()
    
    @Published var shortcuts: [String: KeyboardShortcut] = [:]
    @Published var customShortcuts: [String: KeyboardShortcut] = [:]
    @Published var isRecording: Bool = false
    @Published var recordingForAction: String?
    
    private var cancellables = Set<AnyCancellable>()
    private let userDefaults = UserDefaults.standard
    
    private init() {
        setupDefaultShortcuts()
        loadCustomShortcuts()
    }
    
    private func setupDefaultShortcuts() {
        shortcuts = [
            // File Operations
            "newProject": KeyboardShortcut(.init("n"), modifiers: .command),
            "openProject": KeyboardShortcut(.init("o"), modifiers: .command),
            "saveProject": KeyboardShortcut(.init("s"), modifiers: .command),
            "saveAsProject": KeyboardShortcut(.init("s"), modifiers: [.command, .shift]),
            "importMedia": KeyboardShortcut(.init("i"), modifiers: .command),
            "exportProject": KeyboardShortcut(.init("e"), modifiers: .command),
            "exportShorts": KeyboardShortcut(.init("e"), modifiers: [.command, .shift]),
            
            // Edit Operations
            "undo": KeyboardShortcut(.init("z"), modifiers: .command),
            "redo": KeyboardShortcut(.init("z"), modifiers: [.command, .shift]),
            "cut": KeyboardShortcut(.init("x"), modifiers: .command),
            "copy": KeyboardShortcut(.init("c"), modifiers: .command),
            "paste": KeyboardShortcut(.init("v"), modifiers: .command),
            "duplicate": KeyboardShortcut(.init("d"), modifiers: .command),
            "delete": KeyboardShortcut(.delete),
            "selectAll": KeyboardShortcut(.init("a"), modifiers: .command),
            "deselectAll": KeyboardShortcut(.init("d"), modifiers: [.command, .shift]),
            
            // Timeline Navigation
            "playPause": KeyboardShortcut(.space),
            "goToStart": KeyboardShortcut(.home),
            "goToEnd": KeyboardShortcut(.end),
            "previousFrame": KeyboardShortcut(.leftArrow),
            "nextFrame": KeyboardShortcut(.rightArrow),
            "previousSecond": KeyboardShortcut(.leftArrow, modifiers: .shift),
            "nextSecond": KeyboardShortcut(.rightArrow, modifiers: .shift),
            "jumpBackward": KeyboardShortcut(.leftArrow, modifiers: .option),
            "jumpForward": KeyboardShortcut(.rightArrow, modifiers: .option),
            
            // Playback Controls
            "playFromStart": KeyboardShortcut(.init("k"), modifiers: [.command, .option]),
            "playAround": KeyboardShortcut(.space, modifiers: .shift),
            "loopPlayback": KeyboardShortcut(.init("l"), modifiers: .command),
            "slower": KeyboardShortcut(.init("j")),
            "faster": KeyboardShortcut(.init("l")),
            
            // Timeline Editing
            "blade": KeyboardShortcut(.init("b")),
            "bladeAll": KeyboardShortcut(.init("b"), modifiers: .shift),
            "joinClips": KeyboardShortcut(.init("j")),
            "setInPoint": KeyboardShortcut(.init("i")),
            "setOutPoint": KeyboardShortcut(.init("o")),
            "clearInOut": KeyboardShortcut(.init("x"), modifiers: [.command, .option]),
            "selectClip": KeyboardShortcut(.init("x")),
            
            // Markers
            "addMarker": KeyboardShortcut(.init("m")),
            "addChapter": KeyboardShortcut(.init("m"), modifiers: [.command, .shift]),
            "addTodo": KeyboardShortcut(.init("m"), modifiers: [.command, .option]),
            "nextMarker": KeyboardShortcut(.init("'")),
            "previousMarker": KeyboardShortcut(.init(";")),
            "deleteMarker": KeyboardShortcut(.init("m"), modifiers: [.command, .control]),
            
            // Timeline Tools
            "arrowTool": KeyboardShortcut(.init("a")),
            "bladeTool": KeyboardShortcut(.init("b")),
            "zoomTool": KeyboardShortcut(.init("z")),
            "handTool": KeyboardShortcut(.init("h")),
            
            // Timeline Navigation
            "zoomIn": KeyboardShortcut(.init("="), modifiers: .command),
            "zoomOut": KeyboardShortcut(.init("-"), modifiers: .command),
            "zoomToFit": KeyboardShortcut(.init("z")),
            "zoomToSelection": KeyboardShortcut(.init("z"), modifiers: .shift),
            "centerPlayhead": KeyboardShortcut(.init("c")),
            
            // View Controls
            "showTimeline": KeyboardShortcut(.init("1"), modifiers: [.command, .shift]),
            "showViewer": KeyboardShortcut(.init("2"), modifiers: [.command, .shift]),
            "showInspector": KeyboardShortcut(.init("3"), modifiers: [.command, .shift]),
            "showMediaPool": KeyboardShortcut(.init("4"), modifiers: [.command, .shift]),
            "showThumbnails": KeyboardShortcut(.init("t"), modifiers: [.command, .shift]),
            "showWaveforms": KeyboardShortcut(.init("w"), modifiers: [.command, .shift]),
            
            // Audio Controls
            "audioFadeIn": KeyboardShortcut(.init("="), modifiers: [.command, .option]),
            "audioFadeOut": KeyboardShortcut(.init("-"), modifiers: [.command, .option]),
            "detachAudio": KeyboardShortcut(.init("s"), modifiers: [.command, .shift]),
            "expandAudio": KeyboardShortcut(.init("a"), modifiers: [.command, .control]),
            
            // Video Controls
            "transform": KeyboardShortcut(.init("t")),
            "crop": KeyboardShortcut(.init("c"), modifiers: [.command, .shift]),
            "resetTransform": KeyboardShortcut(.init("t"), modifiers: [.command, .option]),
            
            // AI Director
            "analyzeProject": KeyboardShortcut(.init("a"), modifiers: [.command, .control]),
            "generateCuts": KeyboardShortcut(.init("g"), modifiers: [.command, .control]),
            "createHighlights": KeyboardShortcut(.init("h"), modifiers: [.command, .control]),
            "showTensionCurve": KeyboardShortcut(.init("t"), modifiers: [.command, .control]),
            "showStoryBeats": KeyboardShortcut(.init("s"), modifiers: [.command, .control]),
            
            // Window Management
            "newWindow": KeyboardShortcut(.init("n"), modifiers: [.command, .shift]),
            "closeWindow": KeyboardShortcut(.init("w"), modifiers: .command),
            "minimizeWindow": KeyboardShortcut(.init("m"), modifiers: .command),
            "fullScreen": KeyboardShortcut(.init("f"), modifiers: [.command, .control]),
            
            // Workspace
            "workspace1": KeyboardShortcut(.init("1"), modifiers: [.command, .option]),
            "workspace2": KeyboardShortcut(.init("2"), modifiers: [.command, .option]),
            "workspace3": KeyboardShortcut(.init("3"), modifiers: [.command, .option]),
            "workspace4": KeyboardShortcut(.init("4"), modifiers: [.command, .option]),
            
            // Timeline Selection
            "selectAllForward": KeyboardShortcut(.init("a"), modifiers: [.command, .shift]),
            "selectAllBackward": KeyboardShortcut(.init("a"), modifiers: [.command, .option]),
            "selectToPlayhead": KeyboardShortcut(.init("\\"), modifiers: .command),
            
            // Retiming
            "slowMotion": KeyboardShortcut(.init("r"), modifiers: [.command, .option]),
            "fastMotion": KeyboardShortcut(.init("f"), modifiers: [.command, .option]),
            "normalSpeed": KeyboardShortcut(.init("n"), modifiers: [.command, .option]),
            "reverseClip": KeyboardShortcut(.init("r"), modifiers: [.command, .shift]),
            
            // Quick Actions
            "quickExport": KeyboardShortcut(.init("e"), modifiers: [.command, .option]),
            "renderInPlace": KeyboardShortcut(.init("r"), modifiers: .command),
            "generateProxies": KeyboardShortcut(.init("p"), modifiers: [.command, .shift]),
            "transcribeClip": KeyboardShortcut(.init("t"), modifiers: [.command, .shift]),
        ]
    }
    
    private func loadCustomShortcuts() {
        if let data = userDefaults.data(forKey: "CustomKeyboardShortcuts"),
           let custom = try? JSONDecoder().decode([String: KeyboardShortcut].self, from: data) {
            customShortcuts = custom
        }
    }
    
    func saveCustomShortcuts() {
        if let data = try? JSONEncoder().encode(customShortcuts) {
            userDefaults.set(data, forKey: "CustomKeyboardShortcuts")
        }
    }
    
    func getShortcut(for action: String) -> KeyboardShortcut? {
        return customShortcuts[action] ?? shortcuts[action]
    }
    
    func setCustomShortcut(_ shortcut: KeyboardShortcut, for action: String) {
        customShortcuts[action] = shortcut
        saveCustomShortcuts()
    }
    
    func resetToDefault(for action: String) {
        customShortcuts.removeValue(forKey: action)
        saveCustomShortcuts()
    }
    
    func resetAllToDefaults() {
        customShortcuts.removeAll()
        saveCustomShortcuts()
    }
    
    func startRecording(for action: String) {
        isRecording = true
        recordingForAction = action
    }
    
    func stopRecording() {
        isRecording = false
        recordingForAction = nil
    }
    
    func recordShortcut(_ shortcut: KeyboardShortcut) {
        guard let action = recordingForAction else { return }
        setCustomShortcut(shortcut, for: action)
        stopRecording()
    }
}

// MARK: - Accessibility Manager
@MainActor
class AccessibilityManager: ObservableObject {
    static let shared = AccessibilityManager()
    
    @Published var isVoiceOverEnabled: Bool = false
    @Published var isSwitchControlEnabled: Bool = false
    @Published var isReduceMotionEnabled: Bool = false
    @Published var isIncreaseContrastEnabled: Bool = false
    @Published var preferredContentSizeCategory: ContentSizeCategory = .medium
    
    private var observers: [NSObjectProtocol] = []
    
    private init() {
        setupAccessibilityObservers()
        updateAccessibilityStatus()
    }
    
    private func setupAccessibilityObservers() {
        let center = NotificationCenter.default
        
        observers.append(
            center.addObserver(forName: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in
                    self?.updateAccessibilityStatus()
                }
            }
        )
        
        observers.append(
            // NSApplication.didChangeAccessibilityTrustedNotification not available in current SDK
            center.addObserver(forName: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in
                    self?.updateAccessibilityStatus()
                }
            }
        )
    }
    
    private func updateAccessibilityStatus() {
        isVoiceOverEnabled = NSWorkspace.shared.isVoiceOverEnabled
        isSwitchControlEnabled = NSWorkspace.shared.isSwitchControlEnabled
        isReduceMotionEnabled = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        isIncreaseContrastEnabled = NSWorkspace.shared.accessibilityDisplayShouldIncreaseContrast
    }
    
    func announceToVoiceOver(_ message: String) {
        guard isVoiceOverEnabled else { return }
        
        // Post accessibility announcement using proper API
        NSAccessibility.post(element: NSApp.mainWindow ?? NSApp, 
                           notification: .announcementRequested,
                           userInfo: [.announcement: message])
    }
    
    func createAccessibleDescription(for view: String, action: String? = nil) -> String {
        var description = view
        if let action = action {
            description += ", \(action)"
        }
        return description
    }
    
    deinit {
        observers.forEach { NotificationCenter.default.removeObserver($0) }
    }
}

// MARK: - Keyboard Navigation Helper
struct KeyboardNavigationHelper {
    static func configureForAccessibility<T: View>(_ view: T) -> some View {
        view
            .focusable()
            .accessibilityAddTraits(.isButton)
            .onKeyPress(.tab) {
                // Handle tab navigation
                return .handled
            }
            .onKeyPress(.escape) {
                // Handle escape key
                return .handled
            }
    }
    
    static func makeAccessible<T: View>(_ view: T, 
                                       label: String,
                                       hint: String? = nil,
                                       value: String? = nil) -> some View {
        view
            .accessibilityLabel(label)
            .accessibilityHint(hint ?? "")
            .accessibilityValue(value ?? "")
    }
}

// MARK: - Keyboard Shortcut Settings View
struct KeyboardShortcutSettingsView: View {
    @StateObject private var shortcutManager = KeyboardShortcutManager.shared
    @State private var searchText = ""
    @State private var selectedCategory: ShortcutCategory = .all
    
    var filteredShortcuts: [(String, KeyboardShortcut)] {
        let allShortcuts = shortcutManager.shortcuts.merging(shortcutManager.customShortcuts) { _, custom in custom }
        
        let filtered = allShortcuts.filter { key, _ in
            if searchText.isEmpty {
                return true
            }
            return key.lowercased().contains(searchText.lowercased()) ||
                   getActionDisplayName(key).lowercased().contains(searchText.lowercased())
        }
        
        if selectedCategory == .all {
            return Array(filtered).sorted { $0.0 < $1.0 }
        } else {
            return Array(filtered).filter { key, _ in
                getShortcutCategory(key) == selectedCategory
            }.sorted { $0.0 < $1.0 }
        }
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                Text("Keyboard Shortcuts")
                    .font(.title)
                    .fontWeight(.semibold)
                
                Spacer()
                
                Button("Reset All") {
                    shortcutManager.resetAllToDefaults()
                }
                .buttonStyle(.bordered)
            }
            
            // Search and Filter
            HStack {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    TextField("Search shortcuts...", text: $searchText)
                        .textFieldStyle(.plain)
                }
                .padding(8)
                .background(Color(NSColor.controlBackgroundColor))
                .cornerRadius(6)
                
                Picker("Category", selection: $selectedCategory) {
                    ForEach(ShortcutCategory.allCases, id: \.self) { category in
                        Text(category.displayName).tag(category)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 150)
            }
            
            // Shortcuts List
            ScrollView {
                LazyVStack(spacing: 1) {
                    ForEach(filteredShortcuts, id: \.0) { key, shortcut in
                        ShortcutRow(
                            action: key,
                            displayName: getActionDisplayName(key),
                            shortcut: shortcut,
                            manager: shortcutManager
                        )
                    }
                }
            }
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(8)
        }
        .padding()
        .frame(minWidth: 600, minHeight: 500)
    }
    
    private func getActionDisplayName(_ action: String) -> String {
        // Convert camelCase to readable format
        let result = action.replacingOccurrences(of: "([A-Z])", with: " $1", options: .regularExpression)
        return result.capitalized.trimmingCharacters(in: .whitespaces)
    }
    
    private func getShortcutCategory(_ action: String) -> ShortcutCategory {
        if action.hasPrefix("file") || action.contains("Project") || action.contains("Import") || action.contains("Export") {
            return .file
        } else if action.contains("undo") || action.contains("redo") || action.contains("cut") || action.contains("copy") || action.contains("paste") {
            return .edit
        } else if action.contains("play") || action.contains("Playback") || action.contains("Frame") || action.contains("Speed") {
            return .playback
        } else if action.contains("zoom") || action.contains("Timeline") || action.contains("Tool") {
            return .timeline
        } else if action.contains("show") || action.contains("Window") || action.contains("workspace") {
            return .view
        } else if action.contains("ai") || action.contains("AI") || action.contains("analyze") {
            return .ai
        } else {
            return .other
        }
    }
}

struct ShortcutRow: View {
    let action: String
    let displayName: String
    let shortcut: KeyboardShortcut
    @ObservedObject var manager: KeyboardShortcutManager
    @State private var isEditing = false
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(displayName)
                    .font(.system(size: 13, weight: .medium))
                
                Text(getShortcutCategory(action).displayName)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            HStack(spacing: 8) {
                if isEditing {
                    Text("Press keys...")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.accentColor.opacity(0.1))
                        .cornerRadius(4)
                } else {
                    Text(shortcut.displayString)
                        .font(.system(size: 11, design: .monospaced))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color(NSColor.controlColor))
                        .cornerRadius(4)
                }
                
                Button("Edit") {
                    isEditing.toggle()
                    if isEditing {
                        manager.startRecording(for: action)
                    } else {
                        manager.stopRecording()
                    }
                }
                .buttonStyle(.borderless)
                .font(.system(size: 11))
                
                Button("Reset") {
                    manager.resetToDefault(for: action)
                }
                .buttonStyle(.borderless)
                .font(.system(size: 11))
                .disabled(manager.customShortcuts[action] == nil)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(NSColor.controlBackgroundColor))
        .onReceive(manager.$isRecording) { recording in
            if !recording && isEditing {
                isEditing = false
            }
        }
    }
    
    private func getShortcutCategory(_ action: String) -> ShortcutCategory {
        // Same implementation as in parent view
        if action.hasPrefix("file") || action.contains("Project") || action.contains("Import") || action.contains("Export") {
            return .file
        } else if action.contains("undo") || action.contains("redo") || action.contains("cut") || action.contains("copy") || action.contains("paste") {
            return .edit
        } else if action.contains("play") || action.contains("Playback") || action.contains("Frame") || action.contains("Speed") {
            return .playback
        } else if action.contains("zoom") || action.contains("Timeline") || action.contains("Tool") {
            return .timeline
        } else if action.contains("show") || action.contains("Window") || action.contains("workspace") {
            return .view
        } else if action.contains("ai") || action.contains("AI") || action.contains("analyze") {
            return .ai
        } else {
            return .other
        }
    }
}

enum ShortcutCategory: String, CaseIterable {
    case all = "all"
    case file = "file"
    case edit = "edit"
    case playback = "playback"
    case timeline = "timeline"
    case view = "view"
    case ai = "ai"
    case other = "other"
    
    var displayName: String {
        switch self {
        case .all: return "All"
        case .file: return "File"
        case .edit: return "Edit"
        case .playback: return "Playback"
        case .timeline: return "Timeline"
        case .view: return "View"
        case .ai: return "AI Director"
        case .other: return "Other"
        }
    }
}

// MARK: - Keyboard Shortcut Extensions
extension KeyboardShortcut: Codable {
    enum CodingKeys: String, CodingKey {
        case key, modifiers
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let keyString = try container.decode(String.self, forKey: .key)
        let modifiersRaw = try container.decode(Int.self, forKey: .modifiers)
        
        let key = KeyEquivalent(keyString.first ?? Character(""))
        let modifiers = EventModifiers(rawValue: modifiersRaw)
        
        self.init(key, modifiers: modifiers)
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(String(key.character), forKey: .key)
        try container.encode(modifiers.rawValue, forKey: .modifiers)
    }
    
    var displayString: String {
        var result = ""
        
        if modifiers.contains(.command) { result += "⌘" }
        if modifiers.contains(.option) { result += "⌥" }
        if modifiers.contains(.control) { result += "⌃" }
        if modifiers.contains(.shift) { result += "⇧" }
        
        let keyString = String(key.character).uppercased()
        result += keyString
        
        return result
    }
}