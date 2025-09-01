import Foundation
import SwiftUI
import AutoResolveCore
import AIDirector

public class AppState: ObservableObject {
    static var shared: AppState?
    
    // Project state
    @Published var currentProject: Project?
    @Published var timeline: Timeline?
    @Published var selectedClips: Set<UUID> = []
    
    // UI state
    @Published var isPlaying = false
    @Published var currentTime = Tick.zero
    @Published var zoomLevel: Double = 1.0
    @Published var showInspector = false
    @Published var showAISettings = false
    
    // AI state
    @Published var aiEnabled = true
    @Published var aiAutoApply = false
    @Published var aiConfidenceThreshold = 0.7
    @Published var aiLearningEnabled = true
    @Published var showAISuggestions = true
    @Published var aiAnalyzing = false
    @Published var aiSuggestions: [EditSuggestion] = []
    @Published var aiStats: LearningStatistics?
    
    // Director
    private let director = Director()
    private let commandProcessor = CommandProcessor()
    
    // Playback
    private var playbackTimer: Timer?
    
    // Computed properties
    var hasSelection: Bool {
        !selectedClips.isEmpty
    }
    
    var canBlade: Bool {
        timeline != nil && currentTime != .zero
    }
    
    public init() {
        AppState.shared = self
    }
    
    // MARK: - Initialization
    
    func initialize() {
        // Load last project or create new
        if let lastProject = loadLastProject() {
            currentProject = lastProject
            timeline = lastProject.timeline
        }
        
        // Start AI monitoring
        if aiEnabled {
            startAIMonitoring()
        }
    }
    
    // MARK: - Project Management
    
    func createNewProject() {
        let project = Project(name: "Untitled Project")
        currentProject = project
        timeline = project.timeline
        selectedClips.removeAll()
    }
    
    func openProject() {
        // Show open panel
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.project]
        panel.canChooseDirectories = false
        
        if panel.runModal() == .OK, let url = panel.url {
            // Load project
            if let project = loadProject(from: url) {
                currentProject = project
                timeline = project.timeline
                selectedClips.removeAll()
            }
        }
    }
    
    func saveProject() {
        guard let project = currentProject else { return }
        
        // Save to disk
        saveProject(project)
    }
    
    // MARK: - Media Import
    
    func importMedia() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .audio, .image]
        panel.allowsMultipleSelection = true
        
        if panel.runModal() == .OK {
            for url in panel.urls {
                importMedia(from: url)
            }
        }
    }
    
    private func importMedia(from url: URL) {
        guard let project = currentProject else { return }
        
        // Create media item
        let media = MediaItem(
            url: url,
            duration: getMediaDuration(url)
        )
        
        // Add to project
        project.mediaPool.add(media)
    }
    
    // MARK: - Timeline Operations
    
    func blade() {
        guard let timeline = timeline else { return }
        
        let command = Command.blade(at: currentTime, trackIndex: 0)
        execute(command)
    }
    
    func deleteSelection() {
        guard !selectedClips.isEmpty, let timeline = timeline else { return }
        
        for clipId in selectedClips {
            let command = Command.delete(clipId: clipId)
            execute(command)
        }
        
        selectedClips.removeAll()
    }
    
    func trim(clip: UUID, to newBounds: TickRange) {
        let command = Command.trim(clipId: clip, newBounds: newBounds)
        execute(command)
    }
    
    private func execute(_ command: Command) {
        guard let project = currentProject else { return }
        
        do {
            try commandProcessor.execute(command, on: project)
            
            // Update timeline
            timeline = project.timeline
            
            // Notify AI of changes
            if aiEnabled {
                director.recordUserEdit(command)
            }
        } catch {
            print("Command failed: \(error)")
        }
    }
    
    // MARK: - Playback
    
    func playPause() {
        isPlaying.toggle()
        
        if isPlaying {
            startPlayback()
        } else {
            stopPlayback()
        }
    }
    
    func stepForward() {
        currentTime = currentTime + Tick(value: 1)
    }
    
    func stepBackward() {
        currentTime = max(.zero, currentTime - Tick(value: 1))
    }
    
    private func startPlayback() {
        playbackTimer = Timer.scheduledTimer(withTimeInterval: 1.0/30.0, repeats: true) { _ in
            self.currentTime = self.currentTime + Tick(value: 33) // ~30fps
            
            // Check for end of timeline
            if let timeline = self.timeline {
                let duration = timeline.duration
                if self.currentTime >= duration {
                    self.stopPlayback()
                    self.currentTime = .zero
                }
            }
        }
    }
    
    private func stopPlayback() {
        playbackTimer?.invalidate()
        playbackTimer = nil
        isPlaying = false
    }
    
    // MARK: - Zoom
    
    func zoomIn() {
        zoomLevel = min(10.0, zoomLevel * 1.2)
    }
    
    func zoomOut() {
        zoomLevel = max(0.1, zoomLevel / 1.2)
    }
    
    func zoomToFit() {
        zoomLevel = 1.0
    }
    
    // MARK: - AI Integration
    
    func runAIAnalysis() async {
        guard let timeline = timeline else { return }
        
        aiAnalyzing = true
        defer { aiAnalyzing = false }
        
        // Run analysis
        let suggestions = await director.analyze(timeline)
        
        // Update UI
        await MainActor.run {
            self.aiSuggestions = suggestions
            self.aiStats = self.director.getStatistics()
        }
    }
    
    func applySuggestion(_ suggestion: EditSuggestion) {
        // Convert suggestion to command
        let command = suggestion.toCommand()
        execute(command)
        
        // Record acceptance
        director.recordAcceptance(suggestion)
    }
    
    func rejectSuggestion(_ suggestion: EditSuggestion) {
        // Remove from list
        aiSuggestions.removeAll { $0.id == suggestion.id }
        
        // Record rejection
        director.recordRejection(suggestion)
    }
    
    private func startAIMonitoring() {
        // Monitor timeline changes
        Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            if self.aiEnabled && !self.aiAnalyzing {
                Task {
                    await self.runAIAnalysis()
                }
            }
        }
    }
    
    // MARK: - State Persistence
    
    func saveState() {
        // Save current state to UserDefaults
        UserDefaults.standard.set(zoomLevel, forKey: "zoomLevel")
        UserDefaults.standard.set(aiEnabled, forKey: "aiEnabled")
        UserDefaults.standard.set(aiAutoApply, forKey: "aiAutoApply")
        UserDefaults.standard.set(aiConfidenceThreshold, forKey: "aiConfidenceThreshold")
        
        if let project = currentProject {
            UserDefaults.standard.set(project.url?.path, forKey: "lastProject")
        }
    }
    
    private func loadLastProject() -> Project? {
        guard let path = UserDefaults.standard.string(forKey: "lastProject"),
              let url = URL(string: path) else {
            return nil
        }
        
        return loadProject(from: url)
    }
    
    private func loadProject(from url: URL) -> Project? {
        // Load project from disk
        // Simplified for now
        return nil
    }
    
    private func saveProject(_ project: Project) {
        // Save project to disk
        // Simplified for now
    }
    
    private func getMediaDuration(_ url: URL) -> Tick {
        // Get duration using AVFoundation
        // Simplified for now
        return Tick.from(seconds: 10.0)
    }
}

// MARK: - Supporting Types

extension EditSuggestion {
    func toCommand() -> Command {
        switch type {
        case .cut:
            return .blade(at: tick, trackIndex: 0)
        case .trim(let bounds):
            return .trim(clipId: UUID(), newBounds: bounds) // Need clip ID
        case .delete:
            return .delete(clipId: UUID()) // Need clip ID
        case .transition:
            return .blade(at: tick, trackIndex: 0) // Placeholder
        }
    }
}

// UTType extension for project files
import UniformTypeIdentifiers

extension UTType {
    static let project = UTType(exportedAs: "com.autoresolve.project")
}