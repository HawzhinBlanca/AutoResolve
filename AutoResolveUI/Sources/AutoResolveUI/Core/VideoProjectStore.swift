// AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
// Professional Video Project Store & State Management

import Foundation
import SwiftUI
import Combine
import AVFoundation

// MARK: - Core Application State
@MainActor
class VideoProjectStore: ObservableObject {
    @Published var currentProject: VideoProject?
    @Published var recentProjects: [URL] = []
    @Published var isProjectModified: Bool = false
    @Published var autosaveTimer: Timer?
    
    // Global application state
    @Published var selectedWorkspace: Workspace = .default
    @Published var showInspector: Bool = true
    @Published var showMediaPool: Bool = true
    @Published var showViewer: Bool = true
    
    // Error handling
    @Published var lastError: String?
    @Published var showErrorAlert: Bool = false
    
    private var cancellables = Set<AnyCancellable>()
    private let projectManager = ProjectManager.shared
    
    init() {
        setupAutosave()
        loadRecentProjects()
        
        // Watch for project modifications
        $currentProject
            .dropFirst()
            .sink { [weak self] _ in
                self?.markProjectAsModified()
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Project Management
    func createNewProject(name: String = "Untitled Project") {
        let newProject = VideoProject(name: name)
        currentProject = newProject
        isProjectModified = true
    }
    
    func openProject(from url: URL) {
        do {
            let project = try projectManager.loadProject(from: url)
            currentProject = project
            addToRecentProjects(url)
            isProjectModified = false
        } catch {
            showError("Failed to open project: \(error.localizedDescription)")
        }
    }
    
    func saveProject() {
        guard let project = currentProject else { return }
        
        do {
            let url = try projectManager.saveProject(project)
            addToRecentProjects(url)
            isProjectModified = false
        } catch {
            showError("Failed to save project: \(error.localizedDescription)")
        }
    }
    
    func saveProjectAs(to url: URL) {
        guard var project = currentProject else { return }
        
        do {
            project.name = url.deletingPathExtension().lastPathComponent
            try projectManager.saveProject(project, to: url)
            currentProject = project
            addToRecentProjects(url)
            isProjectModified = false
        } catch {
            showError("Failed to save project: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Recent Projects
    private func loadRecentProjects() {
        recentProjects = UserDefaults.standard.stringArray(forKey: "RecentProjects")?
            .compactMap { URL(string: $0) } ?? []
    }
    
    private func addToRecentProjects(_ url: URL) {
        recentProjects.removeAll { $0 == url }
        recentProjects.insert(url, at: 0)
        if recentProjects.count > 10 {
            recentProjects = Array(recentProjects.prefix(10))
        }
        
        UserDefaults.standard.set(
            recentProjects.map { $0.absoluteString },
            forKey: "RecentProjects"
        )
    }
    
    // MARK: - Autosave
    private func setupAutosave() {
        autosaveTimer = Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.autosave()
            }
        }
    }
    
    private func autosave() {
        guard let project = currentProject,
              project.settings.autoSave,
              isProjectModified else { return }
        
        do {
            try projectManager.autosaveProject(project)
        } catch {
            print("Autosave failed: \(error)")
        }
    }
    
    private func markProjectAsModified() {
        isProjectModified = true
        currentProject?.updateModifiedDate()
    }
    
    // MARK: - Error Handling
    private func showError(_ message: String) {
        lastError = message
        showErrorAlert = true
    }
    
    func cleanup() {
        autosaveTimer?.invalidate()
    }
}

// MARK: - Timeline State Management
@MainActor
class TimelineViewModel: ObservableObject {
    @Published var playhead: TimeInterval = 0
    @Published var zoomLevel: CGFloat = 1.0
    @Published var scrollOffset: CGFloat = 0
    @Published var selectedClips: Set<UUID> = []
    @Published var selectedTracks: Set<UUID> = []
    
    // Playback state
    @Published var isPlaying: Bool = false
    @Published var isLooping: Bool = false
    @Published var playbackRate: Float = 1.0
    @Published var inPoint: TimeInterval?
    @Published var outPoint: TimeInterval?
    
    // Timeline navigation
    @Published var timelineWidth: CGFloat = 1000
    @Published var trackHeight: CGFloat = 80
    @Published var showAudioWaveforms: Bool = true
    @Published var showVideoThumbnails: Bool = true
    @Published var snapToFrames: Bool = true
    @Published var magneticTimeline: Bool = true
    
    // Editing state
    @Published var currentTool: TimelineTool = .arrow
    @Published var isDragging: Bool = false
    @Published var dragOffset: CGFloat = 0
    @Published var isResizing: Bool = false
    
    private var cancellables = Set<AnyCancellable>()
    
    var project: VideoProject? {
        didSet {
            updateTimelineFromProject()
        }
    }
    
    // MARK: - Timeline Navigation
    func setPlayhead(to time: TimeInterval) {
        playhead = max(0, min(time, project?.timeline.duration ?? 0))
    }
    
    func movePlayhead(by delta: TimeInterval) {
        setPlayhead(to: playhead + delta)
    }
    
    func goToStart() {
        setPlayhead(to: inPoint ?? 0)
    }
    
    func goToEnd() {
        setPlayhead(to: outPoint ?? project?.timeline.duration ?? 0)
    }
    
    func previousFrame() {
        guard let frameRate = project?.timeline.frameRate else { return }
        movePlayhead(by: -1.0 / frameRate)
    }
    
    func nextFrame() {
        guard let frameRate = project?.timeline.frameRate else { return }
        movePlayhead(by: 1.0 / frameRate)
    }
    
    // MARK: - Zoom and Scroll
    func zoomIn() {
        zoomLevel = min(zoomLevel * 1.5, 20.0)
    }
    
    func zoomOut() {
        zoomLevel = max(zoomLevel / 1.5, 0.1)
    }
    
    func zoomToFit() {
        guard let duration = project?.timeline.duration, duration > 0 else { return }
        zoomLevel = timelineWidth / CGFloat(duration * 10) // 10 pixels per second base
    }
    
    func centerPlayhead() {
        let playheadPosition = CGFloat(playhead) * zoomLevel
        scrollOffset = playheadPosition - timelineWidth / 2
    }
    
    // MARK: - Selection Management
    func selectClip(_ clipId: UUID) {
        selectedClips = [clipId]
    }
    
    func addClipToSelection(_ clipId: UUID) {
        selectedClips.insert(clipId)
    }
    
    func removeClipFromSelection(_ clipId: UUID) {
        selectedClips.remove(clipId)
    }
    
    func selectAll() {
        // Select all clips in current timeline
        selectedClips.removeAll()
        project?.timeline.videoTracks.forEach { track in
            track.clips.forEach { clip in
                selectedClips.insert(clip.id)
            }
        }
        project?.timeline.audioTracks.forEach { track in
            track.clips.forEach { clip in
                selectedClips.insert(clip.id)
            }
        }
    }
    
    func deselectAll() {
        selectedClips.removeAll()
        selectedTracks.removeAll()
    }
    
    // MARK: - Timeline Tools
    func setTool(_ tool: TimelineTool) {
        currentTool = tool
        if tool != .arrow {
            deselectAll()
        }
    }
    
    // MARK: - Private Methods
    private func updateTimelineFromProject() {
        guard let project = project else { return }
        
        if playhead > project.timeline.duration {
            playhead = project.timeline.duration
        }
        
        // Clear selection if clips no longer exist
        let allClipIds = Set(
            project.timeline.videoTracks.flatMap { $0.clips.map { $0.id } } +
            project.timeline.audioTracks.flatMap { $0.clips.map { $0.id } }
        )
        selectedClips = selectedClips.intersection(allClipIds)
    }
}

// MARK: - Video Player State
@MainActor
class VideoPlayerViewModel: ObservableObject {
    @Published var player: AVPlayer?
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var isPlaying: Bool = false
    @Published var isBuffering: Bool = false
    @Published var volume: Float = 1.0
    @Published var isMuted: Bool = false
    
    // Video properties
    @Published var videoSize: CGSize = .zero
    @Published var aspectRatio: CGFloat = 16.0/9.0
    @Published var colorSpace: String = "Rec. 709"
    
    private var timeObserver: Any?
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        setupPlayer()
    }
    
    private func setupPlayer() {
        player = AVPlayer()
        
        // Observe time
        let interval = CMTime(seconds: 1.0/30.0, preferredTimescale: CMTimeScale(NSEC_PER_SEC))
        timeObserver = player?.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            Task { @MainActor in
                self?.currentTime = time.seconds
            }
        }
        
        // Observe player state
        player?.publisher(for: \.timeControlStatus)
            .sink { [weak self] status in
                self?.isPlaying = status == .playing
                self?.isBuffering = status == .waitingToPlayAtSpecifiedRate
            }
            .store(in: &cancellables)
    }
    
    func loadVideo(from url: URL) {
        let asset = AVAsset(url: url)
        let playerItem = AVPlayerItem(asset: asset)
        
        player?.replaceCurrentItem(with: playerItem)
        
        // Get video properties
        Task {
            let duration = try await asset.load(.duration)
            let tracks = try await asset.load(.tracks)
            
            await MainActor.run {
                self.duration = duration.seconds
                
                if let videoTrack = tracks.first(where: { $0.mediaType == .video }) {
                    Task {
                        let size = try await videoTrack.load(.naturalSize)
                        let transform = try await videoTrack.load(.preferredTransform)
                        
                        await MainActor.run {
                            self.videoSize = size.applying(transform)
                            self.aspectRatio = size.width / size.height
                        }
                    }
                }
            }
        }
    }
    
    func play() {
        player?.play()
    }
    
    func pause() {
        player?.pause()
    }
    
    func togglePlayPause() {
        if isPlaying {
            pause()
        } else {
            play()
        }
    }
    
    func seek(to time: TimeInterval) {
        let cmTime = CMTime(seconds: time, preferredTimescale: CMTimeScale(NSEC_PER_SEC))
        player?.seek(to: cmTime)
    }
    
    func setVolume(_ volume: Float) {
        self.volume = volume
        player?.volume = volume
    }
    
    func toggleMute() {
        isMuted.toggle()
        player?.isMuted = isMuted
    }
    
    func cleanup() {
        if let observer = timeObserver {
            player?.removeTimeObserver(observer)
        }
    }
}

// MARK: - Supporting Types
enum TimelineTool: String, CaseIterable {
    case arrow = "Arrow"
    case blade = "Blade"
    case zoom = "Zoom"
    case hand = "Hand"
    
    var systemImage: String {
        switch self {
        case .arrow: return "arrow.up.left"
        case .blade: return "scissors"
        case .zoom: return "magnifyingglass"
        case .hand: return "hand.raised"
        }
    }
}

// MARK: - Project Manager
class ProjectManager {
    static let shared = ProjectManager()
    
    private let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    private let autosaveDirectory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("AutoResolve_Autosave")
    
    private init() {
        createAutosaveDirectory()
    }
    
    private func createAutosaveDirectory() {
        try? FileManager.default.createDirectory(at: autosaveDirectory, withIntermediateDirectories: true)
    }
    
    func loadProject(from url: URL) throws -> VideoProject {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(VideoProject.self, from: data)
    }
    
    func saveProject(_ project: VideoProject) throws -> URL {
        let url = documentsURL.appendingPathComponent("\(project.name).autoresolve")
        try saveProject(project, to: url)
        return url
    }
    
    func saveProject(_ project: VideoProject, to url: URL) throws {
        let data = try JSONEncoder().encode(project)
        try data.write(to: url)
    }
    
    func autosaveProject(_ project: VideoProject) throws {
        let url = autosaveDirectory.appendingPathComponent("\(project.id.uuidString).autoresolve")
        try saveProject(project, to: url)
    }
    
    func getAutosavedProjects() -> [VideoProject] {
        guard let files = try? FileManager.default.contentsOfDirectory(at: autosaveDirectory, includingPropertiesForKeys: nil) else {
            return []
        }
        
        return files.compactMap { url in
            try? loadProject(from: url)
        }
    }
}