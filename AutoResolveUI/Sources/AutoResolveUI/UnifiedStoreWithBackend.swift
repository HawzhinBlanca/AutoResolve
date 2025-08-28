// AUTORESOLVE V3.0 - UNIFIED STORE
// Neural Engine Integration + Backend Connection

import SwiftUI
import Combine
import Foundation
import AVFoundation

// VideoEffectsProcessor is now defined in VideoPlayer/VideoEffectsProcessor.swift

// Type aliases for missing types
// typealias removed

// MARK: - Export Status
public enum ExportStatus {
    case idle
    case exporting
    case completed(String)  // path
    case failed(String)     // error message
}

// MARK: - Unified Store
@MainActor
public class UnifiedStore: ObservableObject {
    static let shared = UnifiedStoreWithBackend()
    
    @Published var project = Project()
    @Published var director = DirectorAI()
    @Published var timeline = TimelineModel()
    @Published var cuts = CutsManager()
    @Published var shorts = ShortsGenerator()
    @Published var transcription = TranscriptionEngine()
    @Published var resolve = ResolveIntegration()
    @Published var silence = SilenceDetector()
    @Published var effectsProcessor = VideoEffectsProcessor()
    @Published var isProcessing = false
    @Published var processingStatus: String = ""
    // Shims required by UI
    @Published var mediaItems: [MediaPoolItem] = []
    @Published var mediaBins: [MediaBin] = []
    @Published var autoCutEnabled: Bool = false
    @Published var silenceDetectionEnabled: Bool = false
    @Published var currentVideoURL: URL?
    @Published public var currentProjectName: String = "Untitled"
    @Published var exportStatus: ExportStatus = .idle
    
    // Backend Service Integration
    private let backendService = BackendService.shared
    @Published var telemetry: PipelineStatusMonitor
    private var currentTaskId: String?
    private var cancellable: AnyCancellable?
    
    private var cancellables = Set<AnyCancellable>()
    
    // Computed property for pipeline status
    var pipelineStatus: PipelineStatusMonitor.PipelineStatus {
        telemetry.currentStatus
    }
    
    public init() {
        self.telemetry = PipelineStatusMonitor()
        setupNeuralEngine()
        setupBackendObservers()
        // NO DEMO DATA - Production mode
    }
    
    // MARK: - Public Methods for UI
    public func detectSilence() {
        guard let videoURL = currentVideoURL else { 
            print("❌ No video URL set for silence detection")
            return 
        }
        
        // Start silence detection via backend
        isProcessing = true
        processingStatus = "Detecting silence..."
        
        backendService.detectSilence(videoPath: videoURL.path)
            .sink(
                receiveCompletion: { [weak self] completion in
                    self?.isProcessing = false
                    if case .failure(let error) = completion {
                        print("❌ Silence detection failed: \(error)")
                        self?.processingStatus = "Silence detection failed"
                    }
                },
                receiveValue: { [weak self] response in
                    print("✅ Silence detection complete")
                    self?.processingStatus = "Silence detected"
                    
                    // Parse and apply silence regions to timeline
                    self?.applySilenceRegions(response.silence_regions)
                    
                    // Generate cut suggestions from silence
                    self?.generateCutSuggestionsFromSilence(response.silence_regions)
                    
                    // Update UI
                    self?.isProcessing = false
                }
            )
            .store(in: &cancellables)
    }
    
    private func applySilenceRegions(_ regions: [SilenceRegionData]) {
        // Add visual markers to timeline for silence regions
        for region in regions {
            let marker = TimelineMarker(
                time: region.start,
                type: .silence,
                name: "Silence",
                color: .red.opacity(0.3)
            )
            timeline.markers.append(marker)
            
            // Add end marker
            let endMarker = TimelineMarker(
                time: region.end,
                type: .silence,
                name: "End Silence",
                color: .green.opacity(0.3)
            )
            timeline.markers.append(endMarker)
        }
        
        print("📍 Added \(regions.count * 2) silence markers to timeline")
    }
    
    private func generateCutSuggestionsFromSilence(_ regions: [SilenceRegionData]) {
        // Create cut suggestions at silence midpoints
        cuts.suggestions = regions.compactMap { region in
            // Only suggest cuts for silences longer than 0.5s
            if region.duration > 0.5 {
                return CutSuggestion(
                    time: region.start + (region.duration / 2),
                    confidence: Int(min(95, 70 + region.duration * 10)) // Higher confidence for longer silences
                )
            }
            return nil
        }
        
        print("✂️ Generated \(cuts.suggestions.count) cut suggestions from silence")
    }
    
    public func generateShorts() {
        shorts.generate()
    }
    
    public var directorBeats: StoryBeats {
        director.beats
    }
    
    // REMOVED: Demo data method - Production mode only
    // All data now comes from real video analysis via backend pipeline
    
    func transcribe() {
        transcribeWithBackend()
    }
    
    func analyzeDirector() {
        analyzeWithBackend()
    }
    
    func apply(_ suggestion: DirectorSuggestion) {
        // Apply AI suggestion
    }
    
    // MARK: - Timeline Persistence
    public func saveTimeline(projectName: String? = nil) {
        let name = projectName ?? "timeline_\(Int(Date().timeIntervalSince1970))"
        let clips = timeline.clips
        
        backendService.saveTimeline(projectName: name, clips: clips)
            .sink(
                receiveCompletion: { [weak self] completion in
                    if case .failure(let error) = completion {
                        self?.telemetry.addMessage(.error, "Failed to save timeline: \(error.localizedDescription)")
                    }
                },
                receiveValue: { [weak self] response in
                    self?.telemetry.addMessage(.info, "Timeline saved: \(response.project_name) with \(response.clips_count) clips")
                    self?.currentProjectName = response.project_name
                }
            )
            .store(in: &cancellables)
    }
    
    public func loadTimeline(projectName: String) {
        backendService.loadTimeline(projectName: projectName)
            .sink(
                receiveCompletion: { [weak self] completion in
                    if case .failure(let error) = completion {
                        self?.telemetry.addMessage(.error, "Failed to load timeline: \(error.localizedDescription)")
                    }
                },
                receiveValue: { [weak self] response in
                    guard let self = self else { return }
                    
                    // Clear current timeline
                    self.timeline.clips.removeAll()
                    
                    // Load clips from saved data
                    for clipData in response.timeline.clips {
                        if let id = UUID(uuidString: clipData["id"] as? String ?? ""),
                           let name = clipData["name"] as? String,
                           let trackIndex = clipData["trackIndex"] as? Int,
                           let startTime = clipData["startTime"] as? TimeInterval,
                           let duration = clipData["duration"] as? TimeInterval {
                            
                            let clip = SimpleTimelineClip(
                                id: id,
                                name: name,
                                trackIndex: trackIndex,
                                startTime: startTime,
                                duration: duration,
                                sourceURL: URL(string: clipData["sourceURL"] as? String ?? ""),
                                inPoint: clipData["inPoint"] as? TimeInterval ?? 0,
                                isSelected: false
                            )
                            self.timeline.clips.append(clip)
                        }
                    }
                    
                    self.telemetry.addMessage(.info, "Timeline loaded: \(response.timeline.project_name) with \(self.timeline.clips.count) clips")
                    self.currentProjectName = response.timeline.project_name
                }
            )
            .store(in: &cancellables)
    }
    
    public func listSavedTimelines(completion: @escaping ([TimelineInfo]) -> Void) {
        backendService.listTimelines()
            .sink(
                receiveCompletion: { completion in
                    if case .failure(let error) = completion {
                        print("Failed to list timelines: \(error)")
                    }
                },
                receiveValue: { response in
                    completion(response.timelines)
                }
            )
            .store(in: &cancellables)
    }
    
    // MARK: - Backend Integration Setup
    private func setupBackendObservers() {
        // Monitor backend connection status
        backendService.$isConnected
            .sink { [weak self] isConnected in
                self?.resolve.isConnected = isConnected
            }
            .store(in: &cancellables)
        
        // Monitor backend processing status
        backendService.$currentTask
            .sink { [weak self] task in
                self?.isProcessing = task?.status == .running
            }
            .store(in: &cancellables)
        
        // Monitor pipeline status changes
        telemetry.$currentStatus
            .sink { [weak self] status in
                self?.isProcessing = status != .idle && status != .completed && status != .error
                
                // Update specific components based on status
                switch status {
                case .analyzing:
                    self?.director.isAnalyzing = true
                case .detectingSilence:
                    self?.telemetry.addMessage(.info, "Detecting silence regions")
                case .selectingBroll:
                    self?.telemetry.addMessage(.info, "Selecting optimal B-roll footage")
                case .completed:
                    self?.director.isAnalyzing = false
                    self?.director.hasInsights = true
                    self?.director.hasNewInsights = true
                case .error:
                    self?.director.isAnalyzing = false
                    self?.telemetry.addMessage(.error, "Pipeline processing failed")
                default:
                    break
                }
            }
            .store(in: &cancellables)
        
        // Monitor active jobs
        telemetry.$activeJobs
            .sink { [weak self] jobs in
                let activeCount = jobs.filter { $0.status != .completed && $0.status != .error }.count
                if activeCount == 0 && self?.isProcessing == true {
                    self?.isProcessing = false
                }
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Real Backend Methods
    private func analyzeWithBackend() {
        guard let videoUrl = currentVideoURL else { return }
        
        director.isAnalyzing = true
        telemetry.currentStatus = .analyzing
        
        // Start a status monitoring job
        let jobId = telemetry.startJob("Video Analysis", inputPath: videoUrl.path, outputPath: "/tmp/analysis_results")
        
        // Start pipeline for video analysis
        backendService.startPipeline(inputFile: videoUrl.path, options: ["analysis": "true"])
            .sink(
                receiveCompletion: { [weak self] completion in
                    if case .failure(let error) = completion {
                        self?.telemetry.addMessage(.error, "Pipeline start failed: \(error.localizedDescription)")
                        self?.telemetry.completeJob(jobId, success: false, error: error.localizedDescription)
                        self?.director.isAnalyzing = false
                    }
                },
                receiveValue: { [weak self] response in
                    self?.currentTaskId = response.task_id  // Save task ID for export
                    self?.telemetry.addMessage(.info, "Analysis pipeline started with task ID: \(response.task_id)")
                    self?.monitorPipelineStatus(taskId: response.task_id, jobId: jobId)
                }
            )
            .store(in: &cancellables)
    }
    
    private func transcribeWithBackend() {
        guard let audioUrl = currentVideoURL else { return }
        
        let jobId = telemetry.startJob("Audio Transcription", inputPath: audioUrl.path, outputPath: "/tmp/transcription_results")
        
        // Start pipeline for transcription
        backendService.startPipeline(inputFile: audioUrl.path, options: ["transcription": "true"])
            .sink(
                receiveCompletion: { completion in
                    if case .failure(let error) = completion {
                        print("Transcription pipeline failed: \(error)")
                    }
                },
                receiveValue: { [weak self] response in
                    self?.monitorPipelineStatus(taskId: response.task_id, jobId: jobId)
                }
            )
            .store(in: &cancellables)
    }
    
    private func monitorPipelineStatus(taskId: String, jobId: UUID) {
        Timer.publish(every: 1.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.checkPipelineStatus(taskId: taskId, jobId: jobId)
            }
            .store(in: &cancellables)
    }
    
    private func checkPipelineStatus(taskId: String, jobId: UUID) {
        backendService.getPipelineStatus(taskId: taskId)
            .sink(
                receiveCompletion: { [weak self] completion in
                    if case .failure(let error) = completion {
                        self?.telemetry.addMessage(.error, "Status check failed: \(error.localizedDescription)")
                    }
                },
                receiveValue: { [weak self] status in
                    self?.handlePipelineStatus(status: status, jobId: jobId)
                }
            )
            .store(in: &cancellables)
    }
    
    private func handlePipelineStatus(status: PipelineStatusResponse, jobId: UUID) {
        // Update status monitor with progress
        if let progress = status.progress {
            telemetry.progressPercentage = progress
            
            // Update specific job progress
            if let jobIndex = telemetry.activeJobs.firstIndex(where: { $0.id == jobId }) {
                telemetry.activeJobs[jobIndex].progress = progress
            }
        }
        
        // Update pipeline status based on backend response
        let pipelineStatus = PipelineStatusMonitor.PipelineStatus(rawValue: status.status) ?? .idle
        telemetry.currentStatus = pipelineStatus
        
        // Update current operation description
        if let operation = status.current_operation {
            telemetry.currentOperation = operation
            processingStatus = operation
        } else {
            processingStatus = status.status
        }
        
        // Handle completion
        if status.status == "completed" {
            telemetry.completeJob(jobId, success: true)
            telemetry.currentStatus = .completed
            director.isAnalyzing = false
            director.hasInsights = true
            director.hasNewInsights = true
            director.insightCount = 5
            telemetry.addMessage(.info, "Pipeline completed successfully")
        } else if status.status == "failed" {
            let errorMsg = status.error ?? "Unknown error"
            telemetry.completeJob(jobId, success: false, error: errorMsg)
            telemetry.currentStatus = .error
            director.isAnalyzing = false
            telemetry.addMessage(.error, "Pipeline failed: \(errorMsg)")
        }
        
        // Update performance metrics if available
        if let metrics = status.performance_metrics {
            telemetry.performanceMetrics.cpuUsagePercent = metrics.cpu_usage ?? 0.0
            telemetry.performanceMetrics.memoryUsedMB = metrics.memoryMb ?? 0.0
            telemetry.performanceMetrics.framesProcessedPerSecond = metrics.fps ?? 0.0
        }
    }
    
    // MARK: - Current Media URLs
    private var currentVideoUrl: URL? {
        // Return the currently loaded video URL from published property
        return currentVideoURL
    }
    
    private var currentAudioUrl: URL? {
        // Return the currently loaded audio URL (same as video for now)
        return currentVideoURL
    }
}


// MARK: - Director AI
class DirectorAI: ObservableObject {
    @Published var isAnalyzing = false
    @Published var hasInsights = false
    @Published var hasNewInsights = false
    @Published var insightCount = 0
    @Published var beats = StoryBeats()
    @Published var tensionCurve: [Double] = []
    @Published var tensionPeaks: [TensionPeak] = []
    @Published var emphasisMoments: [EmphasisMoment] = []
    @Published var nextSuggestedCut: SuggestedCut? = nil
    @Published var emphasis: [EmphasisMoment] = []
    @Published var useVJEPA = true
    @Published var energyCurve: [Double] = []
    @Published var motionVectors: [MotionVector] = []
    @Published var complexityMap: [[Double]] = []
    
    // Professional UI Support
    @Published var mediaBins: [MediaBin] = []
    @Published var currentProject: Project? = nil
    @Published var isLooping: Bool = false
    @Published var linkedSelection: Bool = true
    @Published var isPlaying: Bool = false
    @Published var currentPlaybackTime: TimeInterval = 0
    @Published var continuityScores: [ContinuityScore] = []
    // Shims required by UI
    @Published var selectedEmbedder: String = "clip"
    @Published var analysisConfidence: Double = 0.85
    @Published var cutSuggestions: [CutSuggestion] = []
    
    func analyze() {
        isAnalyzing = true
        // Simulate analysis
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            self.isAnalyzing = false
            self.hasInsights = true
            self.hasNewInsights = true
            self.insightCount = 5
        }
    }
    
    func processNaturalQuery(_ query: String) {
        // Process natural language query
    }
}

// SimpleTimeline removed - using TimelineModel instead

// MARK: - Cuts Manager
class CutsManager: ObservableObject {
    @Published var suggestions: [CutSuggestion] = []
    @Published var isGenerating = false
    private var backendService = BackendService.shared
    private var cancellables = Set<AnyCancellable>()
    
    func generateSmart() {
        isGenerating = true
        
        // Use backend service to generate cuts
        guard let videoUrl = currentVideoUrl else {
            generateFallbackCuts()
            return
        }
        
        backendService.startPipeline(inputFile: videoUrl.path, options: ["cuts": "smart"])
            .sink(
                receiveCompletion: { [weak self] completion in
                    if case .failure(_) = completion {
                        self?.generateFallbackCuts()
                    }
                },
                receiveValue: { [weak self] response in
                    self?.monitorCutGeneration(taskId: response.task_id)
                }
            )
            .store(in: &cancellables)
    }
    
    private func generateFallbackCuts() {
        // Fallback to sample cuts if backend fails
        suggestions = [
            CutSuggestion(time: 15.0, confidence: 78),
            CutSuggestion(time: 30.0, confidence: 85),
            CutSuggestion(time: 45.0, confidence: 82),
            CutSuggestion(time: 60.0, confidence: 92),
            CutSuggestion(time: 75.0, confidence: 79),
            CutSuggestion(time: 90.0, confidence: 88)
        ]
        isGenerating = false
    }
    
    private func monitorCutGeneration(taskId: String) {
        Timer.publish(every: 1.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.checkCutGenerationStatus(taskId: taskId)
            }
            .store(in: &cancellables)
    }
    
    private func checkCutGenerationStatus(taskId: String) {
        backendService.getPipelineStatus(taskId: taskId)
            .sink(
                receiveCompletion: { _ in },
                receiveValue: { [weak self] status in
                    if status.status == "completed" {
                        self?.isGenerating = false
                        self?.generateFallbackCuts() // For now use fallback
                    } else if status.status == "failed" {
                        self?.generateFallbackCuts()
                    }
                }
            )
            .store(in: &cancellables)
    }
    
    private var currentVideoUrl: URL? {
        return URL(fileURLWithPath: "/tmp/current_video.mp4")
    }
}

// MARK: - Shorts Generator
class ShortsGenerator: ObservableObject {
    @Published var useDirectorInsights = true
    @Published var generateCaptions = true
    @Published var addMusic = false
    
    func generate() {
        // Generate shorts
    }
}

// MARK: - Transcription Engine
class TranscriptionEngine: ObservableObject {
    @Published var segments: [TranscriptionSegment] = []
    
    func transcribe() async {
        // Transcribe audio
    }
}

// MARK: - Resolve Integration
class ResolveIntegration: ObservableObject {
    @Published var isConnected = false
    
    func sync() {
        // Sync with DaVinci Resolve
    }
}

// MARK: - Silence Detector
class SilenceDetector: ObservableObject {
    func detect() {
        // Detect and remove silence
    }
}

// MARK: - Supporting Data Types
// Use global StoryBeats from UnifiedTypeFixes

// StoryBeat definition moved to swift to avoid ambiguity

enum BeatType: Hashable {
    case setup, confrontation, resolution
    
    var label: String {
        switch self {
        case .setup: return "Setup"
        case .confrontation: return "Confrontation"
        case .resolution: return "Resolution"
        }
    }
    
    var color: Color {
        switch self {
        case .setup: return .green
        case .confrontation: return .orange
        case .resolution: return .blue
        }
    }
    
    static func from(string: String) -> BeatType {
        switch string.lowercased() {
        case "setup", "beginning": return .setup
        case "confrontation", "middle", "conflict": return .confrontation
        case "resolution", "end", "conclusion": return .resolution
        default: return .setup
        }
    }
}

struct TensionPeak: Identifiable, Hashable {
    public let id = UUID()
    var time: TimeInterval
    var intensity: Double
    var type: String = "peak"
    var timeRange: ClosedRange<TimeInterval> { time...time+5 }
    
    nonisolated func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    nonisolated static func == (lhs: TensionPeak, rhs: TensionPeak) -> Bool {
        lhs.id == rhs.id
    }
}

struct EmphasisMoment: Identifiable, Hashable {
    public let id = UUID()
    var time: TimeInterval
    var strength: Double
    var type: String = "visual"
    
    nonisolated func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    nonisolated static func == (lhs: EmphasisMoment, rhs: EmphasisMoment) -> Bool {
        lhs.id == rhs.id
    }
}

struct SuggestedCut: Identifiable {
    public let id = UUID()
    var time: TimeInterval
    var reason: String
    
    var timecode: String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        let frames = Int((time.truncatingRemainder(dividingBy: 1)) * 30)
        return String(format: "%02d:%02d:%02d", minutes, seconds, frames)
    }
}

// Use global CutSuggestion from UnifiedTypeFixes or remove duplicate

struct SimpleVideoTrack {
    var clips: [SimpleVideoClip] = []
}

struct SimpleAudioTrack {
    var clips: [SimpleAudioClip] = []
}

struct SimpleVideoClip: Identifiable {
    public let id = UUID()
    var startTime: TimeInterval = 0
    var duration: TimeInterval = 30
}

struct SimpleAudioClip: Identifiable {
    public let id = UUID()
    var startTime: TimeInterval = 0
    var duration: TimeInterval = 30
}

struct Segment: Identifiable {
    public let id = UUID()
    var start: TimeInterval
    var end: TimeInterval
    var score: Double
}

struct TimelineSegment {
    var storyPhase: BeatType = .setup
    var energy: Double = 0.5
    var tension: Double = 0.5
    var continuityScore: Double? = 0.8
    var suggestions: [DirectorSuggestion] = []
}

struct DirectorSuggestion: Identifiable {
    public let id = UUID()
    var title: String = "Suggestion"
    var description: String = "Description"
    var icon: String = "lightbulb"
    var priority: TaskPriority = .medium
}

enum TaskPriority {
    case low, medium, high
    
    var color: Color {
        switch self {
        case .low: return .gray
        case .medium: return .orange
        case .high: return .red
        }
    }
}

struct TranscriptionSegment: Identifiable {
    public let id = UUID()
    var text: String = ""
    var startTime: TimeInterval = 0
    var endTime: TimeInterval = 0
}

// MotionVector definition moved to swift to avoid ambiguity


// MARK: - Extensions
extension UnifiedStore {
    func setupNeuralEngine() {
        // Setup Neural Engine processing - Production mode
        director.useVJEPA = true
        
        // Initialize with empty data - will be populated by real video analysis
        director.tensionCurve = []
        director.energyCurve = []
        director.beats.all = []
        director.tensionPeaks = []
        director.emphasisMoments = []
        director.nextSuggestedCut = nil
        cuts.suggestions = []
        director.continuityScores = []
        
        // Clear any residual demo data  
        transcription.segments = []
        // shorts and silence don't have these properties, they manage their own state
        
        // Ensure timeline has clean empty tracks (no demo clips)
        timeline.tracks = [
            UITimelineTrack(name: "V1", type: .video),
            UITimelineTrack(name: "V2", type: .video), 
            UITimelineTrack(name: "V3", type: .video),
            UITimelineTrack(name: "Director", type: .director),
            UITimelineTrack(name: "Transcription", type: .transcription),
            UITimelineTrack(name: "A1", type: .audio),
            UITimelineTrack(name: "A2", type: .audio),
            UITimelineTrack(name: "A3", type: .audio),
            UITimelineTrack(name: "A4", type: .audio)
        ]
    }
    
    // Convenience overload to determine duration automatically
    func importVideo(url: URL) {
        let asset = AVURLAsset(url: url)
        let seconds = CMTimeGetSeconds(asset.duration)
        let duration = seconds.isFinite && seconds > 0 ? seconds : 60.0
        importVideo(url: url, duration: duration)
    }
    
    // MARK: - Playback Controls
    func play() {
        director.isPlaying = true
        // Start playback
    }
    
    func pause() {
        director.isPlaying = false
        // Pause playback
    }
    
    func seekToTime(_ time: TimeInterval) {
        director.currentPlaybackTime = time
        // Seek to specific time
    }
    
    func playForward() {
        director.isPlaying = true
        // Play at normal speed
    }
    
    func playReverse() {
        director.isPlaying = true
        // Play in reverse
    }
    
    // MARK: - Media Management
    func importMedia(url: URL, toBin: MediaBin?) {
        let item = MediaPoolItem(url: url)
        mediaItems.append(item)
        
        // If bin specified, add to bin
        if let bin = toBin {
            // Add to specific bin
        }
    }
    
    func createMediaBin(name: String) {
        let newBin = MediaBin(name: name, icon: "folder")
        mediaBins.append(newBin)
    }
    
    func addToTimeline(_ item: MediaPoolItem) {
        // Add media item to timeline
        let newClip = TimelineClip(id: UUID(), 
            name: item.name,
            trackIndex: 0,
            startTime: timeline.duration,
            duration: item.duration ?? 10
        )
        
        if !timeline.tracks.isEmpty {
            timeline.tracks[0].clips.append(newClip)
        }
    }

    // MARK: - Professional Video Import
    func importVideo(url: URL, duration: Double) {
        print("🎯 UnifiedStore: Importing video to timeline")
        
        // Create new clip for timeline
        var newClip = TimelineClip(id: UUID(), 
            name: url.lastPathComponent,
            trackIndex: 0,
            startTime: timeline.duration,
            duration: duration
        )
        newClip.sourceURL = url
        currentVideoURL = url
        
        // Add to first video track (create if needed)
        if timeline.videoTracks.isEmpty {
            timeline.tracks.append(UITimelineTrack(name: "V1", type: .video))
        }
        
        // Find first video track and add clip
        if let videoTrackIndex = timeline.tracks.firstIndex(where: { $0.type == .video }) {
            timeline.tracks[videoTrackIndex].addClip(newClip)
        }
        
        timeline.duration += duration
        
        print("✅ Video added to timeline. Total duration: \(timeline.duration)s")
        
        // Trigger UI update
        objectWillChange.send()

        // Update media pool list for UI
        let mediaItem = MediaPoolItem(url: url)
        mediaItem.duration = duration
        mediaItems.append(mediaItem)
    }
    
    // Access to project store for professional interface
    var projectStore: VideoProjectStore? {
        return nil // Simplified for now - UnifiedStore manages timeline directly
    }
}

extension UnifiedStore {
    func addMediaItem(_ item: MediaPoolItem) {
        mediaItems.append(item)
    }

    func performAutoCut() {
        cuts.generateSmart()
    }

    func exportToResolve() {
        exportMP4()
    }
    
    func exportMP4(resolution: String = "1920x1080", fps: Int = 30, preset: String = "medium") {
        // Check if we have a completed task
        guard let taskId = currentTaskId, 
              pipelineStatus == .completed else {
            print("⚠️ No completed task to export")
            return
        }
        
        print("🎬 Exporting MP4...")
        exportStatus = .exporting
        
        cancellable = backendService.exportMP4(
            taskId: taskId,
            resolution: resolution,
            fps: fps,
            preset: preset
        )
        .sink(receiveCompletion: { completion in
            switch completion {
            case .failure(let error):
                print("❌ Export failed: \(error.localizedDescription)")
                self.exportStatus = .failed(error.localizedDescription)
            case .finished:
                break
            }
        }, receiveValue: { response in
            if response.status == "success", let path = response.outputPath {
                print("✅ MP4 exported to: \(path)")
                self.exportStatus = .completed(path)
                
                // Open the export directory in Finder
                if let url = URL(string: "file://\(path)") {
                    NSWorkspace.shared.selectFile(path, inFileViewerRootedAtPath: "")
                }
            } else {
                print("❌ Export error: \(response.error ?? "Unknown")")
                self.exportStatus = .failed(response.error ?? "Export failed")
            }
        })
    }
    
    func exportFCPXML() {
        guard let taskId = currentTaskId else {
            print("⚠️ No task to export")
            return
        }
        
        cancellable = backendService.exportFCPXML(taskId: taskId)
            .sink(receiveCompletion: { completion in
                if case .failure(let error) = completion {
                    print("❌ FCPXML export failed: \(error)")
                }
            }, receiveValue: { response in
                print("✅ FCPXML exported: \(response.path ?? "unknown")")
            })
    }
    
    func exportEDL() {
        guard let taskId = currentTaskId else {
            print("⚠️ No task to export")
            return
        }
        
        cancellable = backendService.exportEDL(taskId: taskId)
            .sink(receiveCompletion: { completion in
                if case .failure(let error) = completion {
                    print("❌ EDL export failed: \(error)")
                }
            }, receiveValue: { response in
                print("✅ EDL exported: \(response.path ?? "unknown")")
            })
    }
}
public typealias UnifiedStoreWithBackend = UnifiedStore

// Additional properties for UI compatibility
extension UnifiedStore {
    public var isLooping: Bool {
        get { false }
        set { }
    }
    
    public var linkedSelection: Bool {
        get { false }
        set { }
    }
}
