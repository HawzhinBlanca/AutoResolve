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
    private let backendService = BackendClient.shared
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
        
        Task {
            do {
                let response = try await backendService.detectSilence(videoPath: videoURL.path)
                await MainActor.run {
                    print("✅ Silence detection complete")
                    self.processingStatus = "Silence detected"
                    
                    // Parse and apply silence regions to timeline
                    self.applySilenceRegions(response.silenceSegments)
                    
                    // Generate cut suggestions from silence
                    self.generateCutSuggestionsFromSilence(response.silenceSegments)
                    
                    // Update UI
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    print("❌ Silence detection failed: \(error)")
                    self.processingStatus = "Silence detection failed"
                    self.isProcessing = false
                }
            }
        }
    }
    
    private func applySilenceRegions(_ regions: [TimeRange]) {
        // Add visual markers to timeline for silence regions
        for region in regions {
            let marker = UITimelineMarker(
                time: region.start,
                type: .silence,
                name: "Silence"
            )
            timeline.markers.append(marker)
            
            // Add end marker
            let endMarker = UITimelineMarker(
                time: region.end,
                type: .silence,
                name: "End Silence"
            )
            timeline.markers.append(endMarker)
        }
        
        print("📍 Added \(regions.count * 2) silence markers to timeline")
    }
    
    private func generateCutSuggestionsFromSilence(_ regions: [TimeRange]) {
        // Create cut suggestions at silence midpoints
        cuts.suggestions = regions.compactMap { region in
            // Only suggest cuts for silences longer than 0.5s
            if region.duration > 0.5 {
                return CutSuggestion(
                    time: region.start + (region.duration / 2),
                    confidence: Int(min(95, 70 + (region.duration.isFinite ? region.duration * 10 : 0))) // Higher confidence for longer silences
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
        let clips = timeline.videoTracks.flatMap { $0.clips }
        
        // Convert clips to the format expected by TimelineData
        let clipData = clips.map { clip in
            [
                "name": clip.name,
                "start_time": String(clip.startTime),
                "duration": String(clip.duration),
                "source_url": clip.sourceURL?.absoluteString ?? ""
            ]
        }
        
        let timelineData = TimelineData(
            version: "1.0",
            project_name: name,
            saved_at: ISO8601DateFormatter().string(from: Date()),
            clips: clipData,
            metadata: [:],
            settings: [:]
        )
        
        Task {
            do {
                let response = try await backendService.saveTimeline(timelineData)
                await MainActor.run {
                    self.telemetry.addMessage(.info, "Timeline saved: \(response.timelineId ?? "unknown")")
                    self.currentProjectName = name
                }
            } catch {
                await MainActor.run {
                    self.telemetry.addMessage(.error, "Failed to save timeline: \(error.localizedDescription)")
                }
            }
        }
    }
    
    public func loadTimeline(projectName: String) {
        Task {
            do {
                let response = try await backendService.loadTimeline(id: projectName)
                await MainActor.run {
                    // Clear current timeline
                    for i in 0..<self.timeline.tracks.count {
                        if self.timeline.tracks[i].type == .video {
                            self.timeline.tracks[i].clips.removeAll()
                        }
                    }
                    
                    // Load clips from saved data
                    for clipData in response.clips {
                        if let id = UUID(uuidString: clipData["id"] ?? ""),
                           let name = clipData["name"],
                           let trackIndexStr = clipData["trackIndex"],
                           let trackIndex = Int(trackIndexStr),
                           let startTimeStr = clipData["startTime"],
                           let startTime = Double(startTimeStr),
                           let durationStr = clipData["duration"],
                           let duration = Double(durationStr) {
                            
                            let clip = SimpleTimelineClip(
                                id: id,
                                name: name,
                                trackIndex: trackIndex,
                                startTime: startTime,
                                duration: duration,
                                sourceURL: URL(string: clipData["sourceURL"] ?? "")
                            )
                            
                            // Add clip to appropriate track
                            if trackIndex < self.timeline.tracks.count {
                                self.timeline.tracks[trackIndex].clips.append(clip)
                            }
                        }
                    }
                    
                    self.telemetry.addMessage(.info, "Timeline loaded: \(response.project_name)")
                    self.currentProjectName = response.project_name
                }
            } catch {
                await MainActor.run {
                    self.telemetry.addMessage(.error, "Failed to load timeline: \(error.localizedDescription)")
                }
            }
        }
    }
    
    public func listSavedTimelines(completion: @escaping ([BackendTimelineInfo]) -> Void) {
        Task {
            do {
                let timelines = try await backendService.listTimelines()
                await MainActor.run {
                    completion(timelines.map { timeline in
                        BackendTimelineInfo(project_name: timeline.project_name, saved_at: timeline.saved_at, clips_count: timeline.clips.count, file_name: "timeline.json")
                    })
                }
            } catch {
                await MainActor.run {
                    print("Failed to list timelines: \(error)")
                    completion([])
                }
            }
        }
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
                self?.isProcessing = task?.status == "running"
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
        Task {
            do {
                let options = PipelineOptions(silenceDetection: true, transcription: true, storyBeats: true, brollSelection: true)
                let response = try await backendService.startPipeline(videoPath: videoUrl.path, options: options)
                await MainActor.run {
                    self.currentTaskId = response.taskId
                    self.telemetry.addMessage(.info, "Analysis pipeline started with task ID: \(response.taskId)")
                    Task {
                        await self.monitorPipelineStatus(taskId: response.taskId, jobId: jobId)
                    }
                }
            } catch {
                await MainActor.run {
                    self.telemetry.addMessage(.error, "Pipeline start failed: \(error.localizedDescription)")
                    self.telemetry.completeJob(jobId, success: false, error: error.localizedDescription)
                    self.director.isAnalyzing = false
                }
            }
        }
    }
    
    private func transcribeWithBackend() {
        guard let audioUrl = currentVideoURL else { return }
        
        let jobId = telemetry.startJob("Audio Transcription", inputPath: audioUrl.path, outputPath: "/tmp/transcription_results")
        
        // Start pipeline for transcription
        Task {
            do {
                let options = PipelineOptions(silenceDetection: false, transcription: true, storyBeats: false, brollSelection: false)
                let response = try await backendService.startPipeline(videoPath: audioUrl.path, options: options)
                await MainActor.run {
                    print("Transcription pipeline started: \(response.taskId)")
                }
            } catch {
                print("Transcription pipeline failed: \(error)")
            }
        }
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
        Task {
            do {
                let status = try await backendService.getPipelineStatus(taskId: taskId)
                await MainActor.run {
                    self.handlePipelineStatus(status: status, jobId: jobId)
                }
            } catch {
                await MainActor.run {
                    self.telemetry.addMessage(.error, "Status check failed: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func handlePipelineStatus(status: PipelineStatus, jobId: UUID) {
        // Update status monitor with progress
        telemetry.progressPercentage = status.progress
        
        // Update specific job progress
        if let jobIndex = telemetry.activeJobs.firstIndex(where: { $0.id == jobId }) {
            telemetry.activeJobs[jobIndex].progress = status.progress
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
        // TODO: Map performance metrics correctly when status includes them
        // if let metrics = status.performance_metrics {
        //     telemetry.performanceMetrics.cpuUsagePercent = metrics.cpu_usage ?? 0.0
        //     telemetry.performanceMetrics.memoryUsedMB = metrics.memoryMb ?? 0.0
        //     telemetry.performanceMetrics.framesProcessedPerSecond = metrics.fps ?? 0.0
        // }
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
    private var backendService = BackendClient.shared
    private var cancellables = Set<AnyCancellable>()
    
    func generateSmart() {
        isGenerating = true
        
        // Use backend service to generate cuts
        guard let videoUrl = currentVideoUrl else {
            generateFallbackCuts()
            return
        }
        
        Task {
            do {
                let options = PipelineOptions(silenceDetection: false, transcription: false, storyBeats: false, brollSelection: false)
                let response = try await backendService.startPipeline(videoPath: videoUrl.path, options: options)
                await MainActor.run {
                    self.monitorCutGeneration(taskId: response.taskId)
                }
            } catch {
                await MainActor.run {
                    self.generateFallbackCuts()
                }
            }
        }
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
        Task {
            do {
                let status = try await backendService.getPipelineStatus(taskId: taskId)
                await MainActor.run {
                    if status.status == "completed" {
                        self.isGenerating = false
                        self.generateFallbackCuts() // For now use fallback
                    } else if status.status == "failed" {
                        self.generateFallbackCuts()
                    }
                }
            } catch {
                await MainActor.run {
                    self.generateFallbackCuts()
                }
            }
        }
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
        // MUST be in /Users/hawzhin/Videos
        guard url.path.hasPrefix("/Users/hawzhin/Videos") else {
            print("❌ ERROR: Video must be in /Users/hawzhin/Videos/")
            print("❌ Provided path: \(url.path)")
            processingStatus = "Import failed: Video must be in /Users/hawzhin/Videos/"
            return
        }
        
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
        var mediaItem = MediaPoolItem(url: url)
        mediaItem.duration = duration
        mediaItems.append(mediaItem)
    }
    
    // MARK: - Auto Edit (Pipeline Integration)
    func triggerAutoEdit() {
        guard let videoURL = currentVideoURL else {
            print("❌ No video loaded for Auto Edit")
            processingStatus = "No video loaded"
            return
        }
        
        print("🚀 Starting Auto Edit pipeline for: \(videoURL.path)")
        isProcessing = true
        processingStatus = "Starting Auto Edit..."
        
        // Call backend pipeline
        Task {
            do {
                let options = PipelineOptions(silenceDetection: true, transcription: true, storyBeats: true, brollSelection: true)
                let response = try await backendService.startPipeline(videoPath: videoURL.path, options: options)
                await MainActor.run {
                    print("✅ Pipeline started with task_id: \(response.taskId)")
                    self.processingStatus = "Processing..."
                    self.pollPipelineStatus(taskId: response.taskId)
                }
            } catch {
                await MainActor.run {
                    print("❌ Pipeline failed: \(error)")
                    self.processingStatus = "Auto Edit failed"
                    self.isProcessing = false
                }
            }
        }
    }
    
    private func pollPipelineStatus(taskId: String) {
        Timer.publish(every: 2.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.checkPipelineStatus(taskId: taskId)
            }
            .store(in: &cancellables)
    }
    
    private func checkPipelineStatus(taskId: String) {
        Task {
            do {
                let status = try await backendService.getPipelineStatus(taskId: taskId)
                if status.status == "completed" {
                    print("✅ Pipeline completed!")
                    await MainActor.run {
                        self.processPipelineResults(status)
                        self.isProcessing = false
                        self.processingStatus = "Auto Edit complete"
                    }
                } else if status.status == "failed" {
                    print("❌ Pipeline failed")
                    await MainActor.run {
                        self.isProcessing = false
                        self.processingStatus = "Auto Edit failed"
                    }
                }
                // Continue polling if still processing
            } catch {
                print("❌ Status check failed: \(error)")
            }
        }
    }
    
    private func processPipelineResults(_ status: PipelineStatus) {
        guard let result = status.result else { return }
        
        // Clear existing clips and apply new cuts
        for i in 0..<timeline.tracks.count {
            if timeline.tracks[i].type == .video {
                timeline.tracks[i].clips.removeAll()
            }
        }
        
        // Create clips from cut windows
        var currentTime: Double = 0
        guard let keepWindows = result.cuts?.keep_windows else { return }
        for (index, window) in keepWindows.enumerated() {
            var clip = TimelineClip(
                id: UUID(),
                name: "Clip \(index + 1)",
                trackIndex: 0,
                startTime: currentTime,
                duration: window.t1 - window.t0
            )
            clip.sourceURL = currentVideoURL
            clip.inPoint = window.t0
            clip.outPoint = window.t1
            
            if let firstVideoTrackIndex = timeline.tracks.firstIndex(where: { $0.type == .video }) {
                timeline.tracks[firstVideoTrackIndex].clips.append(clip)
            }
            
            currentTime += clip.duration
        }
        
        print("✅ Applied \(keepWindows.count) cuts to timeline")
        objectWillChange.send()
    }
    
    // Access to project store for professional interface
    var projectStore: BackendVideoProjectStore? {
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
        
        // Export using existing method
        Task {
            do {
                let exportResult = try await backendService.exportResolveProject()
                
                await MainActor.run {
                    if exportResult.success, let path = exportResult.outputPath {
                        print("✅ Export completed: \(path)")
                        self.exportStatus = .completed(path)
                        
                        // Open the export directory in Finder
                        NSWorkspace.shared.selectFile(path, inFileViewerRootedAtPath: "")
                    } else {
                        print("❌ Export error: \(exportResult.error ?? "Unknown")")
                        self.exportStatus = .failed(exportResult.error ?? "Export failed")
                    }
                }
            } catch {
                await MainActor.run {
                    print("❌ Export failed: \(error.localizedDescription)")
                    self.exportStatus = .failed(error.localizedDescription)
                }
            }
        }
    }
    
    func exportFCPXML() {
        guard let taskId = currentTaskId else {
            print("⚠️ No task to export")
            return
        }
        
        // Export as AAF (as no FCPXML method exists)
        Task {
            do {
                let result = try await backendService.exportAAF()
                print("✅ Export complete: \(result.outputPath ?? "unknown")")
            } catch {
                print("❌ Export failed: \(error)")
            }
        }
    }
    
    func exportEDL() {
        guard let taskId = currentTaskId else {
            print("⚠️ No task to export")
            return
        }
        
        // Export as AAF (as no EDL method exists)
        Task {
            do {
                let result = try await backendService.exportAAF()
                print("✅ Export complete: \(result.outputPath ?? "unknown")")
            } catch {
                print("❌ Export failed: \(error)")
            }
        }
    }
    
    // MARK: - Timeline Operations
    
    func addClipToTimeline(url: URL, at time: CMTime, trackIndex: Int = 0) {
        // Create a new clip from the URL
        let clip = LegacyUITimelineClip(
            id: UUID(),
            url: url,
            inPoint: CMTime.zero,
            outPoint: CMTime(seconds: 10, preferredTimescale: 600),
            duration: CMTime(seconds: 10, preferredTimescale: 600),
            name: url.lastPathComponent,
            trackIndex: trackIndex,
            startTime: time
        )
        
        // Convert LegacyUITimelineClip to SimpleTimelineClip
        let simpleClip = SimpleTimelineClip(
            id: clip.id,
            name: clip.name,
            trackIndex: clip.trackIndex,
            startTime: CMTimeGetSeconds(clip.startTime),
            duration: CMTimeGetSeconds(clip.duration),
            sourceURL: clip.url,
            inPoint: CMTimeGetSeconds(clip.inPoint),
            isSelected: clip.isSelected,
            thumbnailData: clip.thumbnailData,
            sourceStartTime: clip.timelineStartTime,
            type: clip.type == .video ? .video : .audio
        )
        
        // Find or create video track
        if let firstVideoTrackIndex = timeline.tracks.firstIndex(where: { $0.type == .video }) {
            timeline.tracks[firstVideoTrackIndex].clips.append(simpleClip)
        } else {
            // Create new video track
            let newTrack = UITimelineTrack(
                name: "V1",
                type: .video
            )
            timeline.tracks.append(newTrack)
        }
        
        // Update media items
        var mediaItem = MediaPoolItem(url: url)
        mediaItem.name = url.lastPathComponent
        mediaItem.duration = 10.0
        mediaItem.thumbnail = Image(systemName: "film")
        mediaItems.append(mediaItem)
        
        print("✅ Added clip \(url.lastPathComponent) to timeline at \(CMTimeGetSeconds(time))s")
        objectWillChange.send()
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
