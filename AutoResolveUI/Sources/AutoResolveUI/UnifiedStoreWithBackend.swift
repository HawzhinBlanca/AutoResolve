// AUTORESOLVE V3.0 - UNIFIED STORE
// Neural Engine Integration + Backend Connection

import SwiftUI
import Combine
import Foundation

// VideoEffectsProcessor is now defined in VideoPlayer/VideoEffectsProcessor.swift

// MARK: - Unified Store
@MainActor
class UnifiedStore: ObservableObject {
    static let shared = UnifiedStore()
    
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
    
    // Backend Service Integration
    private let backendService = BackendService.shared
    @Published var statusMonitor: PipelineStatusMonitor
    
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        self.statusMonitor = PipelineStatusMonitor()
        setupNeuralEngine()
        setupBackendObservers()
        setupDemoData()
    }
    
    private func setupDemoData() {
        // Add demo timeline data to show UI
        timeline.duration = 120.0 // 2 minute demo video
        
        // Add demo video track with clips
        var videoTrack = TimelineTrack(name: "V1", type: .video)
        let clip1 = TimelineClip(name: "Opening Shot", trackIndex: 0, startTime: 0, duration: 15.0)
        let clip2 = TimelineClip(name: "Main Scene", trackIndex: 0, startTime: 15.0, duration: 45.0)
        let clip3 = TimelineClip(name: "B-Roll 1", trackIndex: 0, startTime: 60.0, duration: 10.0)
        let clip4 = TimelineClip(name: "Interview", trackIndex: 0, startTime: 70.0, duration: 30.0)
        let clip5 = TimelineClip(name: "Closing", trackIndex: 0, startTime: 100.0, duration: 20.0)
        
        videoTrack.clips = [clip1, clip2, clip3, clip4, clip5]
        timeline.tracks.append(videoTrack)
        
        // Add demo audio track with clips
        var audioTrack = TimelineTrack(name: "A1", type: .audio)
        let audio1 = TimelineClip(name: "Background Music", trackIndex: 1, startTime: 0, duration: 120.0)
        let audio2 = TimelineClip(name: "Voiceover", trackIndex: 1, startTime: 0, duration: 60.0)
        
        audioTrack.clips = [audio1, audio2]
        timeline.tracks.append(audioTrack)
        
        // Add demo transcription segments
        transcription.segments = [
            TranscriptionSegment(text: "Welcome to AutoResolve", startTime: 0, endTime: 3),
            TranscriptionSegment(text: "This is a professional video editor", startTime: 3, endTime: 7),
            TranscriptionSegment(text: "Powered by AI and neural networks", startTime: 7, endTime: 11),
            TranscriptionSegment(text: "Create amazing videos effortlessly", startTime: 11, endTime: 15)
        ]
        
        // Add demo story beats
        var beat1 = StoryBeat(type: .setup, timeRange: 0...15, confidence: 85, color: .blue)
        var beat2 = StoryBeat(type: .confrontation, timeRange: 15...60, confidence: 78, color: .orange)
        var beat3 = StoryBeat(type: .resolution, timeRange: 60...90, confidence: 92, color: .red)
        var beat4 = StoryBeat(type: .setup, timeRange: 90...120, confidence: 88, color: .green)
        
        director.beats = StoryBeats(all: [beat1, beat2, beat3, beat4])
        
        // Add demo cut suggestions
        cuts.suggestions = [
            CutSuggestion(time: 15.0, confidence: 95),
            CutSuggestion(time: 30.5, confidence: 88),
            CutSuggestion(time: 45.2, confidence: 82),
            CutSuggestion(time: 60.0, confidence: 91),
            CutSuggestion(time: 75.8, confidence: 79),
            CutSuggestion(time: 90.0, confidence: 86)
        ]
        
        // Set director insights
        director.insightCount = 7
        director.isAnalyzing = false
    }
    
    func transcribe() {
        transcribeWithBackend()
    }
    
    func analyzeDirector() {
        analyzeWithBackend()
    }
    
    func apply(_ suggestion: DirectorSuggestion) {
        // Apply AI suggestion
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
        statusMonitor.$currentStatus
            .sink { [weak self] status in
                self?.isProcessing = status != .idle && status != .completed && status != .error
                
                // Update specific components based on status
                switch status {
                case .analyzing:
                    self?.director.isAnalyzing = true
                case .detectingSilence:
                    self?.statusMonitor.addMessage(.info, "Detecting silence regions")
                case .selectingBroll:
                    self?.statusMonitor.addMessage(.info, "Selecting optimal B-roll footage")
                case .completed:
                    self?.director.isAnalyzing = false
                    self?.director.hasInsights = true
                    self?.director.hasNewInsights = true
                case .error:
                    self?.director.isAnalyzing = false
                    self?.statusMonitor.addMessage(.error, "Pipeline processing failed")
                default:
                    break
                }
            }
            .store(in: &cancellables)
        
        // Monitor active jobs
        statusMonitor.$activeJobs
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
        guard let videoUrl = currentVideoUrl else { return }
        
        director.isAnalyzing = true
        statusMonitor.currentStatus = .analyzing
        
        // Start a status monitoring job
        let jobId = statusMonitor.startJob("Video Analysis", inputPath: videoUrl.path, outputPath: "/tmp/analysis_results")
        
        // Start pipeline for video analysis
        backendService.startPipeline(inputFile: videoUrl.path, options: ["analysis": "true"])
            .sink(
                receiveCompletion: { [weak self] completion in
                    if case .failure(let error) = completion {
                        self?.statusMonitor.addMessage(.error, "Pipeline start failed: \(error.localizedDescription)")
                        self?.statusMonitor.completeJob(jobId, success: false, error: error.localizedDescription)
                        self?.director.isAnalyzing = false
                    }
                },
                receiveValue: { [weak self] response in
                    self?.statusMonitor.addMessage(.info, "Analysis pipeline started with task ID: \(response.task_id)")
                    self?.monitorPipelineStatus(taskId: response.task_id, jobId: jobId)
                }
            )
            .store(in: &cancellables)
    }
    
    private func transcribeWithBackend() {
        guard let audioUrl = currentAudioUrl else { return }
        
        let jobId = statusMonitor.startJob("Audio Transcription", inputPath: audioUrl.path, outputPath: "/tmp/transcription_results")
        
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
                        self?.statusMonitor.addMessage(.error, "Status check failed: \(error.localizedDescription)")
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
            statusMonitor.progressPercentage = progress
            
            // Update specific job progress
            if let jobIndex = statusMonitor.activeJobs.firstIndex(where: { $0.id == jobId }) {
                statusMonitor.activeJobs[jobIndex].progress = progress
            }
        }
        
        // Update pipeline status based on backend response
        let pipelineStatus = PipelineStatusMonitor.PipelineStatus(rawValue: status.status) ?? .idle
        statusMonitor.currentStatus = pipelineStatus
        
        // Update current operation description
        if let operation = status.current_operation {
            statusMonitor.currentOperation = operation
        }
        
        // Handle completion
        if status.status == "completed" {
            statusMonitor.completeJob(jobId, success: true)
            statusMonitor.currentStatus = .completed
            director.isAnalyzing = false
            director.hasInsights = true
            director.hasNewInsights = true
            director.insightCount = 5
            statusMonitor.addMessage(.info, "Pipeline completed successfully")
        } else if status.status == "failed" {
            let errorMsg = status.error ?? "Unknown error"
            statusMonitor.completeJob(jobId, success: false, error: errorMsg)
            statusMonitor.currentStatus = .error
            director.isAnalyzing = false
            statusMonitor.addMessage(.error, "Pipeline failed: \(errorMsg)")
        }
        
        // Update performance metrics if available
        if let metrics = status.performance_metrics {
            statusMonitor.performanceMetrics.cpuUsagePercent = metrics.cpu_usage ?? 0.0
            statusMonitor.performanceMetrics.memoryUsedMB = metrics.memory_mb ?? 0.0
            statusMonitor.performanceMetrics.framesProcessedPerSecond = metrics.fps ?? 0.0
        }
    }
    
    // MARK: - Current Media URLs
    private var currentVideoUrl: URL? {
        // Return the currently loaded video URL
        // This would come from the imported media
        return URL(fileURLWithPath: "/tmp/current_video.mp4")
    }
    
    private var currentAudioUrl: URL? {
        // Return the currently loaded audio URL
        return currentVideoUrl // Same file for audio extraction
    }
}

// MARK: - Project
struct Project {
    var name = "Untitled Project"
    var duration: TimeInterval = 120.0
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
    @Published var continuityScores: [ContinuityScore] = []
    
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
struct StoryBeats {
    var all: [StoryBeat] = []
}

struct StoryBeat: Identifiable, Hashable {
    let id = UUID()
    var type: BeatType = .setup
    var timeRange: ClosedRange<TimeInterval> = 0...10
    var confidence: Int = 80
    var color: Color = .blue
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    static func == (lhs: StoryBeat, rhs: StoryBeat) -> Bool {
        lhs.id == rhs.id
    }
}

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
    let id = UUID()
    var time: TimeInterval
    var intensity: Double
    var type: String = "peak"
    var timeRange: ClosedRange<TimeInterval> { time...time+5 }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    static func == (lhs: TensionPeak, rhs: TensionPeak) -> Bool {
        lhs.id == rhs.id
    }
}

struct EmphasisMoment: Identifiable, Hashable {
    let id = UUID()
    var time: TimeInterval
    var strength: Double
    var type: String = "visual"
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    static func == (lhs: EmphasisMoment, rhs: EmphasisMoment) -> Bool {
        lhs.id == rhs.id
    }
}

struct SuggestedCut: Identifiable {
    let id = UUID()
    var time: TimeInterval
    var reason: String
    
    var timecode: String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        let frames = Int((time.truncatingRemainder(dividingBy: 1)) * 30)
        return String(format: "%02d:%02d:%02d", minutes, seconds, frames)
    }
}

struct CutSuggestion: Identifiable {
    let id = UUID()
    var time: TimeInterval
    var confidence: Int
}

struct SimpleVideoTrack {
    var clips: [SimpleVideoClip] = []
}

struct SimpleAudioTrack {
    var clips: [SimpleAudioClip] = []
}

struct SimpleVideoClip: Identifiable {
    let id = UUID()
    var startTime: TimeInterval = 0
    var duration: TimeInterval = 30
}

struct SimpleAudioClip: Identifiable {
    let id = UUID()
    var startTime: TimeInterval = 0
    var duration: TimeInterval = 30
}

struct Segment: Identifiable {
    let id = UUID()
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
    let id = UUID()
    var title: String = "Suggestion"
    var description: String = "Description"
    var icon: String = "lightbulb"
    var priority: Priority = .medium
}

enum Priority {
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
    let id = UUID()
    var text: String = ""
    var startTime: TimeInterval = 0
    var endTime: TimeInterval = 0
}

struct MotionVector {
    var start: CGPoint
    var end: CGPoint
    var magnitude: Double
}

struct ContinuityScore {
    var fromShot: Int
    var toShot: Int
    var score: Double
}

// MARK: - Extensions
extension UnifiedStore {
    func setupNeuralEngine() {
        // Setup Neural Engine processing
        director.useVJEPA = true
        
        // Initialize with sample data
        director.tensionCurve = (0..<100).map { _ in Double.random(in: 0...1) }
        director.energyCurve = (0..<100).map { _ in Double.random(in: 0...1) }
        
        // Sample story beats
        director.beats.all = [
            StoryBeat(type: .setup, timeRange: 0...30, confidence: 85, color: .green),
            StoryBeat(type: .confrontation, timeRange: 30...90, confidence: 92, color: .orange),
            StoryBeat(type: .resolution, timeRange: 90...120, confidence: 78, color: .blue)
        ]
        
        // Sample tension peaks
        director.tensionPeaks = [
            TensionPeak(time: 45, intensity: 0.8),
            TensionPeak(time: 75, intensity: 0.9),
            TensionPeak(time: 105, intensity: 0.7)
        ]
        
        // Sample emphasis moments
        director.emphasisMoments = [
            EmphasisMoment(time: 25, strength: 0.6, type: "visual"),
            EmphasisMoment(time: 50, strength: 0.8, type: "audio"),
            EmphasisMoment(time: 85, strength: 0.7, type: "motion")
        ]
        
        // Sample next suggested cut
        director.nextSuggestedCut = SuggestedCut(
            time: 42.5,
            reason: "Natural pause detected"
        )
        
        // Sample cuts
        cuts.suggestions = [
            CutSuggestion(time: 15, confidence: 78),
            CutSuggestion(time: 42, confidence: 89),
            CutSuggestion(time: 67, confidence: 76)
        ]
        
        // Sample continuity scores
        director.continuityScores = [
            ContinuityScore(fromShot: 0, toShot: 1, score: 0.85),
            ContinuityScore(fromShot: 1, toShot: 2, score: 0.72),
            ContinuityScore(fromShot: 2, toShot: 3, score: 0.91),
            ContinuityScore(fromShot: 3, toShot: 4, score: 0.68)
        ]
        
        // Sample timeline segments - not used with TimelineModel
        // timeline.segments = [
        //     Segment(start: 0, end: 30, score: 0.8),
        //     Segment(start: 30, end: 60, score: 0.9),
        //     Segment(start: 60, end: 90, score: 0.7),
        //     Segment(start: 90, end: 120, score: 0.85)
        // ]
        
        // Current segment with AI insights - not used with TimelineModel
        /* timeline.currentSegment = TimelineSegment(
            storyPhase: .confrontation,
            energy: 0.75,
            tension: 0.82,
            continuityScore: 0.89,
            suggestions: [
                DirectorSuggestion(
                    title: "Increase Tension",
                    description: "Add dramatic music cue",
                    icon: "music.note",
                    priority: .high
                ),
                DirectorSuggestion(
                    title: "Cut Suggestion",
                    description: "Quick cut at 45.2s for impact",
                    icon: "scissors",
                    priority: .medium
                )
            ]
        ) */
    }
    
    // MARK: - Professional Video Import
    func importVideo(url: URL, duration: Double) {
        print("🎯 UnifiedStore: Importing video to timeline")
        
        // Create new clip for timeline
        var newClip = TimelineClip(
            name: url.lastPathComponent,
            trackIndex: 0,
            startTime: timeline.duration,
            duration: duration
        )
        newClip.sourceURL = url
        
        // Add to first video track (create if needed)
        if timeline.videoTracks.isEmpty {
            timeline.tracks.append(TimelineTrack(name: "V1", type: .video))
        }
        
        // Find first video track and add clip
        if let videoTrackIndex = timeline.tracks.firstIndex(where: { $0.type == .video }) {
            timeline.tracks[videoTrackIndex].addClip(newClip)
        }
        
        timeline.duration += duration
        
        print("✅ Video added to timeline. Total duration: \(timeline.duration)s")
        
        // Trigger UI update
        objectWillChange.send()
    }
    
    // Access to project store for professional interface
    var projectStore: VideoProjectStore? {
        return nil // Simplified for now - UnifiedStore manages timeline directly
    }
}