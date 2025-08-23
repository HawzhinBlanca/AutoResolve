import SwiftUI
import Combine
import AVFoundation
import os.log

// MARK: - Timeline Backend Bridge

@MainActor
public class TimelineBackendBridge: ObservableObject {
    // Dependencies
    @Published public var timeline: TimelineModel
    @Published public var pipelineManager: PipelineIntegrationManager
    @Published public var autoResolveService: AutoResolveService
    
    // Processing state
    @Published public var isAutoProcessing = false
    @Published public var lastProcessingTime: Date?
    @Published public var processingQueue: [ProcessingTask] = []
    @Published public var silenceMarkers: [SilenceMarker] = []
    @Published public var brollSuggestionMarkers: [BRollMarker] = []
    
    // UI state
    @Published public var showProcessingOverlay = false
    @Published public var showSilenceVisualization = true
    @Published public var showBRollSuggestions = true
    @Published public var showTelemetryData = false
    
    // Settings
    @Published public var autoProcessingSettings = AutoProcessingSettings()
    @Published public var visualizationSettings = VisualizationSettings()
    
    private let logger = Logger(subsystem: "com.autoresolve", category: "timeline-bridge")
    private var cancellables = Set<AnyCancellable>()
    private var debounceTimer: Timer?
    private var currentProcessingTask: ProcessingTask?
    
    public init(
        timeline: TimelineModel,
        pipelineManager: PipelineIntegrationManager,
        autoResolveService: AutoResolveService
    ) {
        self.timeline = timeline
        self.pipelineManager = pipelineManager
        self.autoResolveService = autoResolveService
        
        setupObservers()
        loadProcessingHistory()
    }
    
    // MARK: - Main Processing Integration
    
    public func processTimelineWithBackend() async throws {
        logger.info("Processing timeline with AutoResolve backend")
        
        guard let primaryVideoClip = getPrimaryVideoClip() else {
            throw AutoResolveError.processingError("No primary video clip found in timeline")
        }
        
        guard let videoPath = primaryVideoClip.sourceURL?.path else {
            throw AutoResolveError.processingError("Primary video clip has no source path")
        }
        
        // Create processing task
        var task = ProcessingTask(
            id: UUID(),
            type: .fullPipeline,
            inputPath: videoPath,
            startTime: Date(),
            status: .processing
        )
        
        currentProcessingTask = task
        processingQueue.append(task)
        showProcessingOverlay = true
        
        // Cleanup always executes regardless of success or failure
        defer {
            currentProcessingTask = nil
            showProcessingOverlay = false
        }
        
        do {
            // Process with pipeline
            let result = try await pipelineManager.processFullPipeline(
                inputVideoPath: videoPath,
                outputDirectory: getOutputDirectory()
            )
            
            // Apply results to timeline
            await applyProcessingResults(result)
            
            // Update task status
            task.status = .completed
            task.endTime = Date()
            task.result = result
            
            lastProcessingTime = Date()
            logger.info("Timeline processing completed successfully")
            
        } catch {
            task.status = .failed
            task.endTime = Date()
            task.error = error
            
            logger.error("Timeline processing failed: \(error)")
            throw error
        }
    }
    
    public func processOnlySilenceDetection() async throws {
        guard let videoPath = getPrimaryVideoClip()?.sourceURL?.path else {
            throw AutoResolveError.processingError("No video to process")
        }
        
        logger.info("Running silence detection only")
        
        var task = ProcessingTask(
            id: UUID(),
            type: .silenceDetection,
            inputPath: videoPath,
            startTime: Date(),
            status: .processing
        )
        
        processingQueue.append(task)
        
        do {
            let silenceResult = try await pipelineManager.detectSilenceInVideo(videoPath)
            await applySilenceResults(silenceResult)
            
            task.status = .completed
            task.endTime = Date()
            
            logger.info("Silence detection completed: \(silenceResult.silenceSegments.count) segments")
            
        } catch {
            task.status = .failed
            task.error = error
            throw error
        }
    }
    
    public func processBRollSelection() async throws {
        guard let videoPath = getPrimaryVideoClip()?.sourceURL?.path else {
            throw AutoResolveError.processingError("No video to process")
        }
        
        let cuts = extractTimelineCuts()
        guard !cuts.isEmpty else {
            throw AutoResolveError.processingError("No cuts found to process")
        }
        
        logger.info("Running B-roll selection for \(cuts.count) cuts")
        
        var task = ProcessingTask(
            id: UUID(),
            type: .brollSelection,
            inputPath: videoPath,
            startTime: Date(),
            status: .processing
        )
        
        processingQueue.append(task)
        
        do {
            let brollResult = try await pipelineManager.selectBRollForCuts(
                videoPath: videoPath,
                cuts: cuts
            )
            
            await applyBRollResults(brollResult)
            
            task.status = .completed
            task.endTime = Date()
            
            logger.info("B-roll selection completed: \(brollResult.selections.count) suggestions")
            
        } catch {
            task.status = .failed
            task.error = error
            throw error
        }
    }
    
    public func exportToResolve() async throws -> ResolveProjectResult {
        guard let videoPath = getPrimaryVideoClip()?.sourceURL?.path else {
            throw AutoResolveError.processingError("No video to export")
        }
        
        let cuts = extractTimelineCuts()
        let brollSelections = extractBRollSelections()
        
        logger.info("Exporting timeline to DaVinci Resolve")
        
        let result = try await pipelineManager.createResolveProject(
            inputVideoPath: videoPath,
            cuts: cuts,
            brollSelections: brollSelections
        )
        
        if result.success {
            logger.info("Successfully created Resolve project: \(result.projectPath ?? "Unknown path")")
        }
        
        return result
    }
    
    // MARK: - Auto Processing
    
    public func enableAutoProcessing() {
        logger.info("Enabling auto-processing")
        autoProcessingSettings.enabled = true
        setupAutoProcessingObserver()
    }
    
    public func disableAutoProcessing() {
        logger.info("Disabling auto-processing")
        autoProcessingSettings.enabled = false
        debounceTimer?.invalidate()
        debounceTimer = nil
    }
    
    private func setupAutoProcessingObserver() {
        guard autoProcessingSettings.enabled else { return }
        
        // Observe timeline changes
        timeline.objectWillChange
            .debounce(for: .seconds(autoProcessingSettings.debounceDelay), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in
                Task { [weak self] in
                    await self?.triggerAutoProcessing()
                }
            }
            .store(in: &cancellables)
    }
    
    private func triggerAutoProcessing() async {
        guard autoProcessingSettings.enabled,
              !isAutoProcessing,
              !pipelineManager.isProcessingVideo,
              shouldTriggerAutoProcessing() else {
            return
        }
        
        logger.info("Triggering auto-processing")
        
        isAutoProcessing = true
        defer { isAutoProcessing = false }
        
        do {
            switch autoProcessingSettings.mode {
            case .silenceOnly:
                try await processOnlySilenceDetection()
            case .full:
                try await processTimelineWithBackend()
            case .brollOnly:
                try await processBRollSelection()
            }
            
        } catch {
            logger.error("Auto-processing failed: \(error)")
        }
    }
    
    private func shouldTriggerAutoProcessing() -> Bool {
        // Don't auto-process too frequently
        if let lastTime = lastProcessingTime {
            let timeSinceLastProcess = Date().timeIntervalSince(lastTime)
            if timeSinceLastProcess < autoProcessingSettings.minInterval {
                return false
            }
        }
        
        // Check if timeline has meaningful content
        let hasVideoClips = !timeline.tracks.isEmpty && timeline.tracks.contains { !$0.clips.isEmpty }
        return hasVideoClips
    }
    
    // MARK: - Timeline Data Extraction
    
    private func getPrimaryVideoClip() -> TimelineClip? {
        return timeline.tracks.first?.clips.first
    }
    
    private func extractTimelineCuts() -> [TimeRange] {
        var cuts: [TimeRange] = []
        
        for track in timeline.tracks where track.type == .video {
            for clip in track.clips {
                cuts.append(TimeRange(
                    start: clip.startTime,
                    end: clip.startTime + clip.duration
                ))
            }
        }
        
        return cuts.sorted { $0.start < $1.start }
    }
    
    private func extractBRollSelections() -> [BRollSelection] {
        // Extract B-roll selections from timeline markers or separate tracks
        var selections: [BRollSelection] = []
        
        for marker in brollSuggestionMarkers {
            if let brollPath = marker.suggestedBRollPath {
                let selection = BRollSelection(
                    cutIndex: marker.cutIndex,
                    timeRange: marker.timeRange,
                    brollPath: brollPath,
                    confidence: marker.confidence,
                    reason: marker.reason
                )
                selections.append(selection)
            }
        }
        
        return selections
    }
    
    // MARK: - Result Application
    
    private func applyProcessingResults(_ result: ProcessingResult) async {
        logger.info("Applying processing results to timeline")
        
        // Apply silence detection results
        if let silenceSegments = result.silenceSegments {
            await applySilenceVisualization(silenceSegments)
        }
        
        // Apply B-roll selections
        if let brollSelections = result.brollSelections {
            await applyBRollVisualization(brollSelections)
        }
        
        // Update timeline based on cuts
        if let cuts = extractCutsFromResult(result) {
            await updateTimelineFromCuts(cuts)
        }
    }
    
    private func applySilenceResults(_ result: SilenceDetectionResult) async {
        await applySilenceVisualization(result.silenceSegments)
    }
    
    private func applySilenceVisualization(_ silenceSegments: [TimeRange]) async {
        // Create silence markers for visualization
        silenceMarkers = silenceSegments.enumerated().map { index, segment in
            SilenceMarker(
                id: UUID(),
                timeRange: segment,
                confidence: 0.9, // From backend result
                visualizationType: visualizationSettings.silenceVisualizationType
            )
        }
        
        logger.info("Applied \(self.silenceMarkers.count) silence markers to timeline")
    }
    
    private func applyBRollResults(_ result: BRollSelectionResult) async {
        await applyBRollVisualization(result.selections)
    }
    
    private func applyBRollVisualization(_ brollSelections: [BRollSelection]) async {
        // Create B-roll markers for visualization
        brollSuggestionMarkers = brollSelections.map { selection in
            BRollMarker(
                id: UUID(),
                cutIndex: selection.cutIndex,
                timeRange: selection.timeRange,
                suggestedBRollPath: selection.brollPath,
                confidence: selection.confidence,
                reason: selection.reason,
                visualizationType: visualizationSettings.brollVisualizationType
            )
        }
        
        logger.info("Applied \(self.brollSuggestionMarkers.count) B-roll markers to timeline")
    }
    
    private func updateTimelineFromCuts(_ cuts: [TimeRange]) async {
        // This would reconstruct the timeline based on detected cuts
        // Implementation depends on specific requirements
        logger.info("Would update timeline from \(cuts.count) cuts")
    }
    
    private func extractCutsFromResult(_ result: ProcessingResult) -> [TimeRange]? {
        // Extract cuts from processing result
        // This is a simplified implementation
        if let silenceSegments = result.silenceSegments {
            return generateCutsFromSilence(silenceSegments)
        }
        return nil
    }
    
    private func generateCutsFromSilence(_ silenceSegments: [TimeRange]) -> [TimeRange] {
        // Convert silence segments to content cuts
        var cuts: [TimeRange] = []
        var lastEnd: Double = 0
        
        for silence in silenceSegments.sorted(by: { $0.start < $1.start }) {
            if silence.start > lastEnd {
                cuts.append(TimeRange(start: lastEnd, end: silence.start))
            }
            lastEnd = max(lastEnd, silence.end)
        }
        
        // Add final cut if needed
        if lastEnd < timeline.duration {
            cuts.append(TimeRange(start: lastEnd, end: timeline.duration))
        }
        
        return cuts.filter { $0.duration > 1.0 } // Filter out very short cuts
    }
    
    // MARK: - Utility Methods
    
    private func getOutputDirectory() -> String {
        // Get user's preferred output directory or use temp
        return FileManager.default.temporaryDirectory.path
    }
    
    private func setupObservers() {
        // Observe pipeline status changes
        pipelineManager.$isProcessingVideo
            .sink { [weak self] isProcessing in
                if !isProcessing {
                    self?.showProcessingOverlay = false
                }
            }
            .store(in: &cancellables)
        
        // Observe service connection changes
        autoResolveService.$isConnected
            .sink { [weak self] isConnected in
                self?.logger.info("Backend service connection: \(isConnected)")
            }
            .store(in: &cancellables)
    }
    
    private func loadProcessingHistory() {
        // Load previous processing history from disk
        // This is a placeholder for persistent storage
    }
    
    // MARK: - Public Interface
    
    public func clearSilenceMarkers() {
        silenceMarkers.removeAll()
        logger.info("Cleared silence markers")
    }
    
    public func clearBRollMarkers() {
        brollSuggestionMarkers.removeAll()
        logger.info("Cleared B-roll markers")
    }
    
    public func refreshVisualization() async {
        logger.info("Refreshing timeline visualization")
        
        if let lastResult = pipelineManager.lastProcessingResult {
            await applyProcessingResults(lastResult)
        }
    }
    
    public func getProcessingStats() -> ProcessingStats {
        let completedTasks = processingQueue.filter { $0.status == .completed }
        let failedTasks = processingQueue.filter { $0.status == .failed }
        
        let totalProcessingTime = completedTasks.reduce(0) { total, task in
            total + (task.processingDuration ?? 0)
        }
        
        return ProcessingStats(
            totalTasks: processingQueue.count,
            completedTasks: completedTasks.count,
            failedTasks: failedTasks.count,
            totalProcessingTime: totalProcessingTime,
            averageProcessingTime: completedTasks.isEmpty ? 0 : totalProcessingTime / Double(completedTasks.count),
            successRate: processingQueue.isEmpty ? 1.0 : Double(completedTasks.count) / Double(processingQueue.count),
            lastProcessingTime: lastProcessingTime
        )
    }
}

// MARK: - Supporting Models

public struct ProcessingTask: Identifiable {
    public let id: UUID
    public let type: TaskType
    public let inputPath: String
    public let startTime: Date
    public var endTime: Date?
    public var status: TaskStatus
    public var result: ProcessingResult?
    public var error: Error?
    
    public var processingDuration: TimeInterval? {
        guard let endTime = endTime else { return nil }
        return endTime.timeIntervalSince(startTime)
    }
    
    public enum TaskType: String, CaseIterable {
        case fullPipeline = "Full Pipeline"
        case silenceDetection = "Silence Detection"
        case brollSelection = "B-roll Selection"
        case resolveExport = "Resolve Export"
    }
    
    public enum TaskStatus: String, CaseIterable {
        case pending = "Pending"
        case processing = "Processing"
        case completed = "Completed"
        case failed = "Failed"
    }
}

public struct SilenceMarker: Identifiable {
    public let id: UUID
    public let timeRange: TimeRange
    public let confidence: Double
    public let visualizationType: SilenceVisualizationType
}

public struct BRollMarker: Identifiable {
    public let id: UUID
    public let cutIndex: Int
    public let timeRange: TimeRange
    public let suggestedBRollPath: String?
    public let confidence: Double
    public let reason: String
    public let visualizationType: BRollVisualizationType
}

public struct AutoProcessingSettings {
    public var enabled: Bool = false
    public var mode: Mode = .silenceOnly
    public var debounceDelay: Double = 2.0
    public var minInterval: TimeInterval = 10.0
    
    public enum Mode: String, CaseIterable {
        case silenceOnly = "Silence Only"
        case brollOnly = "B-roll Only"
        case full = "Full Pipeline"
    }
}

public struct VisualizationSettings {
    public var silenceVisualizationType: SilenceVisualizationType = .overlay
    public var brollVisualizationType: BRollVisualizationType = .suggestions
    public var showConfidenceScores = true
    public var animateMarkers = true
}

public enum SilenceVisualizationType: String, CaseIterable {
    case overlay = "Overlay"
    case waveform = "Waveform"
    case markers = "Markers"
}

public enum BRollVisualizationType: String, CaseIterable {
    case suggestions = "Suggestions"
    case overlays = "Overlays"
    case timeline = "Timeline Track"
}

public struct ProcessingStats {
    public let totalTasks: Int
    public let completedTasks: Int
    public let failedTasks: Int
    public let totalProcessingTime: TimeInterval
    public let averageProcessingTime: TimeInterval
    public let successRate: Double
    public let lastProcessingTime: Date?
    
    public var formattedStats: String {
        return """
        Total: \(totalTasks)
        Completed: \(completedTasks)
        Failed: \(failedTasks)
        Success Rate: \(String(format: "%.1f", successRate * 100))%
        Avg Time: \(String(format: "%.1f", averageProcessingTime))s
        """
    }
}