import SwiftUI
import Combine
import AVFoundation
import os.log

// MARK: - Pipeline Integration Manager

@MainActor
public class PipelineIntegrationManager: ObservableObject {
    // Core dependencies
    @Published public var autoResolveService: AutoResolveService
    @Published public var timeline: TimelineModel
    @Published internal var project: VideoProjectStore
    
    // Pipeline state
    @Published public var isProcessingVideo = false
    @Published public var processingProgress: Double = 0
    @Published public var processingStatus = ""
    @Published public var lastProcessingResult: ProcessingResult?
    @Published public var detectedSilenceRanges: [TimeRange] = []
    @Published public var brollSuggestions: [PipelineBRollSuggestion] = []
    @Published public var availableBRollClips: [BRollClip] = []
    
    // Settings
    @Published public var pipelineSettings = PipelineSettings()
    @Published public var autoProcessingEnabled = true
    @Published public var realTimePreviewEnabled = false
    
    // Status and monitoring
    @Published public var systemStatus: SystemStatus?
    @Published public var performanceMetrics = LocalPipelinePerformanceMetrics()
    @Published public var processingHistory: [ProcessingHistoryItem] = []
    
    private let logger = Logger.shared
    private var cancellables = Set<AnyCancellable>()
    
    internal init(
        autoResolveService: AutoResolveService,
        timeline: TimelineModel,
        project: VideoProjectStore
    ) {
        self.autoResolveService = autoResolveService
        self.timeline = timeline
        self.project = project
        
        setupObservers()
        loadBRollLibrary()
    }
    
    // MARK: - Main Pipeline Operations
    
    public func processFullPipeline(
        inputVideoPath: String,
        outputDirectory: String
    ) async throws -> ProcessingResult {
        
        logger.info("Starting full AutoResolve pipeline for: \(inputVideoPath)", category: .pipeline)
        
        isProcessingVideo = true
        processingProgress = 0
        processingStatus = "Initializing pipeline..."
        
        defer {
            isProcessingVideo = false
        }
        
        let startTime = Date()
        
        do {
            // Step 1: Validate input file
            processingStatus = "Validating input file..."
            processingProgress = 0.05
            try validateInputFile(inputVideoPath)
            
            // Step 2: Detect silence segments
            processingStatus = "Detecting silence segments..."
            processingProgress = 0.15
            let silenceResult = try await detectSilenceInVideo(inputVideoPath)
            detectedSilenceRanges = silenceResult.silenceSegments
            
            // Step 3: Generate cuts from silence
            processingStatus = "Generating cuts from silence detection..."
            processingProgress = 0.25
            let generatedCuts = generateCutsFromSilence(silenceResult.silenceSegments)
            
            // Step 4: Apply cuts to timeline
            processingStatus = "Applying cuts to timeline..."
            processingProgress = 0.35
            await applyCutsToTimeline(generatedCuts, videoPath: inputVideoPath)
            
            // Step 5: Select B-roll if enabled
            var brollSelections: [BRollSelection]? = nil
            if pipelineSettings.enableBRollSelection && !availableBRollClips.isEmpty {
                processingStatus = "Selecting matching B-roll content..."
                processingProgress = 0.55
                
                let brollResult = try await selectBRollForCuts(
                    videoPath: inputVideoPath,
                    cuts: generatedCuts
                )
                // Convert BackendBRollSelection to BRollSelection
                let convertedSelections = brollResult.selections.map { backend in
                    BRollSelection(
                        cutIndex: backend.cutIndex,
                        timeRange: backend.timeRange,
                        brollClipPath: backend.brollClipPath,
                        confidence: backend.confidence
                    )
                }
                brollSelections = convertedSelections
                await applyBRollSelections(convertedSelections)
                
                processingProgress = 0.75
            }
            
            // Step 6: Export or create Resolve project
            processingStatus = "Finalizing output..."
            processingProgress = 0.85
            
            var resolveProjectPath: String? = nil
            if pipelineSettings.createResolveProject {
                processingStatus = "Creating DaVinci Resolve project..."
                let resolveResult = try await createResolveProject(
                    inputVideoPath: inputVideoPath,
                    cuts: generatedCuts,
                    brollSelections: brollSelections
                )
                resolveProjectPath = resolveResult.projectPath
            }
            
            // Step 7: Complete processing
            processingStatus = "Processing complete!"
            processingProgress = 1.0
            
            let processingTime = Date().timeIntervalSince(startTime)
            
            let result = ProcessingResult(
                success: true,
                message: "Pipeline completed successfully",
                outputPath: resolveProjectPath ?? outputDirectory,
                processingTime: processingTime,
                silenceSegments: detectedSilenceRanges,
                brollSelections: brollSelections,
                resolveProjectPath: resolveProjectPath,
                telemetry: createProcessingTelemetry(startTime: startTime, endTime: Date())
            )
            
            lastProcessingResult = result
            updateProcessingHistory(result, inputPath: inputVideoPath)
            
            logger.info("Pipeline completed in \(processingTime) seconds", category: .pipeline)
            return result
            
        } catch {
            logger.error("Pipeline failed: \(error)", category: .pipeline)
            processingStatus = "Processing failed: \(error.localizedDescription)"
            
            let failedResult = ProcessingResult(
                success: false,
                message: error.localizedDescription,
                outputPath: nil,
                processingTime: Date().timeIntervalSince(startTime),
                silenceSegments: nil,
                brollSelections: nil,
                resolveProjectPath: nil,
                telemetry: nil
            )
            
            lastProcessingResult = failedResult
            updateProcessingHistory(failedResult, inputPath: inputVideoPath)
            
            throw error
        }
    }
    
    // MARK: - Individual Pipeline Steps
    
    public func detectSilenceInVideo(_ videoPath: String) async throws -> SilenceDetectionResult {
        logger.info("Detecting silence in video: \(videoPath)", category: .pipeline)
        
        let settings = BackendSilenceDetectionSettings(
            threshold: pipelineSettings.silenceThreshold,
            minDuration: pipelineSettings.minSilenceDuration,
            padding: pipelineSettings.silencePadding
        )
        
        return try await autoResolveService.detectSilence(
            videoPath: videoPath,
            settings: settings
        )
    }
    
    public func selectBRollForCuts(
        videoPath: String,
        cuts: [TimeRange]
    ) async throws -> BRollSelectionResult {
        logger.info("Selecting B-roll for \(cuts.count) cuts", category: .pipeline)
        
        let backendCuts = cuts.map { BackendTimeRange(start: $0.start, end: $0.end) }
        
        let settings = BackendBRollSettings(
            brollDirectory: pipelineSettings.brollDirectory,
            maxResults: pipelineSettings.maxBRollResults,
            confidenceThreshold: 0.7,
            enableVJEPA: pipelineSettings.enableVJEPA
        )
        
        return try await autoResolveService.selectBRoll(
            videoPath: videoPath,
            cuts: backendCuts,
            settings: settings
        )
    }
    
    public func createResolveProject(
        inputVideoPath: String,
        cuts: [TimeRange],
        brollSelections: [BRollSelection]? = nil
    ) async throws -> ResolveProjectResult {
        logger.info("Creating DaVinci Resolve project", category: .pipeline)
        
        let timelineName = generateTimelineName(for: inputVideoPath)
        let backendCuts = cuts.map { BackendTimeRange(start: $0.start, end: $0.end) }
        let backendBRoll = brollSelections?.map { 
            BackendBRollSelection(
                cutIndex: $0.cutIndex,
                timeRange: $0.timeRange,
                brollPath: $0.brollClipPath,
                confidence: $0.confidence,
                reason: "Selected B-roll"
            )
        }
        
        return try await autoResolveService.createResolveProject(
            timelineName: timelineName,
            videoPath: inputVideoPath,
            cuts: backendCuts,
            brollSelections: backendBRoll
        )
    }
    
    // MARK: - Timeline Integration
    
    private func applyCutsToTimeline(_ cuts: [TimeRange], videoPath: String) async {
        logger.info("Applying \(cuts.count) cuts to timeline", category: .pipeline)
        
        // Clear existing clips
        timeline.tracks.removeAll()
        
        // Create video track
        var videoTrack = UITimelineTrack(
            name: "V1",
            type: .video
        )
        videoTrack.height = 60
        videoTrack.isEnabled = true
        
        // Add clips for each cut
        var currentTime: TimeInterval = 0
        
        for (index, cut) in cuts.enumerated() {
            let clipName = "Segment \(index + 1)"
            let duration = cut.duration
            
            let clip = SimpleTimelineClip(id: UUID(), 
                name: clipName,
                trackIndex: 0,
                startTime: currentTime,
                duration: duration,
                sourceURL: URL(fileURLWithPath: videoPath),
                inPoint: cut.start
            )
            
            // Add to track
            var updatedTrack = videoTrack
            updatedTrack.clips.append(clip)
            
            currentTime += duration
        }
        
        // Update timeline
        timeline.tracks = [videoTrack]
        timeline.duration = currentTime
        timeline.playheadPosition = 0
    }
    
    private func applyBRollSelections(_ selections: [BRollSelection]) async {
        logger.info("Applying \(selections.count) B-roll selections", category: .pipeline)
        
        // Create or get B-roll track
        var brollTrack: UITimelineTrack
        if timeline.tracks.count > 1 {
            brollTrack = timeline.tracks[1]
        } else {
            brollTrack = UITimelineTrack(
                name: "B-roll",
                type: .video
            )
            brollTrack.height = 60
            brollTrack.isEnabled = true
            timeline.tracks.append(brollTrack)
        }
        
        // Add B-roll clips
        for selection in selections {
            let brollClip = SimpleTimelineClip(id: UUID(), 
                name: "B-roll \(selection.cutIndex + 1)",
                trackIndex: 1,
                startTime: selection.timeRange.start,
                duration: selection.timeRange.duration,
                sourceURL: URL(fileURLWithPath: selection.brollClipPath),
                inPoint: 0
            )
            
            // Add B-roll clip to track
            brollTrack.clips.append(brollClip)
        }
        
        // Update timeline
        if timeline.tracks.count > 1 {
            timeline.tracks[1] = brollTrack
        }
    }
    
    // MARK: - B-roll Management
    
    public func loadBRollLibrary() {
        Task {
            do {
                // Load B-roll clips from directory
                let brollDirectory = pipelineSettings.brollDirectory
                let brollClips = try await scanBRollDirectory(brollDirectory)
                
                await MainActor.run {
                    self.availableBRollClips = brollClips
                    logger.info("Loaded \(brollClips.count) B-roll clips", category: .media)
                }
                
            } catch {
                logger.error("Failed to load B-roll library: \(error)", category: .media)
            }
        }
    }
    
    public func addBRollClip(_ url: URL) async throws {
        let brollClip = try await createBRollClip(from: url)
        availableBRollClips.append(brollClip)
        logger.info("Added B-roll clip: \(url.lastPathComponent)", category: .media)
    }
    
    public func removeBRollClip(_ clipId: UUID) {
        availableBRollClips.removeAll { $0.id == clipId }
        logger.info("Removed B-roll clip", category: .media)
    }
    
    // MARK: - Settings and Configuration
    
    public func updatePipelineSettings(_ newSettings: PipelineSettings) {
        pipelineSettings = newSettings
        logger.info("Updated pipeline settings", category: .pipeline)
    }
    
    public func resetToDefaults() {
        pipelineSettings = PipelineSettings()
        logger.info("Reset pipeline settings to defaults", category: .pipeline)
    }
    
    // MARK: - Status and Monitoring
    
    public func refreshSystemStatus() async {
        do {
            let backendStatus = try await autoResolveService.getSystemStatus()
            systemStatus = SystemStatus(
                status: backendStatus.status,
                version: backendStatus.version,
                uptime: backendStatus.uptime,
                memoryUsage: backendStatus.memoryUsage,
                gpuInfo: backendStatus.gpuInfo,
                diskSpace: backendStatus.diskSpace,
                activeOperations: backendStatus.activeOperations
            )
        } catch {
            logger.error("Failed to refresh system status: \(error)", category: .pipeline)
        }
    }
    
    public func updatePerformanceMetrics() async {
        do {
            let telemetry = try await autoResolveService.getTelemetryData()
            
            performanceMetrics = LocalPipelinePerformanceMetrics()
            performanceMetrics.averageProcessingSpeed = telemetry.averageProcessingSpeed
            performanceMetrics.totalVideosProcessed = telemetry.totalVideosProcessed
            performanceMetrics.successRate = telemetry.successRate
            performanceMetrics.memoryEfficiency = telemetry.performanceMetrics.memoryEfficiency
            performanceMetrics.lastUpdateTime = Date()
            
        } catch {
            logger.error("Failed to update performance metrics: \(error)", category: .performance)
        }
    }
    
    // MARK: - Private Implementation
    
    private func setupObservers() {
        // Observe timeline changes for auto-processing
        timeline.objectWillChange
            .debounce(for: .seconds(1), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in
                if self?.autoProcessingEnabled == true {
                    Task { [weak self] in
                        await self?.triggerAutoProcessing()
                    }
                }
            }
            .store(in: &cancellables)
        
        // Observe service connection status
        autoResolveService.$isConnected
            .sink { [weak self] isConnected in
                if isConnected {
                    Task { [weak self] in
                        await self?.refreshSystemStatus()
                    }
                }
            }
            .store(in: &cancellables)
    }
    
    private func validateInputFile(_ path: String) throws {
        let url = URL(fileURLWithPath: path)
        
        guard FileManager.default.fileExists(atPath: path) else {
            throw AutoResolveServiceError.fileNotFound("Input file not found: \(path)")
        }
        
        let asset = AVAsset(url: url)
        guard asset.isReadable else {
            throw AutoResolveServiceError.processingError("Cannot read input file: \(path)")
        }
    }
    
    private func generateCutsFromSilence(_ silenceSegments: [TimeRange]) -> [TimeRange] {
        // Convert silence segments to content segments (cuts)
        var cuts: [TimeRange] = []
        var lastEndTime: Double = 0
        
        for silence in silenceSegments.sorted { (a, b) in a.start < b.start } {
            if silence.start > lastEndTime {
                cuts.append(TimeRange(start: lastEndTime, end: silence.start))
            }
            lastEndTime = max(lastEndTime, silence.end)
        }
        
        // Add final segment if needed
        if lastEndTime < timeline.duration {
            cuts.append(TimeRange(start: lastEndTime, end: timeline.duration))
        }
        
        return cuts.filter { $0.duration > pipelineSettings.minClipDuration }
    }
    
    private func generateTimelineName(for videoPath: String) -> String {
        let filename = URL(fileURLWithPath: videoPath).deletingPathExtension().lastPathComponent
        let timestamp = DateFormatter().string(from: Date())
        return "AutoResolve_\(filename)_\(timestamp)"
    }
    
    private func scanBRollDirectory(_ directory: String) async throws -> [BRollClip] {
        let directoryURL = URL(fileURLWithPath: directory)
        
        guard FileManager.default.fileExists(atPath: directory) else {
            logger.warning("B-roll directory does not exist: \(directory)", category: .media)
            return []
        }
        
        let fileManager = FileManager.default
        let resourceKeys: [URLResourceKey] = [.isRegularFileKey, .contentTypeKey]
        
        guard let enumerator = fileManager.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: resourceKeys,
            options: [.skipsHiddenFiles]
        ) else {
            throw AutoResolveServiceError.pipelineError("Cannot enumerate B-roll directory")
        }
        
        var brollClips: [BRollClip] = []
        
        for case let fileURL as URL in enumerator {
            let resourceValues = try fileURL.resourceValues(forKeys: Set(resourceKeys))
            
            guard let isRegularFile = resourceValues.isRegularFile,
                  isRegularFile,
                  let contentType = resourceValues.contentType,
                  contentType.conforms(to: .movie) else {
                continue
            }
            
            let brollClip = try await createBRollClip(from: fileURL)
            brollClips.append(brollClip)
        }
        
        return brollClips
    }
    
    private func createBRollClip(from url: URL) async throws -> BRollClip {
        let asset = AVAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        
        return BRollClip(
            name: url.deletingPathExtension().lastPathComponent,
            url: url,
            duration: duration,
            category: "General",
            tags: [],
            dateAdded: Date()
        )
    }
    
    private func getFileSize(_ url: URL) -> Int64 {
        do {
            let resourceValues = try url.resourceValues(forKeys: [.fileSizeKey])
            return Int64(resourceValues.fileSize ?? 0)
        } catch {
            return 0
        }
    }
    
    private func createProcessingTelemetry(startTime: Date, endTime: Date) -> ProcessingTelemetry {
        let processingTime = endTime.timeIntervalSince(startTime)
        
        return ProcessingTelemetry(
            processingTime: processingTime,
            realtimeFactor: timeline.duration > 0 ? processingTime / timeline.duration : 1.0,
            memoryUsed: Double(systemStatus?.memoryUsage.used ?? 0) / 1_000_000,
            cpuUsage: 0.5 // Default CPU usage estimate
        )
    }
    
    private func updateProcessingHistory(_ result: ProcessingResult, inputPath: String) {
        let historyItem = ProcessingHistoryItem(
            date: Date(),
            status: result.success ? "Completed" : "Failed",
            processingTime: result.processingTime,
            videoName: URL(fileURLWithPath: inputPath).lastPathComponent
        )
        
        processingHistory.insert(historyItem, at: 0)
        
        // Keep only last 50 items
        if processingHistory.count > 50 {
            processingHistory = Array(processingHistory.prefix(50))
        }
    }
    
    private func triggerAutoProcessing() async {
        guard !isProcessingVideo,
              let firstClip = timeline.tracks.first?.clips.first,
              let videoPath = firstClip.sourceURL?.path else {
            return
        }
        
        logger.info("Triggering auto-processing for timeline changes", category: .pipeline)
        
        do {
            _ = try await processFullPipeline(
                inputVideoPath: videoPath,
                outputDirectory: FileManager.default.temporaryDirectory.path
            )
        } catch {
            logger.error("Auto-processing failed: \(error)", category: .pipeline)
        }
    }
}

// MARK: - Supporting Models

public struct PipelineSettings {
    // Silence Detection
    public var silenceThreshold: Double = -40.0
    public var minSilenceDuration: Double = 0.5
    public var silencePadding: Double = 0.1
    public var minClipDuration: Double = 2.0
    
    // B-roll Selection
    public var enableBRollSelection: Bool = true
    public var brollDirectory: String = ""
    public var maxBRollResults: Int = 5
    public var brollConfidenceThreshold: Double = 0.7
    public var enableVJEPA: Bool = true
    
    // Export Options
    public var createResolveProject: Bool = false
    public var outputFormat: String = "mp4"
    public var outputQuality: String = "high"
    
    public init() {}
}


public struct PipelineBRollSuggestion: Identifiable {
    public let id = UUID()
    public let cutIndex: Int
    public let timeRange: TimeRange
    public let suggestedClips: [BRollClip]
    public let confidence: Double
    public let reason: String
}

public struct LocalPipelinePerformanceMetrics {
    public var averageProcessingSpeed: Double = 0
    public var totalVideosProcessed: Int = 0
    public var successRate: Double = 1.0
    public var memoryEfficiency: Double = 1.0
    public var lastUpdateTime: Date = Date()
}

// ProcessingHistoryItem is defined in BackendTypes.swift to avoid duplication

// MARK: - Pipeline Status Extensions

extension PipelineIntegrationManager {
    
    public var isServiceConnected: Bool {
        autoResolveService.isConnected
    }
    
    public var currentOperationStatus: String {
        if isProcessingVideo {
            return processingStatus
        } else if !autoResolveService.isConnected {
            return "Backend service disconnected"
        } else {
            return "Ready"
        }
    }
    
    public var canStartProcessing: Bool {
        autoResolveService.isConnected && !isProcessingVideo
    }
    
    public func getProcessingSummary() -> String {
        guard let result = lastProcessingResult else {
            return "No processing completed"
        }
        
        if result.success {
            let rtf = result.telemetry?.realtimeFactor ?? 1.0
            return "Last: \(String(format: "%.1fx", rtf)) RTF, \(result.silenceSegments?.count ?? 0) cuts"
        } else {
            return "Last: Failed - \(result.message)"
        }
    }
}
