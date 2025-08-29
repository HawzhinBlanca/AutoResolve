import Foundation
import SwiftUI
import AVFoundation
import Combine

@MainActor
public class AppState: ObservableObject {
    // Core services
    public let transport = Transport()
    public let backendClient = BackendClient()
    public let timebase = Timebase(fps: 30.0)
    public let performanceMonitor = PerformanceMonitor()
    public let persistenceManager = PersistenceManager()
    public let accessibilitySettings = AccessibilitySettings()
    
    // UI State
    @Published public var currentPage: Page = .cut
    @Published public var showImporter = false
    @Published public var showExporter = false
    @Published public var isProcessing = false
    @Published public var statusMessage = "Ready"
    
    // Timeline State
    @Published public var timeline: TimelineModel?
    @Published public var selectedClips: Set<String> = []
    @Published public var zoomLevel: Double = 1.0
    @Published public var scrollOffset: CGFloat = 0
    @Published public var snapSettings = SnapSettings()
    @Published public var currentTool: EditTool = .select
    
    // Video State
    @Published public var videoURL: URL?
    @Published public var player: AVPlayer?
    
    // AI Results
    @Published public var silenceResult: SilenceDetectionResult?
    @Published public var transcriptionResult: TranscriptionResult?
    @Published public var storyBeatsResult: [String: Any]?
    @Published public var brollResult: [BRollSelection]?
    
    // Annotation visibility
    @Published public var showSilence = true
    @Published public var showTranscription = false
    @Published public var showStoryBeats = true
    @Published public var showBRoll = true
    
    // Drag & Drop settings
    @Published public var autoProcessOnImport = false
    
    // Performance tracking
    @Published public var lastProcessingTime: Double?
    @Published public var processingProgress: Double?
    @Published public var displayLink: CADisplayLink?
    
    private var cancellables = Set<AnyCancellable>()
    
    public enum Page: String, CaseIterable {
        case cut = "Cut"
        case edit = "Edit"
        case deliver = "Deliver"
    }
    
    public enum EditTool: String, CaseIterable {
        case select = "Select"
        case blade = "Blade"
        case trim = "Trim"
        case slip = "Slip"
        case slide = "Slide"
        case ripple = "Ripple"
    }
    
    public init() {
        setupBindings()
    }
    
    private func setupBindings() {
        // Sync player with transport
        $player
            .compactMap { $0 }
            .sink { [weak self] player in
                self?.transport.setPlayer(player)
            }
            .store(in: &cancellables)
        
        // Update status from backend
        backendClient.$isConnected
            .sink { [weak self] connected in
                self?.statusMessage = connected ? "Backend Connected" : "Backend Offline"
            }
            .store(in: &cancellables)
        
        backendClient.$currentTask
            .compactMap { $0 }
            .sink { [weak self] status in
                self?.statusMessage = status.message ?? status.currentStep ?? "Processing..."
                self?.isProcessing = status.status == "processing"
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Video Import
    
    public func importVideo(url: URL) {
        // Always set as primary video (for source viewer)
        videoURL = url
        let newPlayer = AVPlayer(url: url)
        player = newPlayer
        
        // Connect transport to new player
        transport.setPlayer(newPlayer)
        
        // Add to timeline without replacing existing clips
        addVideoToTimeline(url: url, at: transport.currentTime)
        
        statusMessage = "Loaded: \(url.lastPathComponent) - Press Space to play"
    }
    
    // MARK: - Timeline Operations
    
    public func zoomIn() {
        zoomLevel = min(10.0, zoomLevel * 1.5)
    }
    
    public func zoomOut() {
        zoomLevel = max(0.1, zoomLevel / 1.5)
    }
    
    public func zoomToFit() {
        zoomLevel = 1.0
        scrollOffset = 0
    }
    
    public func activateSelectTool() {
        currentTool = .select
    }
    
    public func activateBladeTool() {
        currentTool = .blade
        // Immediately perform cut at current playhead position
        cutAtPlayhead()
    }
    
    public func cutAtPlayhead() {
        guard let timeline = timeline else { return }
        
        let cutTime = CMTimeGetSeconds(transport.currentTime)
        var clipsModified = 0
        
        // Cut all clips that intersect with the playhead
        for trackIndex in timeline.tracks.indices {
            var newClips: [SimpleTimelineClip] = []
            
            for clip in timeline.tracks[trackIndex].clips {
                if cutTime > clip.startTime && cutTime < clip.endTime {
                    // Split this clip
                    let leftClip = SimpleTimelineClip(
                        id: UUID(),
                        name: clip.name + " (A)",
                        trackIndex: clip.trackIndex,
                        startTime: clip.startTime,
                        duration: cutTime - clip.startTime,
                        sourceURL: clip.sourceURL,
                        inPoint: clip.inPoint,
                        sourceStartTime: clip.sourceStartTime,
                        type: clip.type,
                        color: clip.color
                    )
                    
                    let rightClip = SimpleTimelineClip(
                        id: UUID(),
                        name: clip.name + " (B)", 
                        trackIndex: clip.trackIndex,
                        startTime: cutTime,
                        duration: clip.endTime - cutTime,
                        sourceURL: clip.sourceURL,
                        inPoint: clip.inPoint + (cutTime - clip.startTime),
                        sourceStartTime: clip.sourceStartTime,
                        type: clip.type,
                        color: clip.color
                    )
                    
                    newClips.append(leftClip)
                    newClips.append(rightClip)
                    clipsModified += 1
                } else {
                    newClips.append(clip)
                }
            }
            
            timeline.tracks[trackIndex].clips = newClips
        }
        
        if clipsModified > 0 {
            statusMessage = "Cut \(clipsModified) clip(s) at \(timebase.timecodeFromTime(transport.currentTime))"
        } else {
            statusMessage = "No clips at playhead to cut"
        }
    }
    
    public func selectClip(_ clipId: String, addToSelection: Bool = false) {
        if addToSelection {
            if selectedClips.contains(clipId) {
                selectedClips.remove(clipId)
            } else {
                selectedClips.insert(clipId)
            }
        } else {
            selectedClips = [clipId]
        }
    }
    
    public func clearSelection() {
        selectedClips.removeAll()
    }
    
    public func addVideoToTimeline(url: URL, at time: CMTime) {
        // If no timeline exists, create one
        if timeline == nil {
            timeline = TimelineModel()
        }
        
        guard let timeline = timeline else { return }
        
        // Get actual video duration
        let asset = AVAsset(url: url)
        let duration = CMTimeGetSeconds(asset.duration)
        
        // Calculate next available position if time is occupied
        var startTime = CMTimeGetSeconds(time)
        if let videoTrack = timeline.tracks.first(where: { $0.type == .video }) {
            // Find a gap in the timeline to place the clip
            let sortedClips = videoTrack.clips.sorted { $0.startTime < $1.startTime }
            for clip in sortedClips {
                if startTime < clip.endTime && startTime + duration > clip.startTime {
                    // Overlap detected, move to end of this clip
                    startTime = clip.endTime + 0.1
                }
            }
        }
        
        // Create a new clip at the calculated position
        let clip = SimpleTimelineClip(
            id: UUID(),
            name: url.lastPathComponent,
            trackIndex: 0,
            startTime: startTime,
            duration: max(1.0, duration), // Ensure minimum 1 second duration
            sourceURL: url
        )
        
        // Find or create video track
        if let videoTrackIndex = timeline.tracks.firstIndex(where: { $0.type == .video }) {
            timeline.tracks[videoTrackIndex].addClip(clip)
        } else {
            var newTrack = UITimelineTrack(name: "V1", type: .video)
            newTrack.addClip(clip)
            timeline.tracks.append(newTrack)
        }
        
        // Update timeline duration if needed
        let newDuration = max(timeline.duration, startTime + duration)
        timeline.duration = newDuration
        
        statusMessage = "Added \(url.lastPathComponent) to timeline at \(String(format: "%.1f", startTime))s"
    }
    
    public func addVideoToSpecificTrack(url: URL, track: UITimelineTrack, at time: CMTime) {
        guard let timeline = timeline else { return }
        
        let asset = AVAsset(url: url)
        let duration = CMTimeGetSeconds(asset.duration)
        let startTime = CMTimeGetSeconds(time)
        
        let clip = SimpleTimelineClip(
            id: UUID(),
            name: url.lastPathComponent,
            trackIndex: track.type == .video ? 0 : 1,
            startTime: startTime,
            duration: max(1.0, duration),
            sourceURL: url
        )
        
        // Add to specific track
        if let trackIndex = timeline.tracks.firstIndex(where: { $0.id == track.id }) {
            timeline.tracks[trackIndex].addClip(clip)
        }
        
        statusMessage = "Added \(url.lastPathComponent) to \(track.name)"
    }
    
    public func deleteClip(_ clipId: String) {
        guard let timeline = timeline else { return }
        
        for i in timeline.tracks.indices {
            timeline.tracks[i].clips.removeAll { $0.id.uuidString == clipId }
        }
        
        selectedClips.remove(clipId)
        statusMessage = "Deleted clip"
    }
    
    public func deleteSelected() {
        guard let timeline = timeline else { return }
        
        let clipsToDelete = selectedClips
        for clipId in clipsToDelete {
            deleteClip(clipId)
        }
        
        if clipsToDelete.count > 0 {
            statusMessage = "Deleted \(clipsToDelete.count) clip(s)"
        }
    }
    
    public func rippleDeleteSelected() {
        guard let timeline = timeline, !selectedClips.isEmpty else { return }
        
        // For ripple delete, remove clips and close gaps
        var gapsToClosure: [(trackIndex: Int, startTime: Double, duration: Double)] = []
        
        // First pass: collect gap information
        for trackIndex in timeline.tracks.indices {
            let track = timeline.tracks[trackIndex]
            let selectedClipsInTrack = track.clips.filter { selectedClips.contains($0.id.uuidString) }
            
            for clip in selectedClipsInTrack {
                gapsToClosure.append((
                    trackIndex: trackIndex,
                    startTime: clip.startTime,
                    duration: clip.duration
                ))
            }
        }
        
        // Sort gaps by start time (process from end to start)
        gapsToClosure.sort { $0.startTime > $1.startTime }
        
        // Second pass: delete clips and close gaps
        for gap in gapsToClosure {
            // Remove clips from this track
            timeline.tracks[gap.trackIndex].clips.removeAll { selectedClips.contains($0.id.uuidString) && $0.startTime == gap.startTime }
            
            // Move all clips after this gap backwards by gap duration
            for clipIndex in timeline.tracks[gap.trackIndex].clips.indices {
                if timeline.tracks[gap.trackIndex].clips[clipIndex].startTime > gap.startTime {
                    timeline.tracks[gap.trackIndex].clips[clipIndex].startTime -= gap.duration
                }
            }
        }
        
        selectedClips.removeAll()
        statusMessage = "Ripple deleted \(gapsToClosure.count) clip(s)"
    }
    
    public func duplicateClip(_ clipId: String) {
        guard let timeline = timeline else { return }
        
        for i in timeline.tracks.indices {
            if let clipIndex = timeline.tracks[i].clips.firstIndex(where: { $0.id.uuidString == clipId }) {
                let originalClip = timeline.tracks[i].clips[clipIndex]
                let newClip = SimpleTimelineClip(
                    id: UUID(),
                    name: originalClip.name + " Copy",
                    trackIndex: originalClip.trackIndex,
                    startTime: originalClip.endTime + 0.1,
                    duration: originalClip.duration,
                    sourceURL: originalClip.sourceURL
                )
                timeline.tracks[i].addClip(newClip)
                statusMessage = "Duplicated \(originalClip.name)"
                break
            }
        }
    }
    
    public func showClipProperties(_ clipId: String) {
        selectedClips = [clipId]
        statusMessage = "Showing properties for clip"
    }
    
    public func processVideo(url: URL) async {
        isProcessing = true
        statusMessage = "Processing video through AI pipeline..."
        
        do {
            let result = try await backendClient.startPipeline(
                videoPath: url.path,
                options: PipelineOptions()
            )
            statusMessage = "Processing started - Task ID: \(result.taskId)"
        } catch {
            statusMessage = "Processing failed: \(error.localizedDescription)"
            isProcessing = false
        }
    }
    
    // MARK: - AI Operations
    
    public func runSilenceDetection() async {
        guard let videoPath = videoURL?.path else { return }
        
        isProcessing = true
        statusMessage = "Detecting silence..."
        
        do {
            silenceResult = try await backendClient.detectSilence(videoPath: videoPath)
            statusMessage = "Silence detection complete"
        } catch {
            statusMessage = "Silence detection failed: \(error.localizedDescription)"
        }
        
        isProcessing = false
    }
    
    public func runTranscription() async {
        guard let videoPath = videoURL?.path else { return }
        
        isProcessing = true
        statusMessage = "Transcribing..."
        
        do {
            let result = try await backendClient.transcribe(videoPath: videoPath)
            transcriptionResult = result
            statusMessage = "Transcription complete"
        } catch {
            statusMessage = "Transcription failed: \(error.localizedDescription)"
        }
        
        isProcessing = false
    }
    
    public func runStoryBeats() async {
        guard let videoPath = videoURL?.path else { return }
        
        isProcessing = true
        statusMessage = "Analyzing story beats..."
        
        do {
            let result = try await backendClient.analyzeStoryBeats(videoPath: videoPath)
            storyBeatsResult = ["beats": result.beats, "totalDuration": result.totalDuration, "confidence": result.confidence]
            statusMessage = "Story beats analysis complete"
        } catch {
            statusMessage = "Story beats failed: \(error.localizedDescription)"
        }
        
        isProcessing = false
    }
    
    public func runBRollSelection() async {
        guard let videoPath = videoURL?.path else { return }
        
        isProcessing = true
        statusMessage = "Selecting B-roll..."
        
        do {
            brollResult = try await backendClient.selectBRoll(videoPath: videoPath)
            statusMessage = "B-roll selection complete"
        } catch {
            statusMessage = "B-roll selection failed: \(error.localizedDescription)"
        }
        
        isProcessing = false
    }
    
    public func runFullPipeline() async {
        guard let videoPath = videoURL?.path else { return }
        
        isProcessing = true
        statusMessage = "Running full pipeline..."
        
        do {
            let response = try await backendClient.startPipeline(
                videoPath: videoPath,
                options: PipelineOptions()
            )
            
            // Poll for status
            var status = try await backendClient.getPipelineStatus(taskId: response.taskId)
            while status.status == "processing" {
                statusMessage = status.message ?? "Processing..."
                try await Task.sleep(for: .seconds(1))
                status = try await backendClient.getPipelineStatus(taskId: response.taskId)
            }
            
            if status.status == "completed" {
                statusMessage = "Pipeline complete"
                // Reload results
                await runSilenceDetection()
                await runTranscription()
                await runStoryBeats()
                await runBRollSelection()
            } else {
                statusMessage = "Pipeline failed: \(status.error ?? "Unknown error")"
            }
        } catch {
            statusMessage = "Pipeline failed: \(error.localizedDescription)"
        }
        
        isProcessing = false
    }
    
    // MARK: - Export Operations
    
    public func exportFCPXML() async {
        guard let timeline = timeline else { return }
        
        isProcessing = true
        statusMessage = "Exporting FCPXML..."
        
        do {
            let result = try await backendClient.exportFCPXML(timelineId: timeline.id.uuidString)
            if result.success {
                statusMessage = "FCPXML exported: \(result.outputPath ?? "artifacts/")"
            } else {
                statusMessage = "Export failed: \(result.error ?? "Unknown error")"
            }
        } catch {
            statusMessage = "Export failed: \(error.localizedDescription)"
        }
        
        isProcessing = false
    }
    
    public func exportEDL() async {
        guard let timeline = timeline else { return }
        
        isProcessing = true
        statusMessage = "Exporting EDL..."
        
        do {
            let result = try await backendClient.exportEDL(timelineId: timeline.id.uuidString)
            if result.success {
                statusMessage = "EDL exported: \(result.outputPath ?? "artifacts/")"
            } else {
                statusMessage = "Export failed: \(result.error ?? "Unknown error")"
            }
        } catch {
            statusMessage = "Export failed: \(error.localizedDescription)"
        }
        
        isProcessing = false
    }
}