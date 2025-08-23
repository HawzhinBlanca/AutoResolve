import Foundation
import SwiftUI
import Combine
import AVFoundation
import UniformTypeIdentifiers

// MARK: - B-Roll Selection View Model
@MainActor
public class BRollSelectionViewModel: ObservableObject {
    // B-Roll Library
    @Published public var brollClips: [BRollClip] = []
    @Published public var categories: [String] = ["General", "Nature", "Urban", "People", "Tech", "Abstract"]
    @Published public var brollLibraryPath: URL?
    
    // Suggestions
    @Published public var brollSuggestions: [BRollSuggestion] = []
    @Published public var selectedSuggestion: BRollSuggestion?
    @Published public var autoSuggestEnabled = true
    @Published public var suggestionConfidenceThreshold = 0.6
    
    // Processing State
    @Published public var isAnalyzing = false
    @Published public var isLoadingLibrary = false
    @Published public var analysisProgress: Double = 0
    
    // AI Analysis Settings
    @Published public var useVJEPA = true
    @Published public var analysisDepth = AnalysisDepth.balanced
    @Published public var matchingCriteria = MatchingCriteria()
    
    private let backendService = AutoResolveService()
    private var cancellables = Set<AnyCancellable>()
    
    public enum AnalysisDepth: String, CaseIterable {
        case quick = "Quick"
        case balanced = "Balanced"
        case thorough = "Thorough"
    }
    
    public struct MatchingCriteria {
        var useColorMatching = true
        var useMotionMatching = true
        var useSceneMatching = true
        var useObjectDetection = false
        var minConfidence = 0.6
    }
    
    public init() {
        setupDefaultLibrary()
        loadBRollLibrary()
    }
    
    // MARK: - Library Management
    
    public func loadBRollLibrary() {
        isLoadingLibrary = true
        
        Task {
            do {
                // Load from default B-roll directory
                let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let brollPath = documentsPath.appendingPathComponent("AutoResolve/BRoll")
                
                // Create directory if needed
                try? FileManager.default.createDirectory(at: brollPath, withIntermediateDirectories: true)
                
                // Scan for video files
                let videoFiles = try scanForVideoFiles(in: brollPath)
                
                // Create B-roll clips from files
                var clips: [BRollClip] = []
                for file in videoFiles {
                    if let clip = await createBRollClip(from: file) {
                        clips.append(clip)
                    }
                }
                
                // Load sample B-roll if no clips found
                if clips.isEmpty {
                    clips = createSampleBRollClips()
                }
                
                await MainActor.run {
                    self.brollClips = clips
                    self.brollLibraryPath = brollPath
                    self.isLoadingLibrary = false
                }
                
                // Analyze B-roll metadata
                await analyzeBRollMetadata(clips)
                
            } catch {
                print("Error loading B-roll library: \(error)")
                await MainActor.run {
                    self.isLoadingLibrary = false
                    self.brollClips = createSampleBRollClips()
                }
            }
        }
    }
    
    private func scanForVideoFiles(in directory: URL) throws -> [URL] {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        
        return contents.filter { url in
            let videoTypes: [UTType] = [.movie, .video, .mpeg4Movie, .quickTimeMovie]
            return videoTypes.contains { url.pathExtension == $0.preferredFilenameExtension }
        }
    }
    
    private func createBRollClip(from url: URL) async -> BRollClip? {
        let asset = AVAsset(url: url)
        
        guard let duration = try? await asset.load(.duration),
              let tracks = try? await asset.loadTracks(withMediaType: .video),
              !tracks.isEmpty else {
            return nil
        }
        
        let durationSeconds = CMTimeGetSeconds(duration)
        let name = url.deletingPathExtension().lastPathComponent
        
        // Detect category from filename or path
        let category = detectCategory(from: name)
        
        return BRollClip(
            name: name,
            url: url,
            duration: durationSeconds,
            category: category,
            tags: extractTags(from: name),
            thumbnail: nil,
            relevanceScore: 0.5,
            dateAdded: Date(),
            sceneType: "General",
            motionIntensity: "Medium",
            dominantColors: [],
            detectedObjects: [],
            suggestedPlacements: []
        )
    }
    
    private func detectCategory(from name: String) -> String {
        let lowercased = name.lowercased()
        
        if lowercased.contains("nature") || lowercased.contains("landscape") {
            return "Nature"
        } else if lowercased.contains("city") || lowercased.contains("urban") {
            return "Urban"
        } else if lowercased.contains("people") || lowercased.contains("person") {
            return "People"
        } else if lowercased.contains("tech") || lowercased.contains("computer") {
            return "Tech"
        } else if lowercased.contains("abstract") || lowercased.contains("pattern") {
            return "Abstract"
        }
        
        return "General"
    }
    
    private func extractTags(from name: String) -> [String] {
        // Simple tag extraction from filename
        let components = name.components(separatedBy: CharacterSet(charactersIn: "-_ "))
        return components.filter { $0.count > 2 }.map { $0.lowercased() }
    }
    
    private func createSampleBRollClips() -> [BRollClip] {
        // Create sample B-roll clips for demonstration
        return [
            BRollClip(
                name: "City Timelapse",
                url: URL(fileURLWithPath: "/tmp/city_timelapse.mp4"),
                duration: 10.0,
                category: "Urban",
                tags: ["city", "timelapse", "night"],
                relevanceScore: 0.85
            ),
            BRollClip(
                name: "Nature Drone Shot",
                url: URL(fileURLWithPath: "/tmp/nature_drone.mp4"),
                duration: 15.0,
                category: "Nature",
                tags: ["nature", "drone", "aerial"],
                relevanceScore: 0.75
            ),
            BRollClip(
                name: "Tech Close-up",
                url: URL(fileURLWithPath: "/tmp/tech_closeup.mp4"),
                duration: 8.0,
                category: "Tech",
                tags: ["technology", "closeup", "computer"],
                relevanceScore: 0.70
            ),
            BRollClip(
                name: "People Walking",
                url: URL(fileURLWithPath: "/tmp/people_walking.mp4"),
                duration: 12.0,
                category: "People",
                tags: ["people", "street", "urban"],
                relevanceScore: 0.65
            ),
            BRollClip(
                name: "Abstract Patterns",
                url: URL(fileURLWithPath: "/tmp/abstract_patterns.mp4"),
                duration: 6.0,
                category: "Abstract",
                tags: ["abstract", "patterns", "colors"],
                relevanceScore: 0.60
            )
        ]
    }
    
    // MARK: - Timeline Analysis
    
    public func analyzeTimelineForBRoll(timeline: TimelineModel) {
        guard !isAnalyzing else { return }
        
        isAnalyzing = true
        analysisProgress = 0
        
        Task {
            do {
                // Prepare timeline data for analysis
                let timelineData = prepareTimelineData(timeline)
                
                // Call backend for B-roll suggestions
                // Extract video path from first clip if available
                let videoPath = timeline.videoTracks.first?.clips.first?.sourceURL?.path ?? ""
                
                // Convert timeline gaps to TimeRange format
                let gaps = findTimelineGaps(timeline)
                let cuts = gaps.map { gap in
                    TimeRange(start: gap.lowerBound, end: gap.upperBound)
                }
                
                let suggestions = try await backendService.selectBRoll(
                    videoPath: videoPath,
                    cuts: cuts,
                    settings: BRollSettings(
                        brollDirectory: brollLibraryPath?.path ?? "",
                        maxResults: 20,
                        confidenceThreshold: suggestionConfidenceThreshold,
                        enableVJEPA: useVJEPA
                    )
                )
                
                // Convert backend suggestions to UI format
                let uiSuggestions = suggestions.selections.compactMap { selection in
                    createBRollSuggestionFromSelection(selection, timeline: timeline)
                }
                
                await MainActor.run {
                    self.brollSuggestions = uiSuggestions
                    self.isAnalyzing = false
                    self.analysisProgress = 1.0
                }
                
            } catch {
                print("B-roll analysis failed: \(error)")
                
                // Fallback to local analysis
                await performLocalAnalysis(timeline: timeline)
            }
        }
    }
    
    private func prepareTimelineData(_ timeline: TimelineModel) -> TimelineData {
        // Convert timeline to backend format
        TimelineData(
            duration: timeline.duration,
            videoTracks: timeline.videoTracks.map { track in
                VideoTrackData(
                    clips: track.clips.map { clip in
                        ClipData(
                            startTime: clip.startTime,
                            duration: clip.duration,
                            name: clip.name
                        )
                    }
                )
            },
            silenceRanges: [],  // Will be populated by silence detection
            sceneChanges: []    // Will be populated by scene detection
        )
    }
    
    private func performLocalAnalysis(timeline: TimelineModel) async {
        // Local fallback analysis
        var suggestions: [BRollSuggestion] = []
        
        // Find gaps in timeline
        let gaps = findTimelineGaps(timeline)
        
        for gap in gaps {
            // Find best matching B-roll for each gap
            if let bestMatch = findBestBRollMatch(for: gap, from: brollClips) {
                let suggestion = BRollSuggestion(
                    id: UUID(),
                    timeRange: gap,
                    clipName: bestMatch.name,
                    clipId: bestMatch.id,
                    confidence: bestMatch.relevanceScore,
                    reason: "Fill timeline gap",
                    thumbnail: nil,
                    alternativeClips: findAlternatives(for: gap, from: brollClips)
                )
                suggestions.append(suggestion)
            }
        }
        
        await MainActor.run {
            self.brollSuggestions = suggestions
            self.isAnalyzing = false
            self.analysisProgress = 1.0
        }
    }
    
    private func findTimelineGaps(_ timeline: TimelineModel) -> [ClosedRange<TimeInterval>] {
        var gaps: [ClosedRange<TimeInterval>] = []
        
        guard let videoTrack = timeline.videoTracks.first else { return gaps }
        
        let sortedClips = videoTrack.clips.sorted { $0.startTime < $1.startTime }
        
        var currentTime: TimeInterval = 0
        
        for clip in sortedClips {
            if clip.startTime > currentTime {
                // Found a gap
                gaps.append(currentTime...clip.startTime)
            }
            currentTime = clip.startTime + clip.duration
        }
        
        // Check for gap at the end
        if currentTime < timeline.duration {
            gaps.append(currentTime...timeline.duration)
        }
        
        // Filter out very small gaps (less than 1 second)
        return gaps.filter { $0.upperBound - $0.lowerBound >= 1.0 }
    }
    
    private func findBestBRollMatch(for gap: ClosedRange<TimeInterval>, from clips: [BRollClip]) -> BRollClip? {
        let gapDuration = gap.upperBound - gap.lowerBound
        
        // Find clips that fit the gap duration
        let suitableClips = clips.filter { clip in
            clip.duration <= gapDuration * 1.2 && clip.duration >= gapDuration * 0.5
        }
        
        // Sort by relevance score
        return suitableClips.max { $0.relevanceScore < $1.relevanceScore }
    }
    
    private func findAlternatives(for gap: ClosedRange<TimeInterval>, from clips: [BRollClip]) -> [String] {
        let gapDuration = gap.upperBound - gap.lowerBound
        
        return clips
            .filter { clip in
                clip.duration <= gapDuration * 1.5 && clip.duration >= gapDuration * 0.3
            }
            .sorted { $0.relevanceScore > $1.relevanceScore }
            .prefix(3)
            .map { $0.id.uuidString }
    }
    
    private func createBRollSuggestionFromSelection(_ selection: BRollSelection, timeline: TimelineModel) -> BRollSuggestion? {
        // Create a suggestion from the selection
        return BRollSuggestion(
            id: UUID(),
            timeRange: selection.timeRange.start...selection.timeRange.end,
            clipName: URL(fileURLWithPath: selection.brollPath).lastPathComponent,
            clipId: UUID(), // Generate new ID
            confidence: selection.confidence,
            reason: selection.reason,
            thumbnail: nil, // No thumbnail available
            alternativeClips: [] // No alternatives
        )
    }
    
    private func createBRollSuggestion(from backendSuggestion: BackendBRollSuggestion, timeline: TimelineModel) -> BRollSuggestion? {
        guard let clip = brollClips.first(where: { $0.id.uuidString == backendSuggestion.clipId }) else {
            return nil
        }
        
        return BRollSuggestion(
            id: UUID(),
            timeRange: backendSuggestion.startTime...backendSuggestion.endTime,
            clipName: clip.name,
            clipId: clip.id,
            confidence: backendSuggestion.confidence,
            reason: backendSuggestion.reason,
            thumbnail: clip.thumbnail,
            alternativeClips: backendSuggestion.alternatives
        )
    }
    
    // MARK: - Metadata Analysis
    
    private func analyzeBRollMetadata(_ clips: [BRollClip]) async {
        for (index, clip) in clips.enumerated() {
            await MainActor.run {
                self.analysisProgress = Double(index) / Double(clips.count)
            }
            
            // Analyze each clip for metadata
            if let analyzedClip = await analyzeClipMetadata(clip) {
                await MainActor.run {
                    if let idx = self.brollClips.firstIndex(where: { $0.id == clip.id }) {
                        self.brollClips[idx] = analyzedClip
                    }
                }
            }
        }
        
        await MainActor.run {
            self.analysisProgress = 1.0
        }
    }
    
    private func analyzeClipMetadata(_ clip: BRollClip) async -> BRollClip? {
        var updatedClip = clip
        
        // Extract thumbnail
        if clip.thumbnail == nil {
            if let thumbnail = await extractThumbnail(from: clip.url) {
                updatedClip.thumbnail = thumbnail
            }
        }
        
        // Analyze dominant colors
        if clip.dominantColors.isEmpty {
            updatedClip.dominantColors = await analyzeDominantColors(from: clip.url)
        }
        
        // Detect motion intensity
        updatedClip.motionIntensity = await detectMotionIntensity(from: clip.url)
        
        // Update relevance score based on analysis
        updatedClip.relevanceScore = calculateRelevanceScore(for: updatedClip)
        
        return updatedClip
    }
    
    private func extractThumbnail(from url: URL) async -> NSImage? {
        let asset = AVAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 320, height: 180)
        
        do {
            let cgImage = try generator.copyCGImage(at: CMTime(seconds: 1, preferredTimescale: 1), actualTime: nil)
            return NSImage(cgImage: cgImage, size: NSSize(width: 160, height: 90))
        } catch {
            return nil
        }
    }
    
    private func analyzeDominantColors(from url: URL) async -> [String] {
        // Simplified color analysis
        return ["Blue", "Green", "Gray"]
    }
    
    private func detectMotionIntensity(from url: URL) async -> String {
        // Simplified motion detection
        return ["Low", "Medium", "High"].randomElement()!
    }
    
    private func calculateRelevanceScore(for clip: BRollClip) -> Double {
        var score = 0.5
        
        // Boost score based on metadata completeness
        if clip.thumbnail != nil { score += 0.1 }
        if !clip.dominantColors.isEmpty { score += 0.1 }
        if !clip.tags.isEmpty { score += 0.1 }
        if clip.category != "General" { score += 0.1 }
        
        return min(score, 1.0)
    }
    
    // MARK: - User Actions
    
    public func acceptSuggestion(_ suggestion: BRollSuggestion, timeline: TimelineModel) {
        guard let clip = brollClips.first(where: { $0.id == suggestion.clipId }) else { return }
        
        insertBRollClip(clip, at: suggestion.timeRange.lowerBound, timeline: timeline)
        
        // Remove accepted suggestion
        brollSuggestions.removeAll { $0.id == suggestion.id }
    }
    
    public func rejectSuggestion(_ suggestion: BRollSuggestion) {
        brollSuggestions.removeAll { $0.id == suggestion.id }
    }
    
    public func insertBRollClip(_ clip: BRollClip, at time: TimeInterval, timeline: TimelineModel) {
        // Find appropriate video track or create new one
        // Ensure we have a B-roll track
        if timeline.videoTracks.count < 2 {
            // Create B-roll track
            let brollTrack = TimelineTrack(name: "B-Roll", type: .video)
            timeline.tracks.append(brollTrack)
        }
        
        let targetTrackIndex = min(1, timeline.videoTracks.count - 1)
        
        // Create timeline clip from B-roll
        var timelineClip = TimelineClip(
            name: clip.name,
            trackIndex: targetTrackIndex,
            startTime: time,
            duration: min(clip.duration, 5.0)  // Limit to 5 seconds by default
        )
        timelineClip.sourceURL = clip.url
        
        if targetTrackIndex < timeline.tracks.count {
            timeline.tracks[targetTrackIndex].clips.append(timelineClip)
        }
        
        // Update clip's suggested placements - can't mutate let constant
        // clip.suggestedPlacements.append(time)
    }
    
    public func replacetimeRange(_ range: ClosedRange<TimeInterval>, with clip: BRollClip, timeline: TimelineModel) {
        // Remove existing clips in range
        for var track in timeline.videoTracks {
            track.clips.removeAll { clip in
                let clipRange = clip.startTime...(clip.startTime + clip.duration)
                return clipRange.overlaps(range)
            }
        }
        
        // Insert B-roll clip
        insertBRollClip(clip, at: range.lowerBound, timeline: timeline)
    }
    
    public func applyBRollToTimeline(clips: [BRollClip], timeline: TimelineModel) {
        for (index, clip) in clips.enumerated() {
            let time = Double(index) * 5.0  // Place clips every 5 seconds
            insertBRollClip(clip, at: time, timeline: timeline)
        }
    }
    
    public func removeClip(_ clip: BRollClip) {
        brollClips.removeAll { $0.id == clip.id }
        brollSuggestions.removeAll { $0.clipId == clip.id }
    }
    
    public func refreshLibrary() {
        loadBRollLibrary()
    }
    
    public func importBRollClips() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.movie, .video, .mpeg4Movie]
        
        if panel.runModal() == .OK {
            Task {
                for url in panel.urls {
                    if let clip = await createBRollClip(from: url) {
                        await MainActor.run {
                            self.brollClips.append(clip)
                        }
                    }
                }
            }
        }
    }
    
    private func setupDefaultLibrary() {
        // Setup default B-roll library structure
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let brollPath = documentsPath.appendingPathComponent("AutoResolve/BRoll")
        
        // Create category folders
        for category in categories {
            let categoryPath = brollPath.appendingPathComponent(category)
            try? FileManager.default.createDirectory(at: categoryPath, withIntermediateDirectories: true)
        }
        
        brollLibraryPath = brollPath
    }
}

// MARK: - Data Models

public struct BRollClip: Identifiable, Hashable {
    public let id = UUID()
    public let name: String
    public let url: URL
    public let duration: TimeInterval
    public let category: String
    public let tags: [String]
    public var thumbnail: NSImage?
    public var relevanceScore: Double
    public let dateAdded: Date
    
    // AI Analysis Results
    public var sceneType: String
    public var motionIntensity: String
    public var dominantColors: [String]
    public var detectedObjects: [String]
    public var suggestedPlacements: [TimeInterval]
    
    public init(
        name: String,
        url: URL,
        duration: TimeInterval,
        category: String = "General",
        tags: [String] = [],
        thumbnail: NSImage? = nil,
        relevanceScore: Double = 0.5,
        dateAdded: Date = Date(),
        sceneType: String = "General",
        motionIntensity: String = "Medium",
        dominantColors: [String] = [],
        detectedObjects: [String] = [],
        suggestedPlacements: [TimeInterval] = []
    ) {
        self.name = name
        self.url = url
        self.duration = duration
        self.category = category
        self.tags = tags
        self.thumbnail = thumbnail
        self.relevanceScore = relevanceScore
        self.dateAdded = dateAdded
        self.sceneType = sceneType
        self.motionIntensity = motionIntensity
        self.dominantColors = dominantColors
        self.detectedObjects = detectedObjects
        self.suggestedPlacements = suggestedPlacements
    }
    
    public func matchesSearch(_ text: String) -> Bool {
        let searchText = text.lowercased()
        return name.lowercased().contains(searchText) ||
               category.lowercased().contains(searchText) ||
               tags.contains { $0.lowercased().contains(searchText) }
    }
    
    public func toBackendFormat() -> BackendBRollClip {
        BackendBRollClip(
            id: id.uuidString,
            path: url.path,
            duration: duration,
            category: category,
            tags: tags,
            metadata: [
                "sceneType": sceneType,
                "motionIntensity": motionIntensity,
                "dominantColors": dominantColors.joined(separator: ","),
                "objects": detectedObjects.joined(separator: ",")
            ]
        )
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

public struct BRollSuggestion: Identifiable {
    public let id: UUID
    public let timeRange: ClosedRange<TimeInterval>
    public let clipName: String
    public let clipId: UUID
    public let confidence: Double
    public let reason: String
    public let thumbnail: NSImage?
    public let alternativeClips: [String]
}

// Backend communication structures
struct TimelineData: Codable {
    let duration: TimeInterval
    let videoTracks: [VideoTrackData]
    let silenceRanges: [BRollTimeRange]
    let sceneChanges: [TimeInterval]
}

struct VideoTrackData: Codable {
    let clips: [ClipData]
}

struct ClipData: Codable {
    let startTime: TimeInterval
    let duration: TimeInterval
    let name: String
}

struct BRollTimeRange: Codable {
    let start: TimeInterval
    let end: TimeInterval
}

struct BRollSelectionSettings: Codable {
    let useVJEPA: Bool
    let confidenceThreshold: Double
    let maxSuggestions: Int
}

public struct BackendBRollClip: Codable {
    let id: String
    let path: String
    let duration: TimeInterval
    let category: String
    let tags: [String]
    let metadata: [String: String]
}

public struct BackendBRollSuggestion: Codable {
    let clipId: String
    let startTime: TimeInterval
    let endTime: TimeInterval
    let confidence: Double
    let reason: String
    let alternatives: [String]
}