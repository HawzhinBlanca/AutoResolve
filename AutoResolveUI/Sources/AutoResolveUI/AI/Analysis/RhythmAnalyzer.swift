import Foundation
import Combine
import OSLog

/// Advanced rhythm analysis system for detecting natural cut points and editing flow
/// Analyzes temporal patterns in audio, visual, and narrative elements to optimize editing rhythm
@MainActor
public class RhythmAnalyzer: ObservableObject {
    private let logger = Logger.shared
    
    // Analysis state
    @Published public var isAnalyzing = false
    @Published public var analysisProgress = 0.0
    @Published public var currentOperation = ""
    
    // Results cache
    private var analysisCache: [String: RhythmAnalysisResult] = [:]
    
    // Rhythm detection parameters
    private let windowSize: Double = 2.0 // 2-second analysis windows
    private let hopSize: Double = 0.5 // 500ms hop between windows
    private let cutPointThreshold: Double = 0.6 // Minimum confidence for cut suggestions
    
    public init() {
        logger.info("RhythmAnalyzer initialized")
    }
    
    // MARK: - Public API
    
    public func analyzeRhythm(
        audioAnalysis: AudioAnalysis,
        visualAnalysis: VisualAnalysis,
        narrativeAnalysis: NarrativeAnalysis,
        duration: TimeInterval
    ) async throws -> RhythmAnalysisResult {
        
        let cacheKey = "rhythm_\(duration)_\(Date().timeIntervalSince1970)"
        if let cached = analysisCache[cacheKey] {
            return cached
        }
        
        logger.info("Starting rhythm analysis")
        
        isAnalyzing = true
        analysisProgress = 0.0
        currentOperation = "Initializing rhythm analysis..."
        
        defer {
            isAnalyzing = false
            analysisProgress = 1.0
            currentOperation = ""
        }
        
        do {
            // Analyze audio rhythm
            currentOperation = "Analyzing audio rhythm..."
            analysisProgress = 0.2
            
            let audioRhythm = analyzeAudioRhythm(audioAnalysis)
            
            // Analyze visual rhythm
            currentOperation = "Analyzing visual rhythm..."
            analysisProgress = 0.4
            
            let visualRhythm = analyzeVisualRhythm(visualAnalysis)
            
            // Analyze narrative rhythm
            currentOperation = "Analyzing narrative rhythm..."
            analysisProgress = 0.6
            
            let narrativeRhythm = analyzeNarrativeRhythm(narrativeAnalysis)
            
            // Calculate overall tempo and sync
            currentOperation = "Computing rhythm synchronization..."
            analysisProgress = 0.8
            
            let overallTempo = calculateOverallTempo(audioRhythm, visualRhythm, narrativeRhythm)
            let syncScore = calculateSyncScore(audioRhythm, visualRhythm, narrativeRhythm)
            
            let rhythmResult = RhythmAnalysisResult(
                visual: visualRhythm,
                audio: audioRhythm,
                narrative: narrativeRhythm,
                overallTempo: overallTempo,
                syncScore: syncScore
            )
            
            // Cache results
            analysisCache[cacheKey] = rhythmResult
            
            logger.info("Rhythm analysis completed successfully")
            return rhythmResult
            
        } catch {
            logger.error("Rhythm analysis failed: \(error)")
            throw error
        }
    }
    
    public func detectCutPoints(
        audioAnalysis: AudioAnalysis,
        visualAnalysis: VisualAnalysis,
        narrativeAnalysis: NarrativeAnalysis,
        duration: TimeInterval
    ) async throws -> [CutPoint] {
        logger.info("Detecting natural cut points")
        
        var cutPoints: [CutPoint] = []
        
        // Analyze silence segments as potential cut points
        for silence in audioAnalysis.silenceSegments {
            if silence.confidence > 0.8 && silence.duration > 0.2 {
                let midPoint = (silence.startTime + silence.endTime) / 2.0
                
                cutPoints.append(CutPoint(
                    timestamp: midPoint,
                    confidence: silence.confidence,
                    reason: "Silence-based cut point",
                    cutType: .silence,
                    priority: .medium(0.5)
                ))
            }
        }
        
        // Skipping beat-based cut points (no beats data in AudioRhythm model)
        
        // Add narrative-based cut points
        for storyBeat in narrativeAnalysis.storyBeats {
            if storyBeat.confidence > 0.7 {
                cutPoints.append(CutPoint(
                    timestamp: storyBeat.timestamp,
                    confidence: storyBeat.confidence,
                    reason: "Narrative beat: \(storyBeat.type.rawValue)",
                    cutType: .narrative,
                    priority: .high(0.9)
                ))
            }
        }
        
        // Remove duplicates and sort by timestamp
        let uniqueCutPoints = Array(Set(cutPoints)).sorted { (a, b) in a.timestamp < b.timestamp }
        
        logger.info("Detected \(uniqueCutPoints.count) cut points")
        return uniqueCutPoints
    }
    
    // MARK: - Private Analysis Methods
    
    private func analyzeAudioRhythm(_ audioAnalysis: AudioAnalysis) -> AudioRhythm {
        let tempo = audioAnalysis.rhythmAnalysis.bpm
        let beatTimestamps = audioAnalysis.rhythmAnalysis.beatTimestamps
        
        return AudioRhythm(
            bpm: tempo,
            beatTimestamps: beatTimestamps,
            tempo: tempo > 140 ? "fast" : tempo > 90 ? "moderate" : "slow",
            timeSignature: audioAnalysis.rhythmAnalysis.timeSignature
        )
    }
    
    private func analyzeVisualRhythm(_ visualAnalysis: VisualAnalysis) -> VisualRhythm {
        let motionIntensity = calculateAverageMotionIntensity(visualAnalysis)
        let sceneChangeFrequency = calculateSceneChangeFrequency(visualAnalysis)
        let bpm = sceneChangeFrequency * 60.0 // Convert to beats per minute
        
        return VisualRhythm(
            bpm: bpm,
            beatTimestamps: [],
            intensity: motionIntensity,
            pattern: motionIntensity > 0.7 ? "dynamic" : motionIntensity > 0.4 ? "regular" : "static"
        )
    }
    
    private func analyzeNarrativeRhythm(_ narrativeAnalysis: NarrativeAnalysis) -> NarrativeRhythm {
        let pacingScore = calculateNarrativePacing(narrativeAnalysis)
        let tensionVariation = calculateTensionVariation(narrativeAnalysis)
        let tensionCurve = narrativeAnalysis.dramaticTension.tensionPoints.map { $0.intensity }
        let climaxPoints = narrativeAnalysis.storyBeats.filter { $0.type == StoryBeatType.climax }.map { $0.timestamp }
        
        return NarrativeRhythm(
            pacing: pacingScore > 0.7 ? "fast" : pacingScore > 0.4 ? "steady" : "slow",
            tensionCurve: tensionCurve,
            climaxPoints: climaxPoints,
            resolution: narrativeAnalysis.storyBeats.last?.timestamp ?? 0,
            intensityCurve: narrativeAnalysis.dramaticTension.tensionPoints.map { $0.intensity }
        )
    }
    
    private func calculateOverallTempo(
        _ audioRhythm: AudioRhythm,
        _ visualRhythm: VisualRhythm,
        _ narrativeRhythm: NarrativeRhythm
    ) -> Double {
        // Weighted average of different rhythm components
        let audioWeight = 0.5
        let visualWeight = 0.3
        let narrativeWeight = 0.2
        
        let narrativePacingValue = narrativeRhythm.pacing == "fast" ? 140.0 : narrativeRhythm.pacing == "steady" ? 120.0 : 80.0
        
        let weightedTempo = (audioRhythm.bpm * audioWeight) +
                           (visualRhythm.bpm * visualWeight) +
                           (narrativePacingValue * narrativeWeight)
        
        return weightedTempo
    }
    
    private func calculateSyncScore(
        _ audioRhythm: AudioRhythm,
        _ visualRhythm: VisualRhythm,
        _ narrativeRhythm: NarrativeRhythm
    ) -> Double {
        // Simple correlation calculation (stub implementation)
        let bpmDifference = abs(audioRhythm.bpm - visualRhythm.bpm) / max(audioRhythm.bpm, visualRhythm.bpm)
        let audioVisualSync = 1.0 - min(bpmDifference, 1.0)
        
        let tensionAverage = narrativeRhythm.intensityCurve.isEmpty ? 0.5 : narrativeRhythm.intensityCurve.reduce(0.0, +) / Double(narrativeRhythm.intensityCurve.count)
        let audioNarrativeSync = 1.0 - abs(visualRhythm.intensity - tensionAverage)
        
        return (audioVisualSync + audioNarrativeSync) / 2.0
    }
    
    // MARK: - Helper Methods
    
    private func calculateAverageBeatStrength(_ audioAnalysis: AudioAnalysis) -> Double {
        let rhythmAnalysis = audioAnalysis.rhythmAnalysis
        guard !rhythmAnalysis.beatTimestamps.isEmpty else {
            return 0.5
        }
        
        let totalStrength = rhythmAnalysis.beatTimestamps.reduce(0.0) { sum, _ in
            sum + 1.0
        }
        
        return totalStrength / Double(rhythmAnalysis.beatTimestamps.count)
    }
    
    private func calculateAverageMotionIntensity(_ visualAnalysis: VisualAnalysis) -> Double {
        // Stub implementation - return moderate motion intensity
        return 0.5
    }
    
    private func calculateSceneChangeFrequency(_ visualAnalysis: VisualAnalysis) -> Double {
        guard visualAnalysis.sceneChanges.count > 1 else {
            return 0.1
        }
        
        // Calculate average time between scene changes (stub implementation)
        let totalTime = 60.0
        
        let averageInterval = totalTime / Double(visualAnalysis.sceneChanges.count - 1)
        return 1.0 / averageInterval // Frequency = 1/period
    }
    
    private func calculateNarrativePacing(_ narrativeAnalysis: NarrativeAnalysis) -> Double {
        guard !narrativeAnalysis.storyBeats.isEmpty else {
            return 0.5
        }
        
        let totalConfidence = narrativeAnalysis.storyBeats.reduce(0.0) { sum, beat in
            sum + beat.confidence
        }
        
        return totalConfidence / Double(narrativeAnalysis.storyBeats.count)
    }
    
    private func calculateTensionVariation(_ narrativeAnalysis: NarrativeAnalysis) -> Double {
        let tensionPoints = narrativeAnalysis.dramaticTension.tensionPoints
        
        guard tensionPoints.count > 1 else {
            return 0.5
        }
        
        let tensionValues = tensionPoints.map { $0.intensity }
        let averageTension = tensionValues.reduce(0.0, +) / Double(tensionValues.count)
        
        let variance = tensionValues.reduce(0.0) { sum, tension in
            sum + pow(tension - averageTension, 2)
        } / Double(tensionValues.count)
        
        return sqrt(variance) // Standard deviation as measure of variation
    }
}

// MARK: - Rhythm Analysis Errors

public enum RhythmAnalysisError: Error, LocalizedError {
    case insufficientData
    case analysisTimeout
    case processingFailed
    case invalidAudioData
    case invalidVisualData
    
    public var errorDescription: String? {
        switch self {
        case .insufficientData:
            return "Insufficient data for rhythm analysis"
        case .analysisTimeout:
            return "Rhythm analysis timed out"
        case .processingFailed:
            return "Rhythm analysis processing failed"
        case .invalidAudioData:
            return "Invalid audio data provided"
        case .invalidVisualData:
            return "Invalid visual data provided"
        }
    }
}
