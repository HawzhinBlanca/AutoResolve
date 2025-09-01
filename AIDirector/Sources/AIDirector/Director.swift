import Foundation
import AutoResolveCore

// MARK: - AI Director Main Orchestrator

public class Director {
    private let understanding: Understanding
    private let planner: Planner
    private let gates: Gates
    private let learning: Learning
    private let eventStore: EventStore
    
    public var isEnabled: Bool = true
    
    public init() {
        self.understanding = Understanding()
        self.planner = Planner()
        self.gates = Gates()
        self.learning = Learning()
        self.eventStore = EventStore()
    }
    
    // MARK: - Main Analysis Pipeline
    
    public func analyze(timeline: Timeline, mediaURL: URL) async throws -> DirectorAnalysis {
        guard isEnabled else {
            return DirectorAnalysis(suggestions: [])
        }
        
        // Phase 1: Understanding
        let silenceRanges = try await understanding.detectSilence(in: mediaURL)
        let sceneCuts = try await understanding.detectScenes(in: mediaURL)
        
        // Phase 2: Planning
        let features = extractFeatures(
            timeline: timeline,
            silence: silenceRanges,
            scenes: sceneCuts
        )
        
        let suggestions = planner.generateSuggestions(
            features: features,
            weights: learning.currentWeights
        )
        
        // Phase 3: Gates
        let validSuggestions = suggestions.filter { suggestion in
            gates.validate(suggestion, in: timeline)
        }
        
        // Phase 4: Learning feedback
        learning.recordSuggestions(validSuggestions)
        
        return DirectorAnalysis(
            suggestions: validSuggestions,
            silence: silenceRanges,
            scenes: sceneCuts,
            confidence: calculateConfidence(validSuggestions)
        )
    }
    
    // MARK: - Feature Extraction
    
    private func extractFeatures(
        timeline: Timeline,
        silence: [SilenceRange],
        scenes: [SceneCut]
    ) -> FeatureVector {
        let totalDuration = timeline.duration.seconds
        let silenceFraction = silence.reduce(0.0) { $0 + $1.duration } / totalDuration
        let cutDensity = Double(scenes.count) / totalDuration
        let avgShotLength = totalDuration / Double(max(1, scenes.count))
        
        return FeatureVector(
            silenceFraction: silenceFraction,
            cutDensity: cutDensity,
            avgShotLength: avgShotLength,
            asrConfidence: 0.0,  // Set by ASR if available
            revertRate: learning.revertRate
        )
    }
    
    private func calculateConfidence(_ suggestions: [EditSuggestion]) -> Double {
        guard !suggestions.isEmpty else { return 0.0 }
        
        let totalConfidence = suggestions.reduce(0.0) { $0 + $1.confidence }
        return totalConfidence / Double(suggestions.count)
    }
}

// MARK: - Director Types

public struct DirectorAnalysis {
    public let suggestions: [EditSuggestion]
    public let silence: [SilenceRange]?
    public let scenes: [SceneCut]?
    public let confidence: Double
    
    public init(
        suggestions: [EditSuggestion],
        silence: [SilenceRange]? = nil,
        scenes: [SceneCut]? = nil,
        confidence: Double = 0.0
    ) {
        self.suggestions = suggestions
        self.silence = silence
        self.scenes = scenes
        self.confidence = confidence
    }
}

public struct EditSuggestion: Identifiable {
    public let id = UUID()
    public let type: SuggestionType
    public let tick: Tick
    public let confidence: Double
    public let reason: String
    
    public enum SuggestionType {
        case cut
        case trim(edge: Command.Edge)
        case delete
        case transition(type: TransitionType)
    }
    
    public enum TransitionType {
        case dissolve
        case wipe
        case fade
    }
}

public struct FeatureVector {
    public let silenceFraction: Double
    public let cutDensity: Double
    public let avgShotLength: Double
    public let asrConfidence: Double
    public let revertRate: Double
}

public struct SilenceRange {
    public let start: Tick
    public let end: Tick
    
    public var duration: Double {
        (end - start).seconds
    }
}

public struct SceneCut {
    public let tick: Tick
    public let confidence: Double
}