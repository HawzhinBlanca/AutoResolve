// AI Component Type Definitions for AutoResolve
// Missing types referenced in AIDirector and Analysis components

import Foundation
import CoreGraphics
import AVFoundation

// MARK: - Analysis Types

public struct ClipAnalysis: Codable, Sendable {
    public let clip: SimpleTimelineClip
    public let visualFeatures: VisualFeatures
    public let audioFeatures: AudioFeatures
    public let narrativeScore: Double
    public let emotionalIntensity: Double
    
    public init(clip: SimpleTimelineClip, visualFeatures: VisualFeatures, audioFeatures: AudioFeatures, narrativeScore: Double, emotionalIntensity: Double) {
        self.clip = clip
        self.visualFeatures = visualFeatures
        self.audioFeatures = audioFeatures
        self.narrativeScore = narrativeScore
        self.emotionalIntensity = emotionalIntensity
    }
}

public struct VisualFeatures: Codable, Sendable {
    public let dominantColors: [CGColor]
    public let brightness: Double
    public let contrast: Double
    public let sharpness: Double
    public let motionIntensity: Double
}

public struct AudioFeatures: Codable, Sendable {
    public let amplitude: Double
    public let frequency: Double
    public let rhythm: Double
    public let silence: Bool
}

public struct NarrativeStructure: Codable, Sendable {
    public let type: NarrativeType
    public let acts: [Act]
    public let turningPoints: [TimeInterval]
    public let climaxTime: TimeInterval?
    
    public enum NarrativeType: String, Codable {
        case linear
        case nonLinear
        case circular
        case episodic
    }
    
    public struct Act: Codable, Sendable {
        public let number: Int
        public let startTime: TimeInterval
        public let endTime: TimeInterval
        public let description: String
    }
}

public struct NarrativeInsight: Codable, Sendable, Identifiable {
    public let id = UUID()
    public let type: InsightType
    public let description: String
    public let relevantTime: TimeInterval
    public let confidence: Double
    
    public enum InsightType: String, Codable {
        case pacing
        case structure
        case emotion
        case rhythm
        case transition
    }
}

public struct ScoredSegment: Codable, Sendable {
    public let segment: TimeRange
    public let score: Double
    public let features: [String: Double]
}

public struct SelectedSegment: Codable, Sendable {
    public let segment: TimeRange
    public let score: Double
    public let reason: String
}

// MARK: - Audio Analysis Types

public struct PeakType: Codable, Sendable {
    public let type: String
    public let intensity: Double
}

public struct SpectralAnalysis: Codable, Sendable {
    public let centroid: Double
    public let rolloff: Double
    public let flux: Double
    public let entropy: Double
}

public struct SpectralFeatures: Codable, Sendable {
    public let avgCentroid: Double
    public let avgRolloff: Double
    public let avgFlux: Double
    public let avgEntropy: Double
}

public struct RhythmAnalysis: Codable, Sendable {
    public let tempo: Double
    public let beats: [BeatData]
    public let groove: GrooveAnalysis
    public let rhythmPattern: String
}

public struct GrooveAnalysis: Codable, Sendable {
    public let swing: Double
    public let syncopation: Double
    public let consistency: Double
}

public struct SpeechRecognitionResult: Codable, Sendable {
    public let transcript: String
    public let confidence: Double
    public let language: String
    public let words: [Word]
    
    public struct Word: Codable, Sendable {
        public let text: String
        public let startTime: TimeInterval
        public let endTime: TimeInterval
        public let confidence: Double
    }
}

public struct MusicAnalysisResult: Codable, Sendable {
    public let genre: String
    public let mood: String
    public let energy: Double
    public let valence: Double
}

public struct AudioQuality: Codable, Sendable {
    public let signalToNoise: Double
    public let clipping: Bool
    public let dynamicRange: Double
    public let overallQuality: QualityLevel
    
    public enum QualityLevel: String, Codable {
        case excellent
        case good
        case fair
        case poor
    }
}

// MARK: - Content Analysis Types

public struct ContentAnalysisResult: Codable, Sendable {
    public let topics: TopicAnalysisResult
    public let keywords: KeywordAnalysisResult
    public let concepts: ConceptAnalysisResult
    public let semantics: SemanticAnalysisResult
    public let structure: ContentStructure
    public let density: InformationDensity
    public let quality: ContentQuality
}

public struct ContentStructure: Codable, Sendable {
    public let type: StructureType
    public let sections: [Section]
    public let coherenceScore: Double
    
    public enum StructureType: String, Codable {
        case linear
        case hierarchical
        case network
        case mixed
    }
    
    public struct Section: Codable, Sendable {
        public let title: String
        public let startTime: TimeInterval
        public let endTime: TimeInterval
        public let importance: Double
    }
}

public struct InformationDensity: Codable, Sendable {
    public let average: Double
    public let peak: Double
    public let distribution: [Double]
}

public struct ContentQuality: Codable, Sendable {
    public let clarity: Double
    public let coherence: Double
    public let depth: Double
    public let relevance: Double
    public let overallScore: Double
}

public struct KeyMoment: Codable, Sendable, Identifiable {
    public let id = UUID()
    public let time: TimeInterval
    public let type: KeyMomentType
    public let description: String
    public let importance: Double
}

public enum KeyMomentType: String, Codable {
    case introduction
    case climax
    case resolution
    case transition
    case emphasis
    case conclusion
}

// MARK: - AI Capabilities

public struct AICapabilities: Codable, Sendable {
    public let hasSceneClassification: Bool
    public let hasObjectDetection: Bool
    public let hasFaceRecognition: Bool
    public let hasEmotionAnalysis: Bool
    public let hasActionRecognition: Bool
    public let hasSpeechRecognition: Bool
    public let hasMusicAnalysis: Bool
    public let hasContentAnalysis: Bool
    
    public init(
        hasSceneClassification: Bool = false,
        hasObjectDetection: Bool = false,
        hasFaceRecognition: Bool = false,
        hasEmotionAnalysis: Bool = false,
        hasActionRecognition: Bool = false,
        hasSpeechRecognition: Bool = false,
        hasMusicAnalysis: Bool = false,
        hasContentAnalysis: Bool = false
    ) {
        self.hasSceneClassification = hasSceneClassification
        self.hasObjectDetection = hasObjectDetection
        self.hasFaceRecognition = hasFaceRecognition
        self.hasEmotionAnalysis = hasEmotionAnalysis
        self.hasActionRecognition = hasActionRecognition
        self.hasSpeechRecognition = hasSpeechRecognition
        self.hasMusicAnalysis = hasMusicAnalysis
        self.hasContentAnalysis = hasContentAnalysis
    }
}
