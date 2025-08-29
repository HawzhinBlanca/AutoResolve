import Foundation
import SwiftUI
import CoreGraphics

public enum SceneCategory: String, Codable, CaseIterable {
    case interior
    case exterior
    case nature
    case urban
    case action
    case dialogue
    case other
}

public struct SceneClassification: Codable, Sendable {
    public let label: String
    public let confidence: Double
    public let category: SceneCategory
    public init(label: String, confidence: Double, category: SceneCategory) {
        self.label = label
        self.confidence = confidence
        self.category = category
    }
}

public struct DetectedObject: Codable, Sendable {
    public let boundingBox: CGRect
    public let label: String
    public let confidence: Double
    public let objectType: ObjectType
    public init(boundingBox: CGRect, label: String, confidence: Double, objectType: ObjectType) {
        self.boundingBox = boundingBox
        self.label = label
        self.confidence = confidence
        self.objectType = objectType
    }
    enum CodingKeys: String, CodingKey {
        case boundingBox, label, confidence, objectType
    }
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        boundingBox = try container.decode(CGRect.self, forKey: .boundingBox)
        label = try container.decode(String.self, forKey: .label)
        confidence = try container.decode(Double.self, forKey: .confidence)
        objectType = try container.decode(ObjectType.self, forKey: .objectType)
    }
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(boundingBox, forKey: .boundingBox)
        try container.encode(label, forKey: .label)
        try container.encode(confidence, forKey: .confidence)
        try container.encode(objectType, forKey: .objectType)
    }
}

public enum ObjectType: String, Codable {
    case geometric
}

public struct DetectedFace: Codable, Sendable {
    public let boundingBox: CGRect
    public let confidence: Double
    public let emotion: String
    public let pose: FacePose
    public init(boundingBox: CGRect, confidence: Double, emotion: String, pose: FacePose) {
        self.boundingBox = boundingBox
        self.confidence = confidence
        self.emotion = emotion
        self.pose = pose
    }
    enum CodingKeys: String, CodingKey {
        case boundingBox, confidence, emotion, pose
    }
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        boundingBox = try container.decode(CGRect.self, forKey: .boundingBox)
        confidence = try container.decode(Double.self, forKey: .confidence)
        emotion = try container.decode(String.self, forKey: .emotion)
        pose = try container.decode(FacePose.self, forKey: .pose)
    }
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(boundingBox, forKey: .boundingBox)
        try container.encode(confidence, forKey: .confidence)
        try container.encode(emotion, forKey: .emotion)
        try container.encode(pose, forKey: .pose)
    }
}

public struct FacePose: Codable, Sendable {
    public let yaw: Double
    public let pitch: Double
    public let roll: Double
    public init(yaw: Double, pitch: Double, roll: Double) {
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
    }
}

public struct CompositionMetrics: Codable, Sendable {
    public let balance: Double
    public let symmetry: Double
    public let ruleOfThirds: Double
    public let leadingLines: Double
    public let framing: Double
    public let depth: Double
    public init(balance: Double = 0, symmetry: Double = 0, ruleOfThirds: Double = 0, leadingLines: Double = 0, framing: Double = 0, depth: Double = 0) {
        self.balance = balance
        self.symmetry = symmetry
        self.ruleOfThirds = ruleOfThirds
        self.leadingLines = leadingLines
        self.framing = framing
        self.depth = depth
    }
}

public struct QualityMetrics: Codable, Sendable {
    public let sharpness: Double
    public let noise: Double
    public let compression: Double
    public let overallQuality: Double
    public let technicalScore: Double
    public let aestheticScore: Double
    public init(sharpness: Double = 0, noise: Double = 0, compression: Double = 0, overallQuality: Double = 0, technicalScore: Double = 0, aestheticScore: Double = 0) {
        self.sharpness = sharpness
        self.noise = noise
        self.compression = compression
        self.overallQuality = overallQuality
        self.technicalScore = technicalScore
        self.aestheticScore = aestheticScore
    }
}

public struct ColorMetrics: Codable, Sendable {
    public let dominantColors: [String]
    public let colorTemperature: Double
    public let vibrance: Double
    public let saturation: Double
    public init(dominantColors: [String] = [], colorTemperature: Double = 6500, vibrance: Double = 0, saturation: Double = 0) {
        self.dominantColors = dominantColors
        self.colorTemperature = colorTemperature
        self.vibrance = vibrance
        self.saturation = saturation
    }
}

public struct MotionMetrics: Codable, Sendable {
    public let globalMotion: Double
    public let localMotion: Double
    public let direction: String
    public init(globalMotion: Double = 0, localMotion: Double = 0, direction: String = "") {
        self.globalMotion = globalMotion
        self.localMotion = localMotion
        self.direction = direction
    }
}

public struct SceneChange: Codable, Sendable {
    public let timestamp: TimeInterval
    public let fromScene: String
    public let toScene: String
    public let confidence: Double
    public let changeType: ChangeType
    public init(timestamp: TimeInterval, fromScene: String, toScene: String, confidence: Double, changeType: ChangeType) {
        self.timestamp = timestamp
        self.fromScene = fromScene
        self.toScene = toScene
        self.confidence = confidence
        self.changeType = changeType
    }
}

public enum ChangeType: String, Codable {
    case cut
}

public struct FrameAnalysis: Codable, Sendable {
    public let timestamp: TimeInterval
    public let sceneClassifications: [SceneClassification]
    public let detectedObjects: [DetectedObject]
    public let faces: [DetectedFace]
    public let compositionMetrics: CompositionMetrics
    public let qualityMetrics: QualityMetrics
    public let colorMetrics: ColorMetrics
    public let motionMetrics: MotionMetrics
    public init(timestamp: TimeInterval, sceneClassifications: [SceneClassification], detectedObjects: [DetectedObject], faces: [DetectedFace], compositionMetrics: CompositionMetrics, qualityMetrics: QualityMetrics, colorMetrics: ColorMetrics, motionMetrics: MotionMetrics) {
        self.timestamp = timestamp
        self.sceneClassifications = sceneClassifications
        self.detectedObjects = detectedObjects
        self.faces = faces
        self.compositionMetrics = compositionMetrics
        self.qualityMetrics = qualityMetrics
        self.colorMetrics = colorMetrics
        self.motionMetrics = motionMetrics
    }
}

public struct VisualAnalysis: Codable, Sendable {
    public let sceneClassifications: [SceneClassification]
    public let dominantScene: String
    public let sceneChanges: [SceneChange]
    public let averageComposition: CompositionMetrics
    public let qualityMetrics: QualityMetrics
    public let colorPalette: [String]
    public let frameAnalyses: [FrameAnalysis]
    public let visualComplexity: Double
    public init(sceneClassifications: [SceneClassification], dominantScene: String, sceneChanges: [SceneChange], averageComposition: CompositionMetrics, qualityMetrics: QualityMetrics, colorPalette: [String], frameAnalyses: [FrameAnalysis], visualComplexity: Double) {
        self.sceneClassifications = sceneClassifications
        self.dominantScene = dominantScene
        self.sceneChanges = sceneChanges
        self.averageComposition = averageComposition
        self.qualityMetrics = qualityMetrics
        self.colorPalette = colorPalette
        self.frameAnalyses = frameAnalyses
        self.visualComplexity = visualComplexity
    }
}

public struct EmotionalPoint: Codable, Sendable {
    public let timestamp: Double
    public let emotion: String
    public let intensity: Double
    public let valence: Double
    public let arousal: Double
    public init(timestamp: Double, emotion: String, intensity: Double, valence: Double, arousal: Double) {
        self.timestamp = timestamp
        self.emotion = emotion
        self.intensity = intensity
        self.valence = valence
        self.arousal = arousal
    }
}

public struct StoryStructure: Codable, Sendable {
    public let type: StoryStructureType
    public let acts: [ActBreakdown]
    public let structurePoints: [StructurePoint]
    public let overallArc: EmotionalArcType
    public init(type: StoryStructureType, acts: [ActBreakdown], structurePoints: [StructurePoint], overallArc: EmotionalArcType) {
        self.type = type
        self.acts = acts
        self.structurePoints = structurePoints
        self.overallArc = overallArc
    }
}

public enum StoryStructureType: String, Codable {
    case threeAct
}

public struct ActBreakdown: Codable, Sendable {
    public let actNumber: Int
    public let startTime: Double
    public let endTime: Double
    public let description: String
    public init(actNumber: Int, startTime: Double, endTime: Double, description: String) {
        self.actNumber = actNumber
        self.startTime = startTime
        self.endTime = endTime
        self.description = description
    }
}

public struct StructurePoint: Codable, Sendable {
    public let timestamp: Double
    public let type: StoryStructureType
    public let description: String
    public let confidence: Double
    public let emotionalValence: Double
    public init(timestamp: Double, type: StoryStructureType, description: String, confidence: Double, emotionalValence: Double) {
        self.timestamp = timestamp
        self.type = type
        self.description = description
        self.confidence = confidence
        self.emotionalValence = emotionalValence
    }
}

public enum EmotionalArcType: String, Codable {
    case positive, tragic, flat, linear, single_turn, double_turn, complex
}

public struct TensionAnalysis: Codable, Sendable {
    public let tensionPoints: [TensionPoint]
    public let averageTension: Double
    public let tensionRange: RangeDouble
    public let tensionPattern: EscalationPatternType
    public let escalationPoints: [EscalationPoint]
    public init(tensionPoints: [TensionPoint], averageTension: Double, tensionRange: RangeDouble, tensionPattern: EscalationPatternType, escalationPoints: [EscalationPoint]) {
        self.tensionPoints = tensionPoints
        self.averageTension = averageTension
        self.tensionRange = tensionRange
        self.tensionPattern = tensionPattern
        self.escalationPoints = escalationPoints
    }
}

public struct TensionPoint: Codable, Sendable {
    public let timestamp: Double
    public let intensity: Double
    public let type: TensionType
    public init(timestamp: Double, intensity: Double, type: TensionType) {
        self.timestamp = timestamp
        self.intensity = intensity
        self.type = type
    }
}

public enum TensionType: String, Codable {
    case peak, valley, plateau, rising, falling, low, moderate, high, neutral
}

public enum EscalationPatternType: String, Codable {
    case linear, exponential, oscillating, rising, falling, flat
}

public struct EscalationPoint: Codable, Sendable {
    public let timestamp: Double
    public let intensityChange: Double
    public let escalationType: EscalationType
    public init(timestamp: Double, intensityChange: Double, escalationType: EscalationType) {
        self.timestamp = timestamp
        self.intensityChange = intensityChange
        self.escalationType = escalationType
    }
}

public enum EscalationType: String, Codable {
    case gradual, sudden
}

public struct PacingAnalysis: Codable, Sendable {
    public let averagePace: String
    public let pacingChanges: [PacingChange]
    public let editingRhythm: EditingRhythm
    public let optimalCutPoints: [Double]
    public init(averagePace: String, pacingChanges: [PacingChange], editingRhythm: EditingRhythm, optimalCutPoints: [Double]) {
        self.averagePace = averagePace
        self.pacingChanges = pacingChanges
        self.editingRhythm = editingRhythm
        self.optimalCutPoints = optimalCutPoints
    }
}

public struct PacingChange: Codable, Sendable {
    public let timestamp: Double
    public let fromPace: String
    public let toPace: String
    public let changeIntensity: Double
    public let confidence: Double
    public init(timestamp: Double, fromPace: String, toPace: String, changeIntensity: Double, confidence: Double) {
        self.timestamp = timestamp
        self.fromPace = fromPace
        self.toPace = toPace
        self.changeIntensity = changeIntensity
        self.confidence = confidence
    }
}

public struct EditingRhythm: Codable, Sendable {
    public let averageEditLength: Double
    public let rhythmPattern: String
    public let paceVariability: Double
    public let editDensity: Double
    public init(averageEditLength: Double, rhythmPattern: String, paceVariability: Double, editDensity: Double) {
        self.averageEditLength = averageEditLength
        self.rhythmPattern = rhythmPattern
        self.paceVariability = paceVariability
        self.editDensity = editDensity
    }
}

public struct ThemeAnalysis: Codable, Sendable {
    public let themes: [Theme]
    public let primaryTheme: Theme
    public let themeStrength: Double
    public let motifs: [Motif]
    public let symbolism: [Symbolism]
    public let themeProgression: ThemeProgression
    public init(themes: [Theme], primaryTheme: Theme, themeStrength: Double, motifs: [Motif], symbolism: [Symbolism], themeProgression: ThemeProgression) {
        self.themes = themes
        self.primaryTheme = primaryTheme
        self.themeStrength = themeStrength
        self.motifs = motifs
        self.symbolism = symbolism
        self.themeProgression = themeProgression
    }
}

public struct NarrativeInsight: Codable, Sendable {
    public let type: InsightType
    public let title: String
    public let description: String
    public let timestamp: Double?
    public let confidence: Double
    public let actionable: Bool
    public init(type: InsightType, title: String, description: String, timestamp: Double? = nil, confidence: Double, actionable: Bool) {
        self.type = type
        self.title = title
        self.description = description
        self.timestamp = timestamp
        self.confidence = confidence
        self.actionable = actionable
    }
}

public enum InsightType: String, Codable {
    case structure, pacing, intensity, theme
}

public struct StoryBeat: Codable, Sendable {
    public let type: StoryBeatType
    public let timestamp: Double
    public let description: String
    public let importance: Double
    public let confidence: Double
    
    public var color: Color {
        switch type {
        case .introduction: return .blue
        case .risingAction: return .orange
        case .climax: return .red
        case .fallingAction: return .purple
        case .resolution: return .green
        default: return .gray
        }
    }
    
    public init(type: StoryBeatType, timestamp: Double, description: String, importance: Double, confidence: Double) {
        self.type = type
        self.timestamp = timestamp
        self.description = description
        self.importance = importance
        self.confidence = confidence
    }
}

public enum StoryBeatType: String, Codable, CaseIterable {
    case incitingIncident, plotPoint, climax, resolution, revelation, setup, development, risingAction, introduction, fallingAction
}

public struct CharacterArc: Codable, Sendable {
    public let character: String
    public let mentions: [CharacterMention]
    public let importance: Double
    public let arcType: CharacterArcType
    public let stages: [CharacterDevelopment]
    public let keyMoments: [Double]
    public init(character: String, mentions: [CharacterMention], importance: Double, arcType: CharacterArcType, stages: [CharacterDevelopment], keyMoments: [Double]) {
        self.character = character
        self.mentions = mentions
        self.importance = importance
        self.arcType = arcType
        self.stages = stages
        self.keyMoments = keyMoments
    }
}

public struct CharacterMention: Codable, Sendable {
    public let timestamp: Double
    public let character: String
    public let context: String
    public let importance: Double
    public init(timestamp: Double, character: String, context: String, importance: Double) {
        self.timestamp = timestamp
        self.character = character
        self.context = context
        self.importance = importance
    }
}

public struct CharacterDevelopment: Codable, Sendable {
    public let stage: CharacterDevelopmentStage
    public let timestamp: Double
    public let description: String
    public init(stage: CharacterDevelopmentStage, timestamp: Double, description: String) {
        self.stage = stage
        self.timestamp = timestamp
        self.description = description
    }
}

public enum CharacterDevelopmentStage: String, Codable {
    case introduction, development, conflict, growth, resolution
}

public enum CharacterArcType: String, Codable {
    case positive, tragic, single_turn, double_turn, complex, triumphOverAdversity, positiveTransformation, tragedy, heroicJourney, redemption, stable, linear
}

public struct Conflict: Codable, Sendable {
    public let type: ConflictType
    public let startTime: Double
    public let endTime: Double
    public let intensity: Double
    public let description: String
    public init(type: ConflictType, startTime: Double, endTime: Double, intensity: Double, description: String) {
        self.type = type
        self.startTime = startTime
        self.endTime = endTime
        self.intensity = intensity
        self.description = description
    }
}

public enum ConflictType: String, Codable {
    case `internal`, external, interpersonal, societal, environmental
}

public struct EmotionalTurningPoint: Codable, Sendable {
    public let timestamp: Double
    public let fromEmotion: String
    public let toEmotion: String
    public let intensity: Double
    public let turningType: EmotionalTurningType
    public init(timestamp: Double, fromEmotion: String, toEmotion: String, intensity: Double, turningType: EmotionalTurningType) {
        self.timestamp = timestamp
        self.fromEmotion = fromEmotion
        self.toEmotion = toEmotion
        self.intensity = intensity
        self.turningType = turningType
    }
}

public enum EmotionalTurningType: String, Codable {
    case peakToValley, valleyToPeak
}

public struct EmotionalRange: Codable, Sendable {
    public let valenceRange: RangeDouble
    public let arousalRange: RangeDouble
    public init(valenceRange: RangeDouble, arousalRange: RangeDouble) {
        self.valenceRange = valenceRange
        self.arousalRange = arousalRange
    }
}

public struct CutPointSuggestion: Codable, Sendable {
    public let timestamp: Double
    public let reason: String
    public let confidence: Double
    public let cutType: CutType
    public let priority: Priority
    public init(timestamp: Double, reason: String, confidence: Double, cutType: CutType, priority: Priority) {
        self.timestamp = timestamp
        self.reason = reason
        self.confidence = confidence
        self.cutType = cutType
        self.priority = priority
    }
}

public enum CutType: String, Codable {
    case structure, dramatic, pacing, rhythmic, silence, narrative
}

public struct RangeDouble: Codable {
    public let min: Double
    public let max: Double
    public init(min: Double, max: Double) {
        self.min = min
        self.max = max
    }
}

public struct ConflictAnalysis: Codable, Sendable {
    public let conflicts: [Conflict]
    public let conflictTypes: [ConflictTypeCount]
    public let resolutionRate: Double
    public let averageResolutionSatisfaction: Double
    public let escalationPattern: EscalationPattern
    public init(conflicts: [Conflict], conflictTypes: [ConflictTypeCount], resolutionRate: Double, averageResolutionSatisfaction: Double, escalationPattern: EscalationPattern) {
        self.conflicts = conflicts
        self.conflictTypes = conflictTypes
        self.resolutionRate = resolutionRate
        self.averageResolutionSatisfaction = averageResolutionSatisfaction
        self.escalationPattern = escalationPattern
    }
}

public struct ConflictTypeCount: Codable {
    public let type: ConflictType
    public let count: Int
}

public struct EscalationPattern: Codable, Sendable {
    public let patternType: EscalationPatternType
    public let averageEscalation: Double
    public let escalationPoints: [EscalationPoint]
    public init(patternType: EscalationPatternType, averageEscalation: Double, escalationPoints: [EscalationPoint]) {
        self.patternType = patternType
        self.averageEscalation = averageEscalation
        self.escalationPoints = escalationPoints
    }
}

public struct EmotionalJourney: Codable, Sendable {
    public let emotionalPoints: [EmotionalPoint]
    public let turningPoints: [EmotionalTurningPoint]
    public let arcType: EmotionalArcType
    public let overallDirection: EmotionalDirection
    public let emotionalRange: EmotionalRange
    public let highlights: [EmotionalHighlight]
    public init(emotionalPoints: [EmotionalPoint], turningPoints: [EmotionalTurningPoint], arcType: EmotionalArcType, overallDirection: EmotionalDirection, emotionalRange: EmotionalRange, highlights: [EmotionalHighlight]) {
        self.emotionalPoints = emotionalPoints
        self.turningPoints = turningPoints
        self.arcType = arcType
        self.overallDirection = overallDirection
        self.emotionalRange = emotionalRange
        self.highlights = highlights
    }
}

public enum EmotionalDirection: String, Codable {
    case rising, falling, stable, ascending, descending, cyclical
}

public struct EmotionalHighlight: Codable, Sendable {
    public let timestamp: Double
    public let emotion: String
    public let intensity: Double
    public let duration: Double
    public let context: String
    public let confidence: Double
    public init(timestamp: Double, emotion: String, intensity: Double, duration: Double, context: String, confidence: Double) {
        self.timestamp = timestamp
        self.emotion = emotion
        self.intensity = intensity
        self.duration = duration
        self.context = context
        self.confidence = confidence
    }
}

public struct EditSequence: Codable, Sendable {
    public let edits: [Transition]
    public init(edits: [Transition]) {
        self.edits = edits
    }
}

public struct Transition: Codable, Sendable {
    public let timestamp: Double
    public let type: String
    public init(timestamp: Double, type: String) {
        self.timestamp = timestamp
        self.type = type
    }
}

// BackendVideoProjectStore is defined in BackendTypes.swift

public struct Theme: Codable {
    public let name: String
    public let confidence: Double
    public let keyMoments: [Double]
    public init(name: String, confidence: Double, keyMoments: [Double]) {
        self.name = name
        self.confidence = confidence
        self.keyMoments = keyMoments
    }
}

public struct Motif: Codable {
    public let description: String
    public let occurrences: [Double]
    public init(description: String, occurrences: [Double]) {
        self.description = description
        self.occurrences = occurrences
    }
}

public struct Symbolism: Codable {
    public let symbol: String
    public let meaning: String
    public let timestamp: Double
    public init(symbol: String, meaning: String, timestamp: Double) {
        self.symbol = symbol
        self.meaning = meaning
        self.timestamp = timestamp
    }
}

public struct ThemeProgression: Codable {
    public let stages: [String]
    public init(stages: [String]) {
        self.stages = stages
    }
}

public struct RhythmAnalysisResult: Codable, Sendable {
    public let visual: VisualRhythm
    public let audio: AudioRhythm
    public let narrative: NarrativeRhythm
    public let overallTempo: Double
    public let syncScore: Double
    public let beats: [RhythmBeatData]
    public let tempo: Double
    public let timeSignature: TimeSignature
    public init(visual: VisualRhythm, audio: AudioRhythm, narrative: NarrativeRhythm, overallTempo: Double, syncScore: Double, beats: [RhythmBeatData], tempo: Double, timeSignature: TimeSignature) {
        self.visual = visual
        self.audio = audio
        self.narrative = narrative
        self.overallTempo = overallTempo
        self.syncScore = syncScore
        self.beats = beats
        self.tempo = tempo
        self.timeSignature = timeSignature
    }
}

public struct AudioRhythm: Codable, Sendable {
    public let bpm: Double
    public let beatTimestamps: [Double]
    public let tempo: String
    public let timeSignature: String
    public init(bpm: Double, beatTimestamps: [Double], tempo: String, timeSignature: String) {
        self.bpm = bpm
        self.beatTimestamps = beatTimestamps
        self.tempo = tempo
        self.timeSignature = timeSignature
    }
}

public struct VisualRhythm: Codable, Sendable {
    public let bpm: Double
    public let beatTimestamps: [Double]
    public let intensity: Double
    public let pattern: String
    public init(bpm: Double, beatTimestamps: [Double], intensity: Double, pattern: String) {
        self.bpm = bpm
        self.beatTimestamps = beatTimestamps
        self.intensity = intensity
        self.pattern = pattern
    }
}

public struct NarrativeRhythm: Codable, Sendable {
    public let pacing: String
    public let tensionCurve: [Double]
    public let climaxPoints: [Double]
    public let resolution: Double
    public let intensityCurve: [Double]
    public init(pacing: String, tensionCurve: [Double], climaxPoints: [Double], resolution: Double, intensityCurve: [Double]) {
        self.pacing = pacing
        self.tensionCurve = tensionCurve
        self.climaxPoints = climaxPoints
        self.resolution = resolution
        self.intensityCurve = intensityCurve
    }
}

public struct CutPoint: Codable, Sendable, Hashable {
    public let timestamp: Double
    public let confidence: Double
    public let reason: String
    public let cutType: CutType
    public let priority: Priority
    public init(timestamp: Double, confidence: Double, reason: String, cutType: CutType, priority: Priority) {
        self.timestamp = timestamp
        self.confidence = confidence
        self.reason = reason
        self.cutType = cutType
        self.priority = priority
    }
}

public enum Priority: Codable, Hashable, Equatable {
    case high(Double)
    case medium(Double)
    case low(Double)
    public var value: Double {
        switch self {
        case .high(let v): return v
        case .medium(let v): return v
        case .low(let v): return v
        }
    }
}

public struct AudioAnalysis: Codable, Sendable {
    public let silenceSegments: [SilenceRegion]
    public let energyLevels: [EnergyLevel]
    public let rhythmAnalysis: AudioRhythm
    public let emotionalTone: EmotionalTone?
    public init(silenceSegments: [SilenceRegion], energyLevels: [EnergyLevel], rhythmAnalysis: AudioRhythm, emotionalTone: EmotionalTone?) {
        self.silenceSegments = silenceSegments
        self.energyLevels = energyLevels
        self.rhythmAnalysis = rhythmAnalysis
        self.emotionalTone = emotionalTone
    }
}

public struct SilenceRegion: Identifiable, Codable, Sendable, Hashable {
    public let id: UUID
    public let startTime: Double
    public let endTime: Double
    public let duration: Double
    public let averageLevel: Double
    public let confidence: Double
    public var isSelected: Bool
    public init(
        _ id: UUID = UUID(),
        startTime: Double,
        endTime: Double,
        duration: Double,
        averageLevel: Double,
        confidence: Double,
        isSelected: Bool = false
    ) {
        self.id = id
        self.startTime = startTime
        self.endTime = endTime
        self.duration = duration
        self.averageLevel = averageLevel
        self.confidence = confidence
        self.isSelected = isSelected
    }
}

public struct EnergyLevel: Codable, Sendable {
    public let timestamp: Double
    public let rms: Double
    public let spectralCentroid: Double
    public init(timestamp: Double, rms: Double, spectralCentroid: Double) {
        self.timestamp = timestamp
        self.rms = rms
        self.spectralCentroid = spectralCentroid
    }
}

// EmotionType enum
public enum EmotionType: String, Codable, CaseIterable {
    case joy, sadness, anger, fear, surprise, disgust, neutral
    case excitement, melancholy, tension, relief, anticipation
}

// Additional emotion analysis types
public struct FacialEmotionPoint: Codable, Sendable {
    public let timestamp: Double
    public let emotion: String
    public let confidence: Double
    public init(timestamp: Double, emotion: String, confidence: Double) {
        self.timestamp = timestamp
        self.emotion = emotion
        self.confidence = confidence
    }
}

public struct FacialEmotionResult: Codable, Sendable {
    public let emotions: [FacialEmotionPoint]
    public let dominantEmotion: String
    public init(emotions: [FacialEmotionPoint], dominantEmotion: String) {
        self.emotions = emotions
        self.dominantEmotion = dominantEmotion
    }
}

public struct AudioEmotionResult: Codable, Sendable {
    public let emotions: [EmotionTimePoint]
    public let overallMood: String
    public init(emotions: [EmotionTimePoint], overallMood: String) {
        self.emotions = emotions
        self.overallMood = overallMood
    }
}

public struct TextSentimentResult: Codable, Sendable {
    public let sentiments: [SentimentPoint]
    public let overallSentiment: Double
    public init(sentiments: [SentimentPoint], overallSentiment: Double) {
        self.sentiments = sentiments
        self.overallSentiment = overallSentiment
    }
}

public struct SentimentPoint: Codable, Sendable {
    public let timestamp: Double
    public let sentiment: Double
    public let text: String
    public init(timestamp: Double, sentiment: Double, text: String) {
        self.timestamp = timestamp
        self.sentiment = sentiment
        self.text = text
    }
}

public struct EditingSuggestion: Codable, Sendable {
    public let timestamp: Double
    public let type: String
    public let reason: String
    public let confidence: Double
    public init(timestamp: Double, type: String, reason: String, confidence: Double) {
        self.timestamp = timestamp
        self.type = type
        self.reason = reason
        self.confidence = confidence
    }
}

// EmotionTimePoint for emotion analysis
public struct EmotionTimePoint: Codable, Sendable {
    public let timestamp: Double
    public let intensity: Double
    public let emotionType: String
    public init(timestamp: Double, intensity: Double, emotionType: String) {
        self.timestamp = timestamp
        self.intensity = intensity
        self.emotionType = emotionType
    }
}

public struct EmotionalTone: Codable, Sendable {
    public let emotionType: String
    public let valence: Double
    public let arousal: Double
    public let confidence: Double
    public init(emotionType: String, valence: Double, arousal: Double, confidence: Double) {
        self.emotionType = emotionType
        self.valence = valence
        self.arousal = arousal
        self.confidence = confidence
    }
}

public struct NarrativeAnalysis: Codable, Sendable {
    public let storyStructure: StoryStructure
    public let dramaticTension: TensionAnalysis
    public let storyBeats: [StoryBeat]
    public let characterArcs: [CharacterArc]
    public let conflicts: [Conflict]
    public let conflictAnalysis: ConflictAnalysis
    public let pacing: PacingAnalysis
    public let themes: [Theme]
    public let emotionalJourney: EmotionalJourney
    public let narrativeInsights: [NarrativeInsight]
    public let suggestedCutPoints: [CutPointSuggestion]
    public init(storyStructure: StoryStructure, dramaticTension: TensionAnalysis, storyBeats: [StoryBeat], characterArcs: [CharacterArc], conflicts: [Conflict], conflictAnalysis: ConflictAnalysis, pacing: PacingAnalysis, themes: [Theme], emotionalJourney: EmotionalJourney, narrativeInsights: [NarrativeInsight], suggestedCutPoints: [CutPointSuggestion]) {
        self.storyStructure = storyStructure
        self.dramaticTension = dramaticTension
        self.storyBeats = storyBeats
        self.characterArcs = characterArcs
        self.conflicts = conflicts
        self.conflictAnalysis = conflictAnalysis
        self.pacing = pacing
        self.themes = themes
        self.emotionalJourney = emotionalJourney
        self.narrativeInsights = narrativeInsights
        self.suggestedCutPoints = suggestedCutPoints
    }
}

public struct ConflictResolution: Codable, Sendable {
    public let resolutionRate: Double
    public let resolvedConflicts: [Conflict]
    public let unresolvedConflicts: [Conflict]
    public let resolutionStyle: ResolutionStyle
    public let satisfaction: Double
    public init(resolutionRate: Double, resolvedConflicts: [Conflict], unresolvedConflicts: [Conflict], resolutionStyle: ResolutionStyle, satisfaction: Double) {
        self.resolutionRate = resolutionRate
        self.resolvedConflicts = resolvedConflicts
        self.unresolvedConflicts = unresolvedConflicts
        self.resolutionStyle = resolutionStyle
        self.satisfaction = satisfaction
    }
}

public enum ResolutionStyle: String, Codable {
    case gradual, dramatic, moderate
}

public struct RhythmBeatData: Codable {
    public let timestamp: Double
    public let confidence: Double
    public init(timestamp: Double, confidence: Double) {
        self.timestamp = timestamp
        self.confidence = confidence
    }
}

public struct TimeSignature: Codable {
    public let numerator: Int
    public let denominator: Int
    public init(numerator: Int, denominator: Int) {
        self.numerator = numerator
        self.denominator = denominator
    }
}

// MARK: - Missing Emotion Types
public enum EmotionTransitionType: String, Codable {
    case sudden
    case gradual
    case smooth
    case abrupt
}

public enum MoodShiftType: String, Codable {
    case positive
    case negative
    case neutral
    case mixed
}

public enum SentimentType: String, Codable {
    case positive
    case negative
    case neutral
    case mixed
}

public struct EmotionTransition: Codable {
    public let fromEmotion: EmotionType
    public let toEmotion: EmotionType
    public let transitionType: EmotionTransitionType
    public let timestamp: TimeInterval
    public let confidence: Double
}

public struct AudioEmotionIntensityProfile: Codable {
    public let intensity: Double
    public let variance: Double
    public let peaks: [TimeInterval]
}

public struct AudioEmotionPoint: Codable {
    public let time: TimeInterval
    public let emotion: EmotionType
    public let intensity: Double
    public let confidence: Double
}

// Extension for missing properties
extension FacialEmotionResult {
    public var emotionPoints: [EmotionTimePoint] {
        return emotions.map { emotion in
            EmotionTimePoint(
                timestamp: emotion.timestamp,
                intensity: emotion.confidence,  // Using confidence as intensity
                emotionType: emotion.emotion
            )
        }
    }
}

extension AudioEmotionResult {
    public var emotionPoints: [AudioEmotionPoint] {
        return emotions.map { emotion in
            AudioEmotionPoint(
                time: emotion.timestamp,
                emotion: EmotionType(rawValue: emotion.emotionType) ?? .neutral,
                intensity: emotion.intensity,
                confidence: 0.8  // Default confidence since EmotionTimePoint doesn't have it
            )
        }
    }
}

extension TextSentimentResult {
    public var sentimentPoints: [EmotionTimePoint] {
        return sentiments.map { sentiment in
            EmotionTimePoint(
                timestamp: sentiment.timestamp,
                intensity: abs(sentiment.sentiment),
                emotionType: (sentiment.sentiment > 0.1 ? "positive" : (sentiment.sentiment < -0.1 ? "negative" : "neutral"))
            )
        }
    }
}

extension StoryStructure {
    public var totalDuration: TimeInterval {
        return acts.last?.endTime ?? 0  // Calculate from acts since we don't have duration property
    }
}

// Missing Analysis Result Types
public struct AutoEditResults: Codable {
    public let editDecisions: [EditDecision]
    public let timeline: TimelineStructure
    public let confidence: Double
}

public struct MediaAnalysisResult: Codable {
    public let duration: TimeInterval
    public let frameRate: Double
    public let resolution: CGSize
    public let audioChannels: Int
}

public enum EditingStyle: String, Codable {
    case fast = "fast"
    case slow = "slow"
    case dynamic = "dynamic"
}

public struct EditingConstraints: Codable {
    public let minClipDuration: TimeInterval
    public let maxClipDuration: TimeInterval
    public let targetDuration: TimeInterval?
}

public struct CutCriteria: Codable {
    public let threshold: Double
    public let minGap: TimeInterval
}

public struct HighlightReel: Codable {
    public let clips: [TimelineClip]
    public let totalDuration: TimeInterval
}

public enum HighlightStyle: String, Codable {
    case energetic = "energetic"
    case emotional = "emotional"
    case narrative = "narrative"
}

public struct TimelineStructure: Codable {
    public let totalDuration: TimeInterval
    public let clipCount: Int
    public let tracks: [TimelineTrack]
}

// Missing Performance Types
public enum CutPointPriority: String, Codable {
    case high = "high"
    case medium = "medium"
    case low = "low"
}

// Fix EmotionType enum
extension EmotionType {
    public static let contentment = EmotionType(rawValue: "contentment")!
}

// Emotional Analysis Types
public struct EmotionalIntensityProfile: Codable {
    public let average: Double
    public let peak: Double
    public let variance: Double
}

public struct EmotionalArc: Codable {
    public let points: [EmotionTimePoint]
    public let dominantEmotion: EmotionType
    public let transitions: [EmotionTransition]
}

// MARK: - Content Analysis Types
public struct ConceptAnalysisResult: Codable {
    public let concepts: [Concept]
    public let confidence: Double
}

public struct Concept: Codable {
    public let name: String
    public let relevance: Double
}

public struct TopicAnalysisResult: Codable {
    public let topics: [Topic]
    public let distribution: [Double]
}

public struct Topic: Codable {
    public let name: String
    public let keywords: [String]
    public let weight: Double
}

public struct KeywordAnalysisResult: Codable {
    public let keywords: [Keyword]
}

public struct Keyword: Codable {
    public let text: String
    public let frequency: Int
    public let relevance: Double
}

public struct SemanticAnalysisResult: Codable {
    public let entities: [Entity]
    public let relationships: [Relationship]
}

public struct Entity: Codable {
    public let name: String
    public let type: String
}

public struct Relationship: Codable {
    public let subject: String
    public let predicate: String
    public let object: String
}

public struct ContextAnalysisResult: Codable {
    public let context: String
    public let relevantSegments: [TimeRange]
}

// Audio Types
public struct AudioSyncResult: Codable {
    public let syncPoints: [SyncPoint]
    public let offset: TimeInterval
}

public struct SyncPoint: Codable {
    public let videoTime: TimeInterval
    public let audioTime: TimeInterval
}

public struct AcousticFeatures: Codable {
    public let pitch: Double
    public let energy: Double
    public let tempo: Double
}

public struct SpectrogramFrame: Codable {
    public let frequencies: [Double]
    public let magnitudes: [Double]
    public let timestamp: TimeInterval
}

public struct AudioTrack: Codable {
    public let id: UUID
    public let name: String
    public let duration: TimeInterval
    public let sampleRate: Int
    public let channels: Int
}

// Timeline Types
public struct Cut: Codable {
    public let start: TimeInterval
    public let end: TimeInterval
    public let reason: String
}

// Missing EditDecision
public struct EditDecision: Codable {
    public let timestamp: TimeInterval
    public let type: String
    public let confidence: Double
}
