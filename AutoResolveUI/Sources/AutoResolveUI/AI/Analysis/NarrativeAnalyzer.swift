import Foundation
import SwiftUI
import Combine
import OSLog

/// Advanced narrative analysis system for story structure, pacing, and dramatic arc detection
/// Analyzes content to understand narrative flow and suggest optimal editing points
@MainActor
public class NarrativeAnalyzer: ObservableObject {
    private let logger = Logger.shared
    
    // Analysis components
    private let storyStructureAnalyzer = StoryStructureAnalyzer()
    private let tensionAnalyzer = TensionAnalyzer()
    private let pacingAnalyzer = PacingAnalyzer()
    private let themeAnalyzer = ThemeAnalyzer()
    
    // Analysis state
    @Published public var isAnalyzing = false
    @Published public var analysisProgress = 0.0
    @Published public var currentOperation = ""
    
    // Results cache
    private var analysisCache: [String: NarrativeAnalysis] = [:]
    
    // Narrative models
    private let storyModels = StoryModels()
    
    public init() {
        logger.info("NarrativeAnalyzer initialized")
    }
    
    // MARK: - Public API
    
    public func analyzeNarrative(
        transcript: String,
        visualAnalysis: VisualAnalysis,
        audioAnalysis: AudioAnalysis,
        duration: TimeInterval
    ) async throws -> NarrativeAnalysis {
        
        let cacheKey = "\(transcript)_\(duration)"
        if let cached = analysisCache[cacheKey] {
            return cached
        }
        
        logger.info("Starting narrative analysis")
        
        isAnalyzing = true
        analysisProgress = 0.0
        currentOperation = "Initializing narrative analysis..."
        
        defer {
            isAnalyzing = false
            analysisProgress = 1.0
            currentOperation = ""
        }
        
        do {
            // Analyze story structure
            currentOperation = "Analyzing story structure..."
            analysisProgress = 0.2
            
            let storyStructure = try await storyStructureAnalyzer.analyze(
                transcript: transcript,
                visualAnalysis: visualAnalysis,
                duration: duration
            )
            
            // Analyze dramatic tension
            currentOperation = "Analyzing dramatic tension..."
            analysisProgress = 0.4
            
            let tensionAnalysis = try await tensionAnalyzer.analyze(
                transcript: transcript,
                visualAnalysis: visualAnalysis,
                audioAnalysis: audioAnalysis,
                duration: duration
            )
            
            // Analyze pacing
            currentOperation = "Analyzing pacing and rhythm..."
            analysisProgress = 0.6
            
            let pacingAnalysis = try await pacingAnalyzer.analyze(
                storyStructure: storyStructure,
                tensionAnalysis: tensionAnalysis,
                audioAnalysis: audioAnalysis,
                duration: duration
            )
            
            // Analyze themes
            currentOperation = "Analyzing themes and motifs..."
            analysisProgress = 0.8
            
            let themeAnalysis = try await themeAnalyzer.analyze(
                transcript: transcript,
                visualAnalysis: visualAnalysis
            )
            
            // Generate narrative insights
            currentOperation = "Generating narrative insights..."
            analysisProgress = 0.9
            
            let insights = generateNarrativeInsights(
                storyStructure: storyStructure,
                tensionAnalysis: tensionAnalysis,
                pacingAnalysis: pacingAnalysis,
                themeAnalysis: themeAnalysis
            )
            
            let narrativeAnalysis = NarrativeAnalysis(
                storyStructure: storyStructure,
                dramaticTension: tensionAnalysis,
                storyBeats: identifyStoryBeats(storyStructure, tensionAnalysis),
                characterArcs: analyzeCharacterArcs(transcript, storyStructure),
                conflicts: identifyConflicts(storyStructure, tensionAnalysis),
                conflictAnalysis: analyzeConflict(storyStructure, tensionAnalysis),
                pacing: pacingAnalysis,
                themes: themeAnalysis.themes,
                emotionalJourney: mapEmotionalJourney(tensionAnalysis, themeAnalysis, audioAnalysis),
                narrativeInsights: insights,
                suggestedCutPoints: generateCutPointSuggestions(storyStructure, tensionAnalysis, pacingAnalysis)
            )
            
            // Cache results
            analysisCache[cacheKey] = narrativeAnalysis
            
            logger.info("Narrative analysis completed successfully")
            return narrativeAnalysis
            
        } catch {
            logger.error("Narrative analysis failed: \(error)")
            throw error
        }
    }
    
    public func analyzeStoryBeats(_ transcript: String, duration: TimeInterval) async throws -> [StoryBeat] {
        logger.info("Analyzing story beats")
        
        let sentences = transcript.components(separatedBy: .punctuationCharacters)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        
        let timePerSentence = duration / Double(sentences.count)
        var storyBeats: [StoryBeat] = []
        
        for (index, sentence) in sentences.enumerated() {
            let timestamp = Double(index) * timePerSentence
            let beatType = classifyStoryBeat(sentence, position: Double(index) / Double(sentences.count))
            let importance = calculateBeatImportance(sentence, beatType: beatType)
            
            storyBeats.append(StoryBeat(
                type: beatType,
                timestamp: timestamp,
                description: sentence.trimmingCharacters(in: .whitespacesAndNewlines),
                importance: analyzeTensionLevel(sentence),
                confidence: importance * 100 // Convert to percentage
            ))
        }
        
        return storyBeats.sorted { (a, b) in a.timestamp < b.timestamp }
    }
    
    // MARK: - Private Implementation
    
    private func generateCutPointSuggestions(
        _ storyStructure: StoryStructure,
        _ tensionAnalysis: TensionAnalysis,
        _ pacingAnalysis: PacingAnalysis
    ) -> [CutPointSuggestion] {
        var suggestions: [CutPointSuggestion] = []
        
        // Add structure-based cut points
        for structurePoint in storyStructure.structurePoints {
            suggestions.append(CutPointSuggestion(
                timestamp: structurePoint.timestamp,
                reason: "Story structure transition: \(structurePoint.type.rawValue)",
                confidence: structurePoint.confidence,
                cutType: .structure,
                priority: calculateStructuralPriority(structurePoint.type)
            ))
        }
        
        // Add tension-based cut points
        for tensionPoint in tensionAnalysis.tensionPoints {
            if tensionPoint.type == .peak || tensionPoint.type == .valley {
                suggestions.append(CutPointSuggestion(
                    timestamp: tensionPoint.timestamp,
                    reason: "Dramatic tension \(tensionPoint.type.rawValue)",
                    confidence: 0.8,
                    cutType: .dramatic,
                    priority: .high( min(1.0, max(0.0, tensionPoint.intensity)) )
                ))
            }
        }
        
        // Add pacing-based cut points
        for pacingChange in pacingAnalysis.pacingChanges {
            if pacingChange.changeIntensity > 0.5 {
                suggestions.append(CutPointSuggestion(
                    timestamp: pacingChange.timestamp,
                    reason: "Pacing transition: \(pacingChange.fromPace) to \(pacingChange.toPace)",
                    confidence: pacingChange.confidence,
                    cutType: .pacing,
                    priority: .high( min(1.0, max(0.0, pacingChange.changeIntensity)) )
                ))
            }
        }
        
        // Sort by priority and confidence
        suggestions.sort {
            if $0.priority.value == $1.priority.value {
                return $0.confidence > $1.confidence
            }
            return $0.priority.value > $1.priority.value
        }
        
        return suggestions
    }
    
    private func identifyStoryBeats(_ storyStructure: StoryStructure, _ tensionAnalysis: TensionAnalysis) -> [StoryBeat] {
        var storyBeats: [StoryBeat] = []
        
        // Convert structure points to story beats
        for structurePoint in storyStructure.structurePoints {
            let beatType = structurePointToBeatType(structurePoint.type)
            let tension = findTensionAtTime(structurePoint.timestamp, in: tensionAnalysis)
            
            storyBeats.append(StoryBeat(
                type: beatType,
                timestamp: structurePoint.timestamp,
                description: structurePoint.description,
                importance: tension,
                confidence: structurePoint.confidence * 100
            ))
        }
        
        // Add tension-based beats
        for tensionPoint in tensionAnalysis.tensionPoints {
            if tensionPoint.intensity > 0.6 && tensionPoint.type != .neutral {
                let beatType = tensionPointToBeatType(tensionPoint.type)
                
                storyBeats.append(StoryBeat(
                    type: beatType,
                    timestamp: tensionPoint.timestamp,
                    description: "Tension \(tensionPoint.type.rawValue)",
                    importance: tensionPoint.intensity,
                    confidence: tensionPoint.intensity * 100
                ))
            }
        }
        
        // Remove duplicates and sort by timestamp
        storyBeats = removeDuplicateBeats(storyBeats)
        storyBeats.sort { $0.timestamp < $1.timestamp }
        
        return storyBeats
    }
    
    private func analyzeCharacterArcs(_ transcript: String, _ storyStructure: StoryStructure) -> [CharacterArc] {
        let characters = extractCharacters(from: transcript)
        var characterArcs: [CharacterArc] = []
        
        for character in characters {
            let mentions = findCharacterMentions(character, in: transcript, duration: 0)
            let arc = analyzeCharacterDevelopment(character, mentions: mentions, structure: storyStructure)
            characterArcs.append(arc)
        }
        
        return characterArcs.sorted { $0.importance > $1.importance }
    }
    
    private func analyzeConflict(_ storyStructure: StoryStructure, _ tensionAnalysis: TensionAnalysis) -> ConflictAnalysis {
        let conflicts = identifyConflicts(storyStructure, tensionAnalysis)
        
        return ConflictAnalysis(
            conflicts: conflicts,
            conflictTypes: conflicts.reduce(into: [:]) { dict, c in dict[c.type, default: 0] += 1 }.map { ConflictTypeCount(type: $0.key, count: $0.value) },
            resolutionRate: conflicts.isEmpty ? 0.0 : Double(conflicts.filter { $0.endTime > $0.startTime }.count) / Double(conflicts.count),
            averageResolutionSatisfaction: 0.7,
            escalationPattern: analyzeEscalationPattern(conflicts, tensionAnalysis)
        )
    }
    
    private func mapEmotionalJourney(
        _ tensionAnalysis: TensionAnalysis,
        _ themeAnalysis: ThemeAnalysis,
        _ audioAnalysis: AudioAnalysis
    ) -> EmotionalJourney {
        var emotionalPoints: [EmotionalPoint] = []
        
        // Map tension points to emotional points
        for tensionPoint in tensionAnalysis.tensionPoints {
            let emotion = tensionToEmotion(tensionPoint)
            emotionalPoints.append(EmotionalPoint(
                timestamp: tensionPoint.timestamp,
                emotion: emotion,
                intensity: tensionPoint.intensity,
                valence: 0.5,
                arousal: min(1.0, tensionPoint.intensity + 0.2)
            ))
        }
        
        // Enhance with audio emotional tone
        if let audioEmotionalTone = audioAnalysis.emotionalTone {
            let audioEmotion = EmotionalPoint(
                timestamp: 0.0, // Averaged across entire duration
                emotion: audioEmotionalTone.emotionType,
                intensity: audioEmotionalTone.arousal,
                valence: audioEmotionalTone.valence,
                arousal: audioEmotionalTone.arousal
            )
            emotionalPoints.append(audioEmotion)
        }
        
        // Calculate emotional arc
        let arcType = calculateEmotionalArc(emotionalPoints)
        let direction: EmotionalDirection = .stable
        return EmotionalJourney(
            emotionalPoints: emotionalPoints.sorted { (a, b) in a.timestamp < b.timestamp },
            turningPoints: [],
            arcType: arcType,
            overallDirection: direction,
            emotionalRange: calculateEmotionalRange(emotionalPoints),
            highlights: findEmotionalHighlights(emotionalPoints)
        )
    }
    
    private func generateNarrativeInsights(
        storyStructure: StoryStructure,
        tensionAnalysis: TensionAnalysis,
        pacingAnalysis: PacingAnalysis,
        themeAnalysis: ThemeAnalysis
    ) -> [NarrativeInsight] {
        var insights: [NarrativeInsight] = []
        
        // Structure insights
        if storyStructure.type != .threeAct {
            insights.append(NarrativeInsight(
                type: .structure,
                title: "Unconventional Story Structure",
                description: "This content follows a \(storyStructure.type.rawValue) structure, which may benefit from specialized editing approaches.",
                confidence: 0.8,
                actionable: true
            ))
        }
        
        // Pacing insights
        if pacingAnalysis.averagePace == "slow" {
            insights.append(NarrativeInsight(
                type: .pacing,
                title: "Slow Overall Pacing",
                description: "The average pacing is slow, which might work for dramatic scenes but could be tightened in action sequences.",
                confidence: 0.7,
                actionable: true
            ))
        }
        
        // Tension insights
        let maxIntensity = tensionAnalysis.tensionPoints.max(by: { $0.intensity < $1.intensity })?.intensity ?? 0
        let minIntensity = tensionAnalysis.tensionPoints.min(by: { $0.intensity < $1.intensity })?.intensity ?? 0
        let tensionRange = maxIntensity - minIntensity
        
        if tensionRange < 0.3 {
            insights.append(NarrativeInsight(
                type: .intensity,
                title: "Limited Tension Range",
                description: "The tension range is low (\(tensionRange)), which may make the content feel flat. Consider adding more dynamic elements.",
                confidence: 0.6,
                actionable: true
            ))
        }
        
        // Theme insights
        if themeAnalysis.themes.count > 3 {
            insights.append(NarrativeInsight(
                type: .theme,
                title: "Multiple Themes",
                description: "Multiple themes detected (\(themeAnalysis.themes.count)). Ensure the primary theme is emphasized.",
                confidence: 0.5,
                actionable: true
            ))
        }
        
        return insights.sorted(by: { $0.confidence > $1.confidence })
    }
    
    // MARK: - Helper Methods
    
    private func classifyStoryBeat(_ sentence: String, position: Double) -> StoryBeatType {
        let lowercased = sentence.lowercased()
        
        if position < 0.15 {
            return .setup
        } else if position < 0.25 {
            return .incitingIncident
        } else if position > 0.85 {
            return .resolution
        } else if position > 0.75 {
            return .climax
        } else if lowercased.contains("but") || lowercased.contains("however") || lowercased.contains("suddenly") {
            return .plotPoint
        } else if lowercased.contains("?") {
            return .revelation
        } else {
            return .development
        }
    }
    
    private func calculateBeatImportance(_ sentence: String, beatType: StoryBeatType) -> Double {
        var importance = 0.5
        
        // Adjust based on beat type
        switch beatType {
        case .setup:
            importance = 0.7
        case .incitingIncident:
            importance = 0.9
        case .plotPoint:
            importance = 0.8
        case .climax:
            importance = 1.0
        case .resolution:
            importance = 0.8
        case .revelation:
            importance = 0.9
        case .development:
            importance = 0.5
        case .introduction:
            importance = 0.5
        case .risingAction:
            importance = 0.7
        }
        
        // Adjust based on content
        let lowercased = sentence.lowercased()
        if lowercased.contains("important") || lowercased.contains("crucial") || lowercased.contains("critical") {
            importance += 0.1
        }
        
        if sentence.contains("!") {
            importance += 0.05
        }
        
        return min(1.0, importance)
    }
    
    private func analyzeEmotionalValence(_ sentence: String) -> Double {
        let positiveWords = ["happy", "joy", "excited", "wonderful", "amazing", "great", "good", "love", "hope", "success"]
        let negativeWords = ["sad", "angry", "terrible", "awful", "hate", "fear", "failure", "disaster", "problem", "crisis"]
        
        let lowercased = sentence.lowercased()
        let positiveCount = positiveWords.filter { lowercased.contains($0) }.count
        let negativeCount = negativeWords.filter { lowercased.contains($0) }.count
        
        if positiveCount > negativeCount {
            return 0.7
        } else if negativeCount > positiveCount {
            return 0.3
        } else {
            return 0.5
        }
    }
    
    private func analyzeTensionLevel(_ sentence: String) -> Double {
        let tensionWords = ["conflict", "problem", "crisis", "urgent", "emergency", "critical", "danger", "threat", "challenge"]
        let calmWords = ["peaceful", "calm", "quiet", "gentle", "soft", "relaxed", "comfortable"]
        
        let lowercased = sentence.lowercased()
        let tensionCount = tensionWords.filter { lowercased.contains($0) }.count
        let calmCount = calmWords.filter { lowercased.contains($0) }.count
        
        if tensionCount > 0 {
            return min(1.0, 0.5 + Double(tensionCount) * 0.2)
        } else if calmCount > 0 {
            return max(0.0, 0.5 - Double(calmCount) * 0.1)
        } else {
            return 0.5
        }
    }
    
    private func calculatePacingWeight(_ sentence: String, beatType: StoryBeatType) -> Double {
        var weight = 0.5
        
        // Adjust based on beat type
        switch beatType {
        case .incitingIncident, .climax:
            weight = 0.9
        case .plotPoint, .revelation:
            weight = 0.8
        case .setup, .resolution:
            weight = 0.6
        case .development:
            weight = 0.4
        case .introduction:
            weight = 0.5
        case .risingAction:
            weight = 0.7
        }
        
        // Adjust based on sentence length (longer sentences = slower pacing)
        let wordCount = sentence.components(separatedBy: .whitespacesAndNewlines).count
        if wordCount > 20 {
            weight -= 0.1
        } else if wordCount < 5 {
            weight += 0.1
        }
        
        return max(0.0, min(1.0, weight))
    }
    
    private func calculateStructuralPriority(_ structureType: StoryStructureType) -> Priority {
        switch structureType {
        case .threeAct:
            return .high(0.9)
        }
    }
    
    private func structurePointToBeatType(_ structureType: StoryStructureType) -> StoryBeatType {
        switch structureType {
        case .threeAct:
            return .plotPoint
        }
    }
    
    private func tensionPointToBeatType(_ tensionType: TensionType) -> StoryBeatType {
        switch tensionType {
        case .peak:
            return .climax
        case .valley:
            return .resolution
        case .low:
            return .risingAction
        case .moderate:
            return .risingAction
        case .high:
            return .climax
        case .rising:
            return .development
        case .falling:
            return .resolution
        case .neutral:
            return .development
        case .plateau:
            return .development
        }
    }
    
    private func calculatePacingWeightForStructure(_ structureType: StoryStructureType) -> Double {
        switch structureType {
        case .threeAct:
            return 0.8
        }
    }
    
    private func findTensionAtTime(_ timestamp: Double, in tensionAnalysis: TensionAnalysis) -> Double {
        let closestPoint = tensionAnalysis.tensionPoints.min { point1, point2 in
            abs(point1.timestamp - timestamp) < abs(point2.timestamp - timestamp)
        }
        return closestPoint?.intensity ?? 0.5
    }
    
    private func removeDuplicateBeats(_ beats: [StoryBeat]) -> [StoryBeat] {
        var uniqueBeats: [StoryBeat] = []
        let timeThreshold = 2.0 // 2 seconds
        
        for beat in beats {
            let isDuplicate = uniqueBeats.contains(where: { existingBeat in
                let timeDiff = abs(Double(existingBeat.timestamp) - Double(beat.timestamp))
                let sameType = existingBeat.type == beat.type
                return timeDiff < timeThreshold && sameType
            })
            
            if !isDuplicate {
                uniqueBeats.append(beat)
            }
        }
        
        return uniqueBeats
    }
    
    private func extractCharacters(from transcript: String) -> [String] {
        // Simple character extraction - looks for proper nouns
        let words = transcript.components(separatedBy: .whitespacesAndNewlines)
        let properNouns = words.filter { word in
            let firstLetter = word.first
            return firstLetter?.isUppercase == true && word.count > 1
        }
        
        // Remove common non-character words
        let excludeWords = ["The", "This", "That", "And", "But", "Or", "So", "Now", "Then", "Here", "There"]
        let characters = properNouns.filter { !excludeWords.contains($0) }
        
        // Return unique characters
        return Array(Set(characters)).prefix(5).map { String($0) }
    }
    
    private func findCharacterMentions(_ character: String, in transcript: String, duration: Double) -> [CharacterMention] {
        var mentions: [CharacterMention] = []
        let sentences = transcript.components(separatedBy: .punctuationCharacters)
        let timePerSentence = duration / Double(sentences.count)
        
        for (index, sentence) in sentences.enumerated() {
            if sentence.contains(character) {
                let timestamp = Double(index) * timePerSentence
                let context = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                
                mentions.append(CharacterMention(
                    timestamp: timestamp,
                    character: character,
                    context: context,
                    importance: calculateMentionImportance(sentence, character: character)
                ))
            }
        }
        
        return mentions
    }
    
    private func calculateMentionImportance(_ sentence: String, character: String) -> Double {
        let importanceWords = ["said", "did", "went", "came", "saw", "found", "discovered", "realized", "decided"]
        let lowercased = sentence.lowercased()
        let importanceCount = importanceWords.filter { lowercased.contains($0) }.count
        
        return min(1.0, 0.3 + Double(importanceCount) * 0.2)
    }
    
    private func analyzeCharacterDevelopment(_ character: String, mentions: [CharacterMention], structure: StoryStructure) -> CharacterArc {
        let totalMentions = mentions.count
        let averageImportance = mentions.isEmpty ? 0.0 : (mentions.reduce(0.0) { $0 + $1.importance }) / Double(totalMentions)
        
        let developmentStages = classifyDevelopmentStages(mentions, totalDuration: structure.totalDuration)
        
        return CharacterArc(
            character: character,
            mentions: mentions,
            importance: averageImportance,
            arcType: classifyArcType(developmentStages),
            stages: developmentStages.map { CharacterDevelopment(stage: $0, timestamp: 0, description: "") },
            keyMoments: findKeyCharacterMoments(mentions).map { $0.timestamp }
        )
    }
    
    private func classifyDevelopmentStages(_ mentions: [CharacterMention], totalDuration: Double) -> [CharacterDevelopmentStage] {
        let stages: [CharacterDevelopmentStage] = [
            .introduction, .development, .conflict, .growth, .resolution
        ]
        
        var developmentStages: [CharacterDevelopmentStage] = []
        let stageSize = totalDuration / Double(stages.count)
        
        for (index, stage) in stages.enumerated() {
            let stageStart = Double(index) * stageSize
            let stageEnd = Double(index + 1) * stageSize
            
            let stageMentions = mentions.filter { $0.timestamp >= stageStart && $0.timestamp < stageEnd }
            if !stageMentions.isEmpty {
                developmentStages.append(stage)
            }
        }
        
        return developmentStages
    }
    
    private func classifyArcType(_ stages: [CharacterDevelopmentStage]) -> CharacterArcType {
        if stages.contains(.growth) && stages.contains(.resolution) {
            return .positive
        } else if stages.contains(.conflict) && !stages.contains(.resolution) {
            return .tragic
        } else if stages.count <= 2 {
            return .flat
        } else {
            return .complex
        }
    }
    
    private func findKeyCharacterMoments(_ mentions: [CharacterMention]) -> [CharacterMention] {
        return mentions.sorted(by: { $0.importance > $1.importance }).prefix(3).map { $0 }
    }
    
    private func identifyConflicts(_ storyStructure: StoryStructure, _ tensionAnalysis: TensionAnalysis) -> [Conflict] {
        var conflicts: [Conflict] = []
        
        // Identify conflicts from tension peaks
        let highTensionPoints = tensionAnalysis.tensionPoints.filter { $0.intensity > 0.6 }
        
        for tensionPoint in highTensionPoints {
            let conflictType = classifyConflictFromTension(tensionPoint)
            
            conflicts.append(Conflict(
                type: conflictType,
                startTime: tensionPoint.timestamp,
                endTime: findConflictResolution(tensionPoint, in: tensionAnalysis),
                intensity: tensionPoint.intensity,
                description: "Conflict at \(formatTime(tensionPoint.timestamp))"
            ))
        }
        
        return mergeSimilarConflicts(conflicts)
    }
    
    private func classifyConflictFromTension(_ tensionPoint: TensionPoint) -> ConflictType {
        if tensionPoint.intensity < 0.3 {
            return .internal
        } else if tensionPoint.intensity > 0.8 {
            return .external
        } else {
            return .interpersonal
        }
    }
    
    private func findConflictResolution(_ conflictPoint: TensionPoint, in tensionAnalysis: TensionAnalysis) -> Double {
        // Find the next valley after this peak
        let subsequentPoints = tensionAnalysis.tensionPoints.filter { $0.timestamp > conflictPoint.timestamp }
        
        if let resolution = subsequentPoints.first(where: { $0.type == .valley || $0.intensity < 0.4 }) {
            return resolution.timestamp
        }
        
        return conflictPoint.timestamp + 30.0 // Default 30 seconds later
    }
    
    private func mergeSimilarConflicts(_ conflicts: [Conflict]) -> [Conflict] {
        var mergedConflicts: [Conflict] = []
        
        for conflict in conflicts {
            let similarConflict = mergedConflicts.first { existing in
                abs(Double(existing.startTime) - Double(conflict.startTime)) < 10.0 && existing.type == conflict.type
            }
            
            if similarConflict == nil {
                mergedConflicts.append(conflict)
            }
        }
        
        return mergedConflicts
    }
    
    private func classifyConflictTypes(_ conflicts: [Conflict]) -> [ConflictType: Int] {
        var types: [ConflictType: Int] = [:]
        
        for conflict in conflicts {
            types[conflict.type, default: 0] += 1
        }
        
        return types
    }
    
    private func analyzeConflictResolution(_ conflicts: [Conflict], _ storyStructure: StoryStructure) -> ConflictResolution {
        let resolvedConflicts = conflicts.filter { $0.endTime > $0.startTime }
        let unresolvedConflicts = conflicts.filter { $0.endTime <= $0.startTime }
        
        let resolutionRate = conflicts.isEmpty ? 0.0 : Double(resolvedConflicts.count) / Double(conflicts.count)
        
        return ConflictResolution(
            resolutionRate: resolutionRate,
            resolvedConflicts: resolvedConflicts,
            unresolvedConflicts: unresolvedConflicts,
            resolutionStyle: classifyResolutionStyle(resolvedConflicts),
            satisfaction: calculateResolutionSatisfaction(resolvedConflicts)
        )
    }
    
    private func classifyResolutionStyle(_ resolvedConflicts: [Conflict]) -> ResolutionStyle {
        if resolvedConflicts.allSatisfy({ $0.intensity < 0.5 }) {
            return .gradual
        } else if resolvedConflicts.contains(where: { $0.intensity > 0.8 }) {
            return .dramatic
        } else {
            return .moderate
        }
    }
    
    private func calculateResolutionSatisfaction(_ resolvedConflicts: [Conflict]) -> Double {
        guard !resolvedConflicts.isEmpty else { return 0.0 }
        
        let avgIntensity = resolvedConflicts.reduce(0.0) { $0 + $1.intensity } / Double(resolvedConflicts.count)
        let avgDuration = resolvedConflicts.reduce(0.0) { $0 + ($1.endTime - $1.startTime) } / Double(resolvedConflicts.count)
        
        // Higher satisfaction for moderate intensity and appropriate duration
        let intensityScore = 1.0 - abs(avgIntensity - 0.7) // Optimal around 0.7
        let durationScore = min(1.0, avgDuration / 60.0) // Better if at least 1 minute
        
        return (intensityScore + durationScore) / 2.0
    }
    
    private func analyzeEscalationPattern(_ conflicts: [Conflict], _ tensionAnalysis: TensionAnalysis) -> EscalationPattern {
        let sortedConflicts = conflicts.sorted { (a, b) in a.startTime < b.startTime }
        
        if sortedConflicts.count < 2 {
            return EscalationPattern(patternType: .flat, averageEscalation: 0.0, escalationPoints: [])
        }
        
        var escalationPoints: [EscalationPoint] = []
        var totalEscalation = 0.0
        
        for i in 1..<sortedConflicts.count {
            let prev = sortedConflicts[i-1]
            let current = sortedConflicts[i]
            let escalation = current.intensity - prev.intensity
            
            if abs(escalation) > 0.1 {
                escalationPoints.append(EscalationPoint(
                    timestamp: current.startTime,
                    intensityChange: escalation,
                    escalationType: escalation > 0 ? .gradual : .sudden
                ))
            }
            
            totalEscalation += escalation
        }
        
        let avgEscalation = totalEscalation / Double(sortedConflicts.count - 1)
        let pattern: EscalationPatternType
        
        if avgEscalation > 0.1 {
            pattern = .rising
        } else if avgEscalation < -0.1 {
            pattern = .falling
        } else {
            pattern = .flat
        }
        
        return EscalationPattern(patternType: pattern, averageEscalation: avgEscalation, escalationPoints: escalationPoints)
    }
    
    private func tensionToEmotion(_ tensionPoint: TensionPoint) -> String {
        switch tensionPoint.type {
        case .peak:
            return tensionPoint.intensity > 0.5 ? "excitement" : "anxiety"
        case .valley:
            return tensionPoint.intensity > 0.5 ? "calm" : "despair"
        case .rising:
            return "anticipation"
        case .falling:
            return "relief"
        case .low:
            return "boredom"
        case .moderate:
            return "interest"
        case .high:
            return "tension"
        case .neutral:
            return "neutral"
        case .plateau:
            return "stable"
        }
    }
    
    private func calculateEmotionalArc(_ emotionalPoints: [EmotionalPoint]) -> EmotionalArcType {
        guard emotionalPoints.count > 1 else {
            return .flat
        }
        
        let sortedPoints = emotionalPoints.sorted { (a, b) in a.timestamp < b.timestamp }
        let startValence = sortedPoints.first!.valence
        let endValence = sortedPoints.last!.valence
        
        let turningPoints = findEmotionalTurningPoints(sortedPoints)
        let direction = classifyEmotionalDirection(startValence, endValence, turningPoints)
        let arcType = classifyEmotionalArcType(turningPoints, direction)
        
        return arcType
    }
    
    private func findEmotionalTurningPoints(_ sortedPoints: [EmotionalPoint]) -> [EmotionalTurningPoint] {
        var turningPoints: [EmotionalTurningPoint] = []
        
        for i in 1..<(sortedPoints.count - 1) {
            let prev = sortedPoints[i-1]
            let current = sortedPoints[i]
            let next = sortedPoints[i+1]
            
            let prevChange = current.valence - prev.valence
            let nextChange = next.valence - current.valence
            
            // Check for direction change
            if (prevChange > 0 && nextChange < 0) || (prevChange < 0 && nextChange > 0) {
                turningPoints.append(EmotionalTurningPoint(
                    timestamp: current.timestamp,
                    fromEmotion: prev.emotion,
                    toEmotion: next.emotion,
                    intensity: abs(prevChange) + abs(nextChange),
                    turningType: prevChange > 0 ? .peakToValley : .valleyToPeak
                ))
            }
        }
        
        return turningPoints
    }
    
    private func classifyEmotionalDirection(_ startValence: Double, _ endValence: Double, _ turningPoints: [EmotionalTurningPoint]) -> EmotionalDirection {
        let overallChange = endValence - startValence
        
        if abs(overallChange) < 0.1 {
            return .stable
        } else if overallChange > 0.3 {
            return .ascending
        } else if overallChange < -0.3 {
            return .descending
        } else if turningPoints.count > 2 {
            return .cyclical
        } else {
            return .stable
        }
    }
    
    private func classifyEmotionalArcType(_ turningPoints: [EmotionalTurningPoint], _ direction: EmotionalDirection) -> EmotionalArcType {
        if turningPoints.isEmpty {
            return direction == .stable ? .flat : .linear
        } else if turningPoints.count == 1 {
            return .single_turn
        } else if turningPoints.count >= 3 {
            return .complex
        } else {
            return .double_turn
        }
    }
    
    private func findDominantEmotion(_ emotionalPoints: [EmotionalPoint]) -> String {
        var emotionFrequency: [String: Double] = [:]
        
        for point in emotionalPoints {
            emotionFrequency[point.emotion, default: 0.0] += point.intensity
        }
        
        return emotionFrequency.max { $0.value < $1.value }?.key ?? "neutral"
    }
    
    private func calculateEmotionalRange(_ emotionalPoints: [EmotionalPoint]) -> EmotionalRange {
        let valences = emotionalPoints.map { $0.valence }
        let arousals = emotionalPoints.map { $0.arousal }
        
        return EmotionalRange(
            valenceRange: RangeDouble(min: valences.min() ?? 0.0, max: valences.max() ?? 1.0),
            arousalRange: RangeDouble(min: arousals.min() ?? 0.0, max: arousals.max() ?? 1.0),
            emotionalBreadth: calculateEmotionalBreadth(emotionalPoints),
            intensityVariation: calculateIntensityVariation(emotionalPoints)
        )
    }
    
    private func calculateEmotionalBreadth(_ emotionalPoints: [EmotionalPoint]) -> Double {
        let uniqueEmotions = Set(emotionalPoints.map { $0.emotion })
        return min(1.0, Double(uniqueEmotions.count) / 10.0) // Normalized to 10 possible emotions
    }
    
    private func calculateIntensityVariation(_ emotionalPoints: [EmotionalPoint]) -> Double {
        let intensities = emotionalPoints.map { $0.intensity }
        guard intensities.count > 1 else { return 0.0 }
        
        let mean = intensities.reduce(0, +) / Double(intensities.count)
        let variance = intensities.reduce(0) { $0 + pow($1 - mean, 2) } / Double(intensities.count)
        
        return sqrt(variance)
    }
    
    private func findEmotionalClimax(_ emotionalPoints: [EmotionalPoint]) -> [EmotionalPoint] {
        let highIntensityThreshold = 0.7
        return emotionalPoints.filter { $0.intensity > highIntensityThreshold }
                              .sorted { $0.intensity > $1.intensity }
                              .prefix(3)
                              .map { $0 }
    }
    
    private func findEmotionalResolution(_ emotionalPoints: [EmotionalPoint]) -> [EmotionalPoint] {
        // Find points in the last 20% of timeline with moderate to low intensity
        guard !emotionalPoints.isEmpty else { return [] }
        
        let sortedPoints = emotionalPoints.sorted { (a, b) in a.timestamp < b.timestamp }
        let totalDuration = sortedPoints.last!.timestamp
        let resolutionThreshold = totalDuration * 0.8
        
        return sortedPoints.filter { $0.timestamp > resolutionThreshold && $0.intensity < 0.6 }
    }
    
    private func findEmotionalHighlights(_ emotionalPoints: [EmotionalPoint]) -> [EmotionalHighlight] {
        let climaxPoints = findEmotionalClimax(emotionalPoints)
        let resolutionPoints = findEmotionalResolution(emotionalPoints)
        return (climaxPoints + resolutionPoints).map { point in
            EmotionalHighlight(
                timestamp: point.timestamp,
                emotion: point.emotion,
                intensity: point.intensity,
                duration: 2.0,
                context: "",
                confidence: min(1.0, point.intensity)
            )
        }
    }
    
    private func formatTime(_ seconds: Double) -> String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return String(format: "%02d:%02d", minutes, remainingSeconds)
    }
}

// MARK: - Supporting Classes

private class StoryStructureAnalyzer {
    func analyze(transcript: String, visualAnalysis: VisualAnalysis, duration: TimeInterval) async throws -> StoryStructure {
        // Simplified structure analysis - would implement more sophisticated NLP
        return StoryStructure(
            type: .threeAct,
            acts: [
                ActBreakdown(actNumber: 1, startTime: 0, endTime: duration/3, description: "Setup"),
                ActBreakdown(actNumber: 2, startTime: duration/3, endTime: 2*duration/3, description: "Confrontation"),
                ActBreakdown(actNumber: 3, startTime: 2*duration/3, endTime: duration, description: "Resolution")
            ],
            structurePoints: generateDefaultStructurePoints(duration: duration),
            overallArc: .positive
        )
    }
    
    private func generateDefaultStructurePoints(duration: TimeInterval) -> [StructurePoint] {
        return [
            StructurePoint(timestamp: 0.0, type: .threeAct, description: "Setup", confidence: 0.8, emotionalValence: 0.5),
            StructurePoint(timestamp: duration * 0.25, type: .threeAct, description: "Plot Point 1", confidence: 0.7, emotionalValence: 0.6),
            StructurePoint(timestamp: duration * 0.75, type: .threeAct, description: "Plot Point 2", confidence: 0.7, emotionalValence: 0.8),
            StructurePoint(timestamp: duration, type: .threeAct, description: "Resolution", confidence: 0.8, emotionalValence: 0.6)
        ]
    }
    
    private func generateActBreakdowns(duration: TimeInterval) -> [ActBreakdown] {
        return [
            ActBreakdown(actNumber: 1, startTime: 0.0, endTime: duration * 0.25, description: "Setup"),
            ActBreakdown(actNumber: 2, startTime: duration * 0.25, endTime: duration * 0.75, description: "Confrontation"),
            ActBreakdown(actNumber: 3, startTime: duration * 0.75, endTime: duration, description: "Resolution")
        ]
    }
}

private class TensionAnalyzer {
    func analyze(transcript: String, visualAnalysis: VisualAnalysis, audioAnalysis: AudioAnalysis, duration: TimeInterval) async throws -> TensionAnalysis {
        // Generate tension points based on audio energy and visual complexity
        var tensionPoints: [TensionPoint] = []
        
        // Use audio energy as tension indicator
        for energyLevel in audioAnalysis.energyLevels {
            let tensionType: TensionType
            if energyLevel.rms > 0.7 {
                tensionType = .peak
            } else if energyLevel.rms < 0.3 {
                tensionType = .valley
            } else {
                tensionType = .neutral
            }
            
            tensionPoints.append(TensionPoint(
                timestamp: energyLevel.timestamp,
                intensity: energyLevel.rms,
                type: tensionType,
                emotionalValence: energyLevel.rms > 0.5 ? 0.7 : 0.3,
                confidence: 0.8
            ))
        }
        
        return TensionAnalysis(
            tensionPoints: tensionPoints,
            averageTension: tensionPoints.reduce(0.0, { $0 + $1.intensity }) / Double(max(1, tensionPoints.count)),
            tensionRange: RangeDouble(min: tensionPoints.map{ $0.intensity }.min() ?? 0.0, max: tensionPoints.map{ $0.intensity }.max() ?? 1.0),
            tensionPattern: .rising,
            escalationPoints: []
        )
    }
}

private class PacingAnalyzer {
    func analyze(storyStructure: StoryStructure, tensionAnalysis: TensionAnalysis, audioAnalysis: AudioAnalysis, duration: TimeInterval) async throws -> PacingAnalysis {
        // Simplified pacing analysis
        let pacingChanges = generatePacingChanges(duration: duration)
        
        return PacingAnalysis(
            averagePace: "moderate",
            pacingChanges: pacingChanges,
            editingRhythm: calculateEditingRhythm(pacingChanges),
            optimalCutPoints: generateOptimalCutPoints(pacingChanges)
        )
    }
    
    private func generatePacingChanges(duration: TimeInterval) -> [PacingChange] {
        return [
            PacingChange(timestamp: duration * 0.2, fromPace: "slow", toPace: "moderate", changeIntensity: 0.6, confidence: 0.7),
            PacingChange(timestamp: duration * 0.6, fromPace: "moderate", toPace: "fast", changeIntensity: 0.8, confidence: 0.8),
            PacingChange(timestamp: duration * 0.9, fromPace: "fast", toPace: "moderate", changeIntensity: 0.7, confidence: 0.75)
        ]
    }
    
    private func calculateEditingRhythm(_ pacingChanges: [PacingChange]) -> EditingRhythm {
        return EditingRhythm(averageEditLength: 3.5, rhythmPattern: "accelerating", paceVariability: 0.6, editDensity: 0.3)
    }
    
    private func generateOptimalCutPoints(_ pacingChanges: [PacingChange]) -> [Double] {
        return pacingChanges.map { $0.timestamp }
    }
}

private class ThemeAnalyzer {
    func analyze(transcript: String, visualAnalysis: VisualAnalysis) async throws -> ThemeAnalysis {
        // Simplified theme analysis - would implement more sophisticated NLP
        return ThemeAnalysis(
            themes: [
                Theme(name: "Journey", confidence: 0.8, keyMoments: []),
                Theme(name: "Growth", confidence: 0.6, keyMoments: [])
            ],
            primaryTheme: Theme(name: "Journey", confidence: 0.7, keyMoments: []),
            themeStrength: 0.7,
            motifs: [],
            symbolism: [],
            themeProgression: ThemeProgression(stages: [])
        )
    }
}

private struct StoryModels {
    // Placeholder for story analysis models
    // Would contain trained models for structure recognition, theme analysis, etc.
}
