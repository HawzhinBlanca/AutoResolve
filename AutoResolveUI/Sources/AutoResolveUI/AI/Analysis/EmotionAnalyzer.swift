import Foundation
import SwiftUI
import Combine
import OSLog

@MainActor
public class EmotionAnalyzer: ObservableObject {
    private let logger = Logger.shared
    
    @Published public var isAnalyzing = false
    @Published public var analysisProgress = 0.0
    @Published public var currentOperation = ""
    
    public init() {}
    
    public func analyzeEmotions(
        transcript: String,
        visualAnalysis: VisualAnalysis,
        audioAnalysis: AudioAnalysis,
        duration: TimeInterval
    ) async throws -> TextSentimentResult {
        isAnalyzing = true
        defer { isAnalyzing = false; analysisProgress = 1.0; currentOperation = "" }
        currentOperation = "Analyzing transcript sentiment"
        
        let sentences = transcript
            .split(whereSeparator: { ".!?".contains($0) })
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let step = sentences.isEmpty ? 1.0 : max(1.0, duration / Double(sentences.count))
        
        var points: [SentimentPoint] = []
        var total: Double = 0
        for (idx, sentence) in sentences.enumerated() {
            let score = heuristicSentimentScore(sentence)
            total += score
            points.append(SentimentPoint(timestamp: Double(idx) * step, sentiment: score, text: sentence))
        }
        let overall = sentences.isEmpty ? 0.0 : total / Double(sentences.count)
        return TextSentimentResult(sentiments: points, overallSentiment: overall)
    }
    
    public func detectEmotionalHighlights(_ emotionTimeline: [EmotionTimePoint]) async throws -> [EmotionalHighlight] {
        guard emotionTimeline.count >= 3 else { return [] }
        var highlights: [EmotionalHighlight] = []
        for i in 1..<(emotionTimeline.count - 1) {
            let prev = emotionTimeline[i-1]
            let cur = emotionTimeline[i]
            let next = emotionTimeline[i+1]
            if cur.intensity > prev.intensity && cur.intensity > next.intensity && cur.intensity > 0.7 {
                let start = max(0, i - 2)
                let end = min(emotionTimeline.count - 1, i + 2)
                let context = (start...end).map { emotionTimeline[$0].emotionType }.joined(separator: " → ")
                highlights.append(
                    EmotionalHighlight(
                        timestamp: cur.timestamp,
                        emotion: cur.emotionType,
                        intensity: cur.intensity,
                        duration: 2.0,
                        context: "Emotional sequence: \(context)",
                        confidence: min(1.0, cur.intensity)
                    )
                )
            }
        }
        return Array(highlights.sorted { $0.intensity > $1.intensity }.prefix(10))
    }
    
    private func heuristicSentimentScore(_ text: String) -> Double {
        let positives = ["great", "good", "love", "amazing", "wonderful", "joy", "excited"]
        let negatives = ["bad", "terrible", "hate", "sad", "angry", "awful", "fear"]
        let lower = text.lowercased()
        let p = positives.reduce(0) { $0 + (lower.contains($1) ? 1 : 0) }
        let n = negatives.reduce(0) { $0 + (lower.contains($1) ? 1 : 0) }
        let raw = Double(p - n)
        return max(-1.0, min(1.0, raw / 3.0))
    }
}
