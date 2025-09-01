import Foundation
import AutoResolveCore

// MARK: - Planner Module (Greedy + Priority Queue)

public class Planner {
    // Feature weights (updated by learning)
    public var weights = FeatureWeights()
    
    // Planning parameters
    private let maxSuggestions = 20
    private let minConfidence = 0.5
    
    public init() {}
    
    // MARK: - Suggestion Generation
    
    public func generateSuggestions(
        features: FeatureVector,
        weights: FeatureWeights
    ) -> [EditSuggestion] {
        self.weights = weights
        
        var suggestions: [EditSuggestion] = []
        
        // Priority queue for suggestions
        var priorityQueue = PriorityQueue<EditSuggestion> { a, b in
            a.confidence > b.confidence
        }
        
        // Generate cut suggestions based on silence
        if features.silenceFraction > 0.1 {
            let silenceSuggestions = generateSilenceCuts(features)
            silenceSuggestions.forEach { priorityQueue.enqueue($0) }
        }
        
        // Generate scene-based suggestions
        if features.cutDensity < 0.5 {
            let sceneSuggestions = generateSceneCuts(features)
            sceneSuggestions.forEach { priorityQueue.enqueue($0) }
        }
        
        // Generate pacing suggestions
        if features.avgShotLength > 10.0 {
            let pacingSuggestions = generatePacingCuts(features)
            pacingSuggestions.forEach { priorityQueue.enqueue($0) }
        }
        
        // Extract top suggestions
        while !priorityQueue.isEmpty && suggestions.count < maxSuggestions {
            if let suggestion = priorityQueue.dequeue() {
                if suggestion.confidence >= minConfidence {
                    suggestions.append(suggestion)
                }
            }
        }
        
        return suggestions
    }
    
    // MARK: - Suggestion Generators
    
    private func generateSilenceCuts(_ features: FeatureVector) -> [EditSuggestion] {
        // Generate cuts at silence boundaries
        var suggestions: [EditSuggestion] = []
        
        // Mock implementation - would use actual silence ranges
        let confidence = weights.silenceWeight * (1.0 - features.silenceFraction)
        
        suggestions.append(EditSuggestion(
            type: .cut,
            tick: Tick.from(seconds: 5.0),
            confidence: confidence,
            reason: "Cut at silence boundary"
        ))
        
        return suggestions
    }
    
    private func generateSceneCuts(_ features: FeatureVector) -> [EditSuggestion] {
        // Generate cuts at scene changes
        var suggestions: [EditSuggestion] = []
        
        let confidence = weights.sceneWeight * features.cutDensity
        
        suggestions.append(EditSuggestion(
            type: .cut,
            tick: Tick.from(seconds: 8.5),
            confidence: confidence,
            reason: "Cut at scene change"
        ))
        
        return suggestions
    }
    
    private func generatePacingCuts(_ features: FeatureVector) -> [EditSuggestion] {
        // Generate cuts to improve pacing
        var suggestions: [EditSuggestion] = []
        
        let targetShotLength = 4.0  // Target 4 seconds per shot
        let pacingScore = 1.0 - abs(features.avgShotLength - targetShotLength) / targetShotLength
        let confidence = weights.pacingWeight * pacingScore
        
        suggestions.append(EditSuggestion(
            type: .cut,
            tick: Tick.from(seconds: 12.0),
            confidence: confidence,
            reason: "Improve pacing rhythm"
        ))
        
        return suggestions
    }
}

// MARK: - Feature Weights

public struct FeatureWeights {
    public var silenceWeight: Double = 0.8
    public var sceneWeight: Double = 0.9
    public var pacingWeight: Double = 0.6
    public var asrWeight: Double = 0.7
    public var revertWeight: Double = -0.5
    
    public mutating func clamp() {
        // Clamp weights by ±10% as per blueprint
        silenceWeight = clamp(silenceWeight, by: 0.1)
        sceneWeight = clamp(sceneWeight, by: 0.1)
        pacingWeight = clamp(pacingWeight, by: 0.1)
        asrWeight = clamp(asrWeight, by: 0.1)
        revertWeight = clamp(revertWeight, by: 0.1)
    }
    
    private func clamp(_ value: Double, by percentage: Double) -> Double {
        let delta = value * percentage
        return max(-1.0, min(1.0, value + delta))
    }
}

// MARK: - Priority Queue

struct PriorityQueue<T> {
    private var heap: [T] = []
    private let compare: (T, T) -> Bool
    
    init(compare: @escaping (T, T) -> Bool) {
        self.compare = compare
    }
    
    var isEmpty: Bool {
        heap.isEmpty
    }
    
    mutating func enqueue(_ element: T) {
        heap.append(element)
        siftUp(heap.count - 1)
    }
    
    mutating func dequeue() -> T? {
        guard !heap.isEmpty else { return nil }
        
        if heap.count == 1 {
            return heap.removeLast()
        }
        
        let first = heap[0]
        heap[0] = heap.removeLast()
        siftDown(0)
        
        return first
    }
    
    private mutating func siftUp(_ index: Int) {
        var child = index
        var parent = (child - 1) / 2
        
        while child > 0 && compare(heap[child], heap[parent]) {
            heap.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }
    
    private mutating func siftDown(_ index: Int) {
        var parent = index
        
        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var candidate = parent
            
            if left < heap.count && compare(heap[left], heap[candidate]) {
                candidate = left
            }
            
            if right < heap.count && compare(heap[right], heap[candidate]) {
                candidate = right
            }
            
            if candidate == parent {
                break
            }
            
            heap.swapAt(parent, candidate)
            parent = candidate
        }
    }
}