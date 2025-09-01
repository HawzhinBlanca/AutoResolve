import Foundation
import AutoResolveCore

// MARK: - Learning Module (Adaptive Weights)

public class Learning {
    private let eventStore = AIEventStore()
    
    // Learning parameters
    private let updateInterval: TimeInterval = 7 * 24 * 60 * 60  // Weekly
    private let clampPercentage = 0.1  // ±10% per update
    
    // Current weights
    public private(set) var currentWeights = FeatureWeights()
    
    // Tracking metrics
    private var acceptedSuggestions: [EditSuggestion] = []
    private var rejectedSuggestions: [EditSuggestion] = []
    private var revertedSuggestions: [EditSuggestion] = []
    private var lastUpdateDate = Date()
    
    public var revertRate: Double {
        let total = acceptedSuggestions.count
        let reverted = revertedSuggestions.count
        return total > 0 ? Double(reverted) / Double(total) : 0.0
    }
    
    public var keepRate: Double {
        let total = acceptedSuggestions.count + rejectedSuggestions.count
        let kept = acceptedSuggestions.count
        return total > 0 ? Double(kept) / Double(total) : 0.0
    }
    
    public init() {
        loadWeights()
    }
    
    // MARK: - Feedback Recording
    
    public func recordSuggestions(_ suggestions: [EditSuggestion]) {
        // Record that suggestions were made
        for suggestion in suggestions {
            eventStore.record(AIEvent(
                type: .suggestionMade,
                metadata: [
                    "type": String(describing: suggestion.type),
                    "confidence": suggestion.confidence,
                    "tick": suggestion.tick.value
                ]
            ))
        }
    }
    
    public func recordAcceptance(_ suggestion: EditSuggestion) {
        acceptedSuggestions.append(suggestion)
        
        eventStore.record(AIEvent(
            type: .suggestionAccepted,
            metadata: [
                "suggestionId": suggestion.id.uuidString,
                "type": String(describing: suggestion.type)
            ]
        ))
        
        checkForUpdate()
    }
    
    public func recordRejection(_ suggestion: EditSuggestion) {
        rejectedSuggestions.append(suggestion)
        
        eventStore.record(AIEvent(
            type: .suggestionRejected,
            metadata: [
                "suggestionId": suggestion.id.uuidString,
                "type": String(describing: suggestion.type)
            ]
        ))
        
        checkForUpdate()
    }
    
    public func recordRevert(_ suggestion: EditSuggestion) {
        revertedSuggestions.append(suggestion)
        
        // Move from accepted to reverted
        if let index = acceptedSuggestions.firstIndex(where: { $0.id == suggestion.id }) {
            acceptedSuggestions.remove(at: index)
        }
        
        eventStore.record(AIEvent(
            type: .suggestionReverted,
            metadata: [
                "suggestionId": suggestion.id.uuidString,
                "type": String(describing: suggestion.type)
            ]
        ))
        
        checkForUpdate()
    }
    
    // MARK: - Weight Updates
    
    private func checkForUpdate() {
        let timeSinceUpdate = Date().timeIntervalSince(lastUpdateDate)
        
        if timeSinceUpdate >= updateInterval {
            updateWeights()
        }
    }
    
    private func updateWeights() {
        // Calculate adjustment factors based on keep/reject rates
        let keepRate = self.keepRate
        let targetKeepRate = 0.7  // Target 70% keep rate
        
        let adjustment = (keepRate - targetKeepRate) * clampPercentage
        
        // Update weights based on which suggestions were accepted
        analyzeAcceptancePatterns()
        
        // Apply adjustment with clamping
        currentWeights.silenceWeight += currentWeights.silenceWeight * adjustment
        currentWeights.sceneWeight += currentWeights.sceneWeight * adjustment
        currentWeights.pacingWeight += currentWeights.pacingWeight * adjustment
        currentWeights.asrWeight += currentWeights.asrWeight * adjustment
        
        // Penalize if high revert rate
        if revertRate > 0.2 {
            currentWeights.revertWeight = -0.5
        }
        
        // Clamp weights
        currentWeights.clamp()
        
        // Save and reset
        saveWeights()
        lastUpdateDate = Date()
        
        // Record update
        eventStore.record(AIEvent(
            type: .learningUpdated,
            metadata: [
                "keepRate": keepRate,
                "revertRate": revertRate,
                "silenceWeight": currentWeights.silenceWeight,
                "sceneWeight": currentWeights.sceneWeight
            ]
        ))
    }
    
    private func analyzeAcceptancePatterns() {
        // Analyze which types of suggestions are accepted more
        let acceptedTypes = acceptedSuggestions.map { $0.type }
        let rejectedTypes = rejectedSuggestions.map { $0.type }
        
        // Count by type
        var acceptancByType: [String: Double] = [:]
        
        for suggestion in acceptedSuggestions {
            let key = String(describing: suggestion.type)
            acceptancByType[key, default: 0] += 1
        }
        
        for suggestion in rejectedSuggestions {
            let key = String(describing: suggestion.type)
            acceptancByType[key, default: 0] -= 0.5
        }
        
        // Adjust weights based on patterns
        // This is simplified - real implementation would be more sophisticated
    }
    
    // MARK: - Persistence
    
    private func loadWeights() {
        // Load from UserDefaults or file
        if let data = UserDefaults.standard.data(forKey: "AIDirector.Weights"),
           let weights = try? JSONDecoder().decode(FeatureWeights.self, from: data) {
            currentWeights = weights
        }
    }
    
    private func saveWeights() {
        // Save to UserDefaults or file
        if let data = try? JSONEncoder().encode(currentWeights) {
            UserDefaults.standard.set(data, forKey: "AIDirector.Weights")
        }
    }
    
    // MARK: - Statistics
    
    public func getStatistics() -> LearningStatistics {
        return LearningStatistics(
            keepRate: keepRate,
            revertRate: revertRate,
            totalSuggestions: acceptedSuggestions.count + rejectedSuggestions.count,
            acceptedSuggestions: acceptedSuggestions.count,
            rejectedSuggestions: rejectedSuggestions.count,
            revertedSuggestions: revertedSuggestions.count,
            currentWeights: currentWeights,
            lastUpdate: lastUpdateDate
        )
    }
}

// MARK: - Learning Statistics

public struct LearningStatistics {
    public let keepRate: Double
    public let revertRate: Double
    public let totalSuggestions: Int
    public let acceptedSuggestions: Int
    public let rejectedSuggestions: Int
    public let revertedSuggestions: Int
    public let currentWeights: FeatureWeights
    public let lastUpdate: Date
    
    public var description: String {
        """
        Learning Statistics:
        - Keep Rate: \(String(format: "%.1f%%", keepRate * 100))
        - Revert Rate: \(String(format: "%.1f%%", revertRate * 100))
        - Total Suggestions: \(totalSuggestions)
        - Accepted: \(acceptedSuggestions)
        - Rejected: \(rejectedSuggestions)
        - Reverted: \(revertedSuggestions)
        - Last Update: \(lastUpdate)
        """
    }
}

// Make FeatureWeights Codable for persistence
extension FeatureWeights: Codable {}