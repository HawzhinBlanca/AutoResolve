import Foundation
import AutoResolveCore

// MARK: - AIDirector Event Store

public class AIEventStore {
    private var events: [AIEvent] = []
    private let maxEvents = 10000
    
    public init() {}
    
    public func record(_ event: AIEvent) {
        events.append(event)
        
        // Circular buffer
        if events.count > maxEvents {
            events.removeFirst(events.count - maxEvents)
        }
    }
    
    public func getEvents(type: AIEventType? = nil, last: Int? = nil) -> [AIEvent] {
        var filtered = events
        
        if let type = type {
            filtered = filtered.filter { $0.type == type }
        }
        
        if let last = last {
            filtered = Array(filtered.suffix(last))
        }
        
        return filtered
    }
    
    public func analyze() -> AIEventAnalysis {
        let suggestionEvents = events.filter { $0.type == .suggestionMade }
        let acceptedEvents = events.filter { $0.type == .suggestionAccepted }
        let rejectedEvents = events.filter { $0.type == .suggestionRejected }
        
        let acceptanceRate = Double(acceptedEvents.count) / Double(max(1, suggestionEvents.count))
        
        return AIEventAnalysis(
            totalEvents: events.count,
            suggestionsMade: suggestionEvents.count,
            suggestionsAccepted: acceptedEvents.count,
            suggestionsRejected: rejectedEvents.count,
            acceptanceRate: acceptanceRate
        )
    }
}

// MARK: - Event Types

public struct AIEvent {
    public let id = UUID()
    public let timestamp = Date()
    public let type: AIEventType
    public let metadata: [String: Any]
    
    public init(type: AIEventType, metadata: [String: Any] = [:]) {
        self.type = type
        self.metadata = metadata
    }
}

public enum AIEventType {
    case analysisStarted
    case analysisCompleted
    case suggestionMade
    case suggestionAccepted
    case suggestionRejected
    case suggestionReverted
    case gateTriggered
    case learningUpdated
}

public struct AIEventAnalysis {
    public let totalEvents: Int
    public let suggestionsMade: Int
    public let suggestionsAccepted: Int
    public let suggestionsRejected: Int
    public let acceptanceRate: Double
}