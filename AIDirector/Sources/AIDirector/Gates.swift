import Foundation
import AutoResolveCore
import Accelerate

// MARK: - Gates Module (Quality Control)

public class Gates {
    // Gate thresholds (from blueprint)
    private let minShotDuration = 0.8  // 0.8 seconds minimum
    private let audioPopThreshold: Float = 6.0  // 6 dB step
    private let audioPopWindow = 0.01  // 10ms window
    private let crossfadeDuration = 0.08  // 80ms crossfade
    
    private let eventStore = AIEventStore()
    
    public init() {}
    
    // MARK: - Gate Validation
    
    public func validate(_ suggestion: EditSuggestion, in timeline: Timeline) -> Bool {
        var passed = true
        
        // Apply all gates
        passed = passed && minShotGate(suggestion, timeline: timeline)
        passed = passed && renderSanityGate(suggestion, timeline: timeline)
        
        // MidWord gate only if ASR present
        if hasASR(timeline) {
            passed = passed && midWordGate(suggestion, timeline: timeline)
        }
        
        passed = passed && audioPopGate(suggestion, timeline: timeline)
        
        // Record gate results
        if !passed {
            eventStore.record(AIEvent(
                type: .gateTriggered,
                metadata: [
                    "suggestionId": suggestion.id.uuidString,
                    "tick": suggestion.tick.value
                ]
            ))
        }
        
        return passed
    }
    
    // MARK: - Individual Gates
    
    // MinShot Gate: No cuts creating shots < 0.8s
    private func minShotGate(_ suggestion: EditSuggestion, timeline: Timeline) -> Bool {
        guard case .cut = suggestion.type else { return true }
        
        // Find clips around the cut point
        for track in timeline.tracks {
            for clip in track.clips {
                // Check if cut would create a short segment
                if clip.contains(suggestion.tick) {
                    let beforeDuration = (suggestion.tick - clip.startTick).seconds
                    let afterDuration = (clip.endTick - suggestion.tick).seconds
                    
                    if beforeDuration < minShotDuration || afterDuration < minShotDuration {
                        return false  // Gate triggered
                    }
                }
            }
        }
        
        return true
    }
    
    // RenderSanity Gate: No black frames at transitions
    private func renderSanityGate(_ suggestion: EditSuggestion, timeline: Timeline) -> Bool {
        // This would check for black frames using luma threshold
        // Simplified for now
        
        switch suggestion.type {
        case .transition:
            // Ensure frames exist on both sides
            return hasFramesAt(suggestion.tick, in: timeline)
        default:
            return true
        }
    }
    
    // MidWord Gate: No cuts in middle of words (ASR required)
    private func midWordGate(_ suggestion: EditSuggestion, timeline: Timeline) -> Bool {
        guard case .cut = suggestion.type else { return true }
        
        // This would check ASR word boundaries
        // For now, assume we have word timing data
        
        // Mock implementation
        let wordBoundaries: [Tick] = [
            Tick.from(seconds: 1.2),
            Tick.from(seconds: 2.5),
            Tick.from(seconds: 3.8)
        ]
        
        // Check if cut is too close to word boundary
        let threshold = Tick.from(seconds: 0.1)  // 100ms threshold
        
        for boundary in wordBoundaries {
            let distance = abs(suggestion.tick.value - boundary.value)
            if distance < threshold.value {
                return false  // Too close to word boundary
            }
        }
        
        return true
    }
    
    // AudioPop Gate: Detect audio discontinuities
    private func audioPopGate(_ suggestion: EditSuggestion, timeline: Timeline) -> Bool {
        guard case .cut = suggestion.type else { return true }
        
        // This would analyze audio waveform around cut point
        // Check for sudden amplitude changes
        
        // Mock implementation
        let hasAudioPop = checkForAudioPop(at: suggestion.tick)
        
        if hasAudioPop {
            // Would insert crossfade automatically
            return false  // For now, reject cuts with pops
        }
        
        return true
    }
    
    // MARK: - Helper Methods
    
    private func hasASR(_ timeline: Timeline) -> Bool {
        // Check if ASR data is available
        // Could check for CoreML model or backend ASR
        
        // Check if CoreML Whisper is available
        if Bundle.main.url(forResource: "whisper-tiny.en", withExtension: "mlmodelc") != nil {
            return true
        }
        
        // Otherwise log and disable
        print("AID-ASR-OFF: ASR unavailable, MidWord gate disabled")
        return false
    }
    
    private func hasFramesAt(_ tick: Tick, in timeline: Timeline) -> Bool {
        // Check if there are valid frames at the given tick
        for track in timeline.tracks {
            for clip in track.clips {
                if clip.contains(tick) {
                    return true
                }
            }
        }
        return false
    }
    
    private func checkForAudioPop(at tick: Tick) -> Bool {
        // Would analyze audio samples around the tick
        // Looking for step > 6dB in 10ms window
        
        // Mock implementation
        return false  // Assume no pop for now
    }
    
    // MARK: - Crossfade Insertion
    
    public func insertCrossfade(at tick: Tick, duration: Double = 0.08) -> Command {
        // Create a transition command
        // This would be applied to fix audio pops
        
        return Command.blade(at: tick, trackIndex: 0)  // Placeholder
    }
}

// MARK: - Gate Statistics

extension Gates {
    public func getStatistics() -> GateStatistics {
        let analysis = eventStore.analyze()
        
        return GateStatistics(
            totalValidations: analysis.totalEvents,
            gatesTriggered: analysis.suggestionsMade,
            passRate: 1.0 - (Double(analysis.suggestionsRejected) / Double(max(1, analysis.totalEvents)))
        )
    }
}

public struct GateStatistics {
    public let totalValidations: Int
    public let gatesTriggered: Int
    public let passRate: Double
}