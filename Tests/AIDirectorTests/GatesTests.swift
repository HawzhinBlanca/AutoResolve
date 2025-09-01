import XCTest
@testable import AIDirector
@testable import AutoResolveCore

final class GatesTests: XCTestCase {
    var gates: Gates!
    var timeline: Timeline!
    
    override func setUp() {
        super.setUp()
        gates = Gates()
        timeline = Timeline()
        
        // Setup test timeline
        let track = Track(index: 0, type: .video)
        let clip = Clip(
            id: UUID(),
            startTick: Tick.from(seconds: 0),
            endTick: Tick.from(seconds: 5),
            mediaRef: UUID()
        )
        track.clips.append(clip)
        timeline.tracks.append(track)
    }
    
    func testMinShotGate() {
        // Test that cuts creating shots < 0.8s are rejected
        let suggestion = EditSuggestion(
            type: .cut,
            tick: Tick.from(seconds: 0.5),
            confidence: 0.9,
            reason: "Test cut"
        )
        
        let result = gates.validate(suggestion, in: timeline)
        XCTAssertFalse(result, "Should reject cut creating short shot")
    }
    
    func testValidCut() {
        // Test that valid cuts are accepted
        let suggestion = EditSuggestion(
            type: .cut,
            tick: Tick.from(seconds: 2.5),
            confidence: 0.9,
            reason: "Test cut"
        )
        
        let result = gates.validate(suggestion, in: timeline)
        XCTAssertTrue(result, "Should accept valid cut")
    }
    
    func testRenderSanityGate() {
        // Test transition at valid point
        let suggestion = EditSuggestion(
            type: .transition,
            tick: Tick.from(seconds: 2.5),
            confidence: 0.8,
            reason: "Test transition"
        )
        
        let result = gates.validate(suggestion, in: timeline)
        XCTAssertTrue(result, "Should accept transition at valid point")
    }
    
    func testGateStatistics() {
        let stats = gates.getStatistics()
        
        XCTAssertGreaterThanOrEqual(stats.passRate, 0.0)
        XCTAssertLessThanOrEqual(stats.passRate, 1.0)
    }
}