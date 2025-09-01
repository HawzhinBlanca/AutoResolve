import XCTest
@testable import AutoResolveCore
@testable import AIDirector
@testable import AutoResolveUI

final class EndToEndTests: XCTestCase {
    var project: Project!
    var director: Director!
    var appState: AppState!
    
    override func setUp() async throws {
        try await super.setUp()
        
        project = Project(name: "Test Project")
        director = Director()
        appState = AppState()
        
        // Setup test timeline
        setupTestTimeline()
    }
    
    private func setupTestTimeline() {
        let timeline = Timeline()
        
        // Add test tracks and clips
        let videoTrack = Track(index: 0, type: .video)
        let audioTrack = Track(index: 1, type: .audio)
        
        // Add clips
        for i in 0..<5 {
            let start = Tick.from(seconds: Double(i * 10))
            let end = Tick.from(seconds: Double((i + 1) * 10))
            
            let clip = Clip(
                id: UUID(),
                startTick: start,
                endTick: end,
                mediaRef: UUID()
            )
            
            videoTrack.clips.append(clip)
        }
        
        timeline.tracks = [videoTrack, audioTrack]
        project.timeline = timeline
    }
    
    func testFullAnalysisPipeline() async throws {
        // Run AI analysis
        let suggestions = await director.analyze(project.timeline)
        
        XCTAssertFalse(suggestions.isEmpty, "Should generate suggestions")
        
        // Validate all suggestions pass gates
        for suggestion in suggestions {
            XCTAssertGreaterThan(suggestion.confidence, 0.5)
        }
    }
    
    func testCommandExecution() throws {
        // Test blade command
        let bladeCommand = Command.blade(at: Tick.from(seconds: 5), trackIndex: 0)
        try project.execute(bladeCommand)
        
        // Verify timeline was modified
        XCTAssertEqual(project.timeline.tracks[0].clips.count, 2)
    }
    
    func testUndoRedo() throws {
        let processor = CommandProcessor()
        
        // Execute command
        let command = Command.blade(at: Tick.from(seconds: 5), trackIndex: 0)
        try processor.execute(command, on: project)
        
        let clipsAfterBlade = project.timeline.tracks[0].clips.count
        
        // Undo
        try processor.undo(on: project)
        XCTAssertEqual(project.timeline.tracks[0].clips.count, 1)
        
        // Redo
        try processor.redo(on: project)
        XCTAssertEqual(project.timeline.tracks[0].clips.count, clipsAfterBlade)
    }
    
    func testPerformanceGates() throws {
        measure {
            // Measure timeline operations
            for _ in 0..<100 {
                let tick = Tick.from(seconds: Double.random(in: 0...50))
                let _ = Command.blade(at: tick, trackIndex: 0)
            }
        }
    }
}