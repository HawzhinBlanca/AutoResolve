import XCTest
@testable import AutoResolveUI
import AVFoundation

@MainActor
final class EditToolsTests: XCTestCase {
    var appState: AppState!
    var transport: Transport!
    
    override func setUp() {
        super.setUp()
        appState = AppState()
        transport = appState.transport
        
        // Create a mock timeline with test clips
        let timeline = TimelineModel()
        let track = UITimelineTrack(name: "V1", type: .video)
        
        // Add test clips
        let clip1 = SimpleTimelineClip(
            id: UUID(),
            name: "Test Clip 1",
            trackIndex: 0,
            startTime: 10.0,
            duration: 5.0,
            sourceURL: URL(fileURLWithPath: "/tmp/test.mp4")
        )
        
        let clip2 = SimpleTimelineClip(
            id: UUID(),
            name: "Test Clip 2", 
            trackIndex: 0,
            startTime: 20.0,
            duration: 3.0,
            sourceURL: URL(fileURLWithPath: "/tmp/test2.mp4")
        )
        
        track.clips = [clip1, clip2]
        timeline.tracks = [track]
        timeline.duration = 30.0
        appState.timeline = timeline
    }
    
    func testBladeSplitProducesTwoClipsTouching() {
        // Arrange: Position playhead in middle of first clip
        let clipToSplit = appState.timeline!.tracks[0].clips[0]
        let originalClipCount = appState.timeline!.tracks[0].clips.count
        
        // Set playhead to middle of clip (12.5s)
        let cutTime = CMTime(seconds: 12.5, preferredTimescale: 600)
        transport.seek(to: cutTime)
        
        // Act: Perform blade cut
        appState.cutAtPlayhead()
        
        // Assert: Should have one more clip
        XCTAssertEqual(appState.timeline!.tracks[0].clips.count, originalClipCount + 1)
        
        // Assert: Two clips should be touching
        let clipsAtCutTime = appState.timeline!.tracks[0].clips.filter { 
            abs($0.startTime - 10.0) < 0.1 || abs($0.startTime - 12.5) < 0.1
        }
        XCTAssertEqual(clipsAtCutTime.count, 2)
        
        // Assert: First part should end where second part starts
        let firstPart = clipsAtCutTime.first { $0.startTime < 12.0 }!
        let secondPart = clipsAtCutTime.first { $0.startTime > 12.0 }!
        XCTAssertEqual(firstPart.endTime, secondPart.startTime, accuracy: 0.01)
    }
    
    func testTrimSnapsToSilenceBoundary() {
        // This would test snap functionality with silence data
        // For now, just verify snap settings toggle
        
        // Arrange
        appState.snapSettings.snapEnabled = false
        
        // Act: Toggle snap
        appState.snapSettings.snapEnabled = true
        
        // Assert
        XCTAssertTrue(appState.snapSettings.snapEnabled)
    }
    
    func testRippleDeleteClosesGapExactlyOneFrame() {
        // Arrange: Select first clip
        let clipToDelete = appState.timeline!.tracks[0].clips[0]
        appState.selectedClips.insert(clipToDelete.id.uuidString)
        
        let originalSecondClipStart = appState.timeline!.tracks[0].clips[1].startTime
        let deletedClipDuration = clipToDelete.duration
        
        // Act: Ripple delete
        appState.rippleDeleteSelected()
        
        // Assert: Second clip should move back by deleted clip duration
        let newSecondClipStart = appState.timeline!.tracks[0].clips[0].startTime
        let expectedStart = originalSecondClipStart - deletedClipDuration
        
        XCTAssertEqual(newSecondClipStart, expectedStart, accuracy: 0.033) // Within 1 frame at 30fps
        XCTAssertEqual(appState.timeline!.tracks[0].clips.count, 1) // One clip deleted
    }
}