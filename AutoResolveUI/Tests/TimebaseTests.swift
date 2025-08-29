import XCTest
import CoreMedia
@testable import AutoResolveUI

final class TimebaseTests: XCTestCase {
    var timebase: Timebase!
    
    override func setUp() {
        super.setUp()
        timebase = Timebase(fps: 30.0)
    }
    
    // MARK: - Frame/Time Conversion Tests
    
    func testFrameToTimeConversion() {
        // Test frame 0
        let time0 = timebase.timeFromFrame(0)
        XCTAssertEqual(CMTimeGetSeconds(time0), 0.0, accuracy: 0.001)
        
        // Test frame 30 (1 second at 30fps)
        let time30 = timebase.timeFromFrame(30)
        XCTAssertEqual(CMTimeGetSeconds(time30), 1.0, accuracy: 0.001)
        
        // Test frame 90 (3 seconds at 30fps)
        let time90 = timebase.timeFromFrame(90)
        XCTAssertEqual(CMTimeGetSeconds(time90), 3.0, accuracy: 0.001)
    }
    
    func testTimeToFrameConversion() {
        // Test 0 seconds
        let frame0 = timebase.frameFromTime(CMTime(seconds: 0, preferredTimescale: 600))
        XCTAssertEqual(frame0, 0)
        
        // Test 1 second
        let frame30 = timebase.frameFromTime(CMTime(seconds: 1.0, preferredTimescale: 600))
        XCTAssertEqual(frame30, 30)
        
        // Test 2.5 seconds (should round to frame 75)
        let frame75 = timebase.frameFromTime(CMTime(seconds: 2.5, preferredTimescale: 600))
        XCTAssertEqual(frame75, 75)
    }
    
    // MARK: - Pixel/Time Conversion Tests
    
    func testPixelToTimeConversion() {
        // At zoom 1.0, 100 pixels = 1 second
        let time1 = timebase.timeFromPixels(100, zoomLevel: 1.0)
        XCTAssertEqual(CMTimeGetSeconds(time1), 1.0, accuracy: 0.001)
        
        // At zoom 2.0, 200 pixels = 1 second
        let time2 = timebase.timeFromPixels(200, zoomLevel: 2.0)
        XCTAssertEqual(CMTimeGetSeconds(time2), 1.0, accuracy: 0.001)
        
        // At zoom 0.5, 50 pixels = 1 second
        let time3 = timebase.timeFromPixels(50, zoomLevel: 0.5)
        XCTAssertEqual(CMTimeGetSeconds(time3), 1.0, accuracy: 0.001)
    }
    
    func testTimeToPixelConversion() {
        let oneSecond = CMTime(seconds: 1.0, preferredTimescale: 600)
        
        // At zoom 1.0, 1 second = 100 pixels
        let pixels1 = timebase.pixelsFromTime(oneSecond, zoomLevel: 1.0)
        XCTAssertEqual(pixels1, 100.0, accuracy: 0.001)
        
        // At zoom 2.0, 1 second = 200 pixels
        let pixels2 = timebase.pixelsFromTime(oneSecond, zoomLevel: 2.0)
        XCTAssertEqual(pixels2, 200.0, accuracy: 0.001)
        
        // At zoom 0.5, 1 second = 50 pixels
        let pixels3 = timebase.pixelsFromTime(oneSecond, zoomLevel: 0.5)
        XCTAssertEqual(pixels3, 50.0, accuracy: 0.001)
    }
    
    // MARK: - Timecode Tests
    
    func testTimecodeFormatting() {
        // Test 00:00:00:00
        let tc1 = timebase.timecodeFromTime(CMTime(seconds: 0, preferredTimescale: 600))
        XCTAssertEqual(tc1, "00:00:00:00")
        
        // Test 00:00:01:00 (1 second)
        let tc2 = timebase.timecodeFromTime(CMTime(seconds: 1.0, preferredTimescale: 600))
        XCTAssertEqual(tc2, "00:00:01:00")
        
        // Test 00:01:00:00 (1 minute)
        let tc3 = timebase.timecodeFromTime(CMTime(seconds: 60.0, preferredTimescale: 600))
        XCTAssertEqual(tc3, "00:01:00:00")
        
        // Test 01:00:00:00 (1 hour)
        let tc4 = timebase.timecodeFromTime(CMTime(seconds: 3600.0, preferredTimescale: 600))
        XCTAssertEqual(tc4, "01:00:00:00")
        
        // Test with frames: 00:00:01:15 (1.5 seconds at 30fps)
        let tc5 = timebase.timecodeFromTime(CMTime(seconds: 1.5, preferredTimescale: 600))
        XCTAssertEqual(tc5, "00:00:01:15")
    }
    
    func testTimecodeParsing() {
        // Test parsing 00:00:00:00
        let time1 = timebase.timeFromTimecode("00:00:00:00")
        XCTAssertNotNil(time1)
        XCTAssertEqual(CMTimeGetSeconds(time1!), 0.0, accuracy: 0.001)
        
        // Test parsing 00:00:01:00
        let time2 = timebase.timeFromTimecode("00:00:01:00")
        XCTAssertNotNil(time2)
        XCTAssertEqual(CMTimeGetSeconds(time2!), 1.0, accuracy: 0.001)
        
        // Test parsing 00:00:01:15 (1.5 seconds at 30fps)
        let time3 = timebase.timeFromTimecode("00:00:01:15")
        XCTAssertNotNil(time3)
        XCTAssertEqual(CMTimeGetSeconds(time3!), 1.5, accuracy: 0.001)
        
        // Test invalid timecode
        let time4 = timebase.timeFromTimecode("invalid")
        XCTAssertNil(time4)
    }
    
    // MARK: - Snapping Tests
    
    func testFrameSnapping() {
        // Test snapping to nearest frame
        let unsnapped = CMTime(seconds: 1.016, preferredTimescale: 600) // Between frames
        let snapped = timebase.snapToFrame(unsnapped)
        
        // Should snap to frame 30 (1.0 seconds)
        let expectedFrame = 30
        let actualFrame = timebase.frameFromTime(snapped)
        XCTAssertEqual(actualFrame, expectedFrame)
    }
    
    func testSnapPointFinding() {
        let clips = [
            TimelineClip(
                startTime: CMTime(seconds: 1.0, preferredTimescale: 600),
                duration: CMTime(seconds: 2.0, preferredTimescale: 600)
            ),
            TimelineClip(
                startTime: CMTime(seconds: 5.0, preferredTimescale: 600),
                duration: CMTime(seconds: 3.0, preferredTimescale: 600)
            )
        ]
        
        let markers = [
            TimelineMarker(
                time: CMTime(seconds: 4.0, preferredTimescale: 600),
                name: "Marker 1"
            )
        ]
        
        let settings = SnapSettings()
        
        // Test snapping near clip start
        let nearClipStart = CMTime(seconds: 0.98, preferredTimescale: 600)
        let snappedPoint = timebase.findSnapPoint(
            for: nearClipStart,
            clips: clips,
            markers: markers,
            settings: settings,
            zoomLevel: 1.0
        )
        
        XCTAssertNotNil(snappedPoint)
        // Should snap to clip at 1.0 seconds
        XCTAssertEqual(CMTimeGetSeconds(snappedPoint!), 1.0, accuracy: 0.01)
        
        // Test snapping near marker
        let nearMarker = CMTime(seconds: 3.95, preferredTimescale: 600)
        let snappedToMarker = timebase.findSnapPoint(
            for: nearMarker,
            clips: clips,
            markers: markers,
            settings: settings,
            zoomLevel: 1.0
        )
        
        XCTAssertNotNil(snappedToMarker)
        // Should snap to marker at 4.0 seconds
        XCTAssertEqual(CMTimeGetSeconds(snappedToMarker!), 4.0, accuracy: 0.01)
    }
}