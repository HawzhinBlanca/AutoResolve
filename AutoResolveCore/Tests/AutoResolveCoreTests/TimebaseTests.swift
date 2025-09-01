import XCTest
@testable import AutoResolveCore

final class TimebaseTests: XCTestCase {
    func testTickCreation() {
        let tick = Tick(1000)
        XCTAssertEqual(tick.value, 1000)
        XCTAssertEqual(tick.milliseconds, 1000)
        XCTAssertEqual(tick.seconds, 1.0)
    }
    
    func testTickFromSeconds() {
        let tick = Tick.from(seconds: 2.5)
        XCTAssertEqual(tick.value, 2500)
        XCTAssertEqual(tick.seconds, 2.5)
    }
    
    func testTickArithmetic() {
        let t1 = Tick(1000)
        let t2 = Tick(500)
        
        XCTAssertEqual((t1 + t2).value, 1500)
        XCTAssertEqual((t1 - t2).value, 500)
        XCTAssertEqual((t1 * 2.0).value, 2000)
    }
    
    func testTickComparison() {
        let t1 = Tick(1000)
        let t2 = Tick(2000)
        
        XCTAssertTrue(t1 < t2)
        XCTAssertFalse(t2 < t1)
        XCTAssertEqual(t1, Tick(1000))
    }
    
    func testTickRange() {
        let range = TickRange(start: Tick(1000), end: Tick(3000))
        
        XCTAssertEqual(range.duration.value, 2000)
        XCTAssertTrue(range.contains(Tick(2000)))
        XCTAssertFalse(range.contains(Tick(4000)))
    }
    
    func testTickRangeOverlap() {
        let r1 = TickRange(start: Tick(1000), end: Tick(3000))
        let r2 = TickRange(start: Tick(2000), end: Tick(4000))
        let r3 = TickRange(start: Tick(4000), end: Tick(5000))
        
        XCTAssertTrue(r1.overlaps(r2))
        XCTAssertFalse(r1.overlaps(r3))
    }
    
    func testTickRangeIntersection() {
        let r1 = TickRange(start: Tick(1000), end: Tick(3000))
        let r2 = TickRange(start: Tick(2000), end: Tick(4000))
        
        let intersection = r1.intersection(r2)
        XCTAssertNotNil(intersection)
        XCTAssertEqual(intersection?.start.value, 2000)
        XCTAssertEqual(intersection?.end.value, 3000)
    }
    
    func testTimebaseFrameConversion() {
        let timebase = Timebase(fps: 30.0)
        
        let tick = Tick.from(seconds: 1.0)
        XCTAssertEqual(timebase.tickToFrame(tick), 30)
        
        let frame: Int64 = 60
        let frameTick = timebase.frameToTick(frame)
        XCTAssertEqual(frameTick.seconds, 2.0, accuracy: 0.001)
    }
    
    func testTimebaseSnapToFrame() {
        let timebase = Timebase(fps: 30.0)
        
        let tick = Tick.from(seconds: 1.016) 
        let snapped = timebase.snapToFrame(tick)
        
        XCTAssertEqual(snapped.seconds, 1.0, accuracy: 0.034)
    }
}