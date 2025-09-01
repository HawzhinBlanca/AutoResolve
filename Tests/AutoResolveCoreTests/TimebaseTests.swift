import XCTest
@testable import AutoResolveCore

final class TimebaseTests: XCTestCase {
    func testTickCreation() {
        let tick = Tick(value: 1000)
        XCTAssertEqual(tick.value, 1000)
    }
    
    func testTickFromSeconds() {
        let tick = Tick.from(seconds: 1.0)
        XCTAssertEqual(tick.value, 1000)
    }
    
    func testTickToSeconds() {
        let tick = Tick(value: 500)
        XCTAssertEqual(tick.seconds, 0.5, accuracy: 0.001)
    }
    
    func testTickArithmetic() {
        let tick1 = Tick(value: 100)
        let tick2 = Tick(value: 200)
        
        XCTAssertEqual((tick1 + tick2).value, 300)
        XCTAssertEqual((tick2 - tick1).value, 100)
    }
    
    func testTickComparison() {
        let tick1 = Tick(value: 100)
        let tick2 = Tick(value: 200)
        
        XCTAssertTrue(tick1 < tick2)
        XCTAssertTrue(tick2 > tick1)
        XCTAssertTrue(tick1 <= tick1)
        XCTAssertTrue(tick1 >= tick1)
    }
    
    func testTickRange() {
        let start = Tick(value: 100)
        let end = Tick(value: 500)
        let range = TickRange(start: start, end: end)
        
        XCTAssertEqual(range.duration.value, 400)
        XCTAssertTrue(range.contains(Tick(value: 300)))
        XCTAssertFalse(range.contains(Tick(value: 600)))
    }
}