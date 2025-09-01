import XCTest
@testable import AutoResolveCore

final class EventStoreTests: XCTestCase {
    var store: EventStore!
    
    override func setUp() {
        super.setUp()
        store = try! EventStore(path: ":memory:")
    }
    
    func testAddAndRetrieveEvent() throws {
        let event = TimelineEvent(
            type: .cut,
            range: TickRange(start: Tick(1000), end: Tick(2000)),
            metadata: ["label": "Cut 1"]
        )
        
        try store.add(event)
        
        let events = try store.allEvents()
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events.first?.id, event.id)
        XCTAssertEqual(events.first?.type, .cut)
        XCTAssertEqual(events.first?.metadata["label"], "Cut 1")
    }
    
    func testEventsInRange() throws {
        let events = [
            TimelineEvent(type: .cut, range: TickRange(start: Tick(0), end: Tick(1000))),
            TimelineEvent(type: .cut, range: TickRange(start: Tick(2000), end: Tick(3000))),
            TimelineEvent(type: .cut, range: TickRange(start: Tick(4000), end: Tick(5000)))
        ]
        
        for event in events {
            try store.add(event)
        }
        
        let range = TickRange(start: Tick(1500), end: Tick(3500))
        let foundEvents = try store.events(in: range)
        
        XCTAssertEqual(foundEvents.count, 1)
        XCTAssertEqual(foundEvents.first?.range.start.value, 2000)
    }
    
    func testEventsByType() throws {
        try store.add(TimelineEvent(type: .cut, range: TickRange(start: Tick(0), end: Tick(1000))))
        try store.add(TimelineEvent(type: .silence, range: TickRange(start: Tick(1000), end: Tick(2000))))
        try store.add(TimelineEvent(type: .cut, range: TickRange(start: Tick(2000), end: Tick(3000))))
        
        let cuts = try store.events(ofType: .cut)
        let silences = try store.events(ofType: .silence)
        
        XCTAssertEqual(cuts.count, 2)
        XCTAssertEqual(silences.count, 1)
    }
    
    func testRemoveEvent() throws {
        let event = TimelineEvent(type: .cut, range: TickRange(start: Tick(0), end: Tick(1000)))
        try store.add(event)
        
        var events = try store.allEvents()
        XCTAssertEqual(events.count, 1)
        
        try store.remove(event.id)
        
        events = try store.allEvents()
        XCTAssertEqual(events.count, 0)
    }
    
    func testClearStore() throws {
        for i in 0..<5 {
            let event = TimelineEvent(
                type: .cut,
                range: TickRange(start: Tick(Int64(i * 1000)), end: Tick(Int64((i + 1) * 1000)))
            )
            try store.add(event)
        }
        
        var events = try store.allEvents()
        XCTAssertEqual(events.count, 5)
        
        try store.clear()
        
        events = try store.allEvents()
        XCTAssertEqual(events.count, 0)
    }
    
    func testTransaction() throws {
        var events: [TimelineEvent] = []
        for i in 0..<3 {
            let event = TimelineEvent(
                type: .cut,
                range: TickRange(start: Tick(Int64(i * 1000)), end: Tick(Int64((i + 1) * 1000)))
            )
            events.append(event)
        }
        
        try store.transaction {
            for event in events {
                try store.add(event)
            }
        }
        
        let storedEvents = try store.allEvents()
        XCTAssertEqual(storedEvents.count, 3)
    }
    
    func testEventOrdering() throws {
        let events = [
            TimelineEvent(type: .cut, range: TickRange(start: Tick(3000), end: Tick(4000))),
            TimelineEvent(type: .cut, range: TickRange(start: Tick(1000), end: Tick(2000))),
            TimelineEvent(type: .cut, range: TickRange(start: Tick(5000), end: Tick(6000)))
        ]
        
        for event in events {
            try store.add(event)
        }
        
        let ordered = try store.allEvents()
        
        XCTAssertEqual(ordered[0].range.start.value, 1000)
        XCTAssertEqual(ordered[1].range.start.value, 3000)
        XCTAssertEqual(ordered[2].range.start.value, 5000)
    }
}