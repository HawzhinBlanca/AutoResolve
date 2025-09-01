import Foundation

public struct Tick: Comparable, Hashable, Codable {
    public let value: Int64
    
    public init(_ v: Int64) { 
        value = v 
    }
    
    public static func from(seconds: TimeInterval) -> Tick { 
        Tick(Int64(seconds * 1000)) 
    }
    
    public static func from(milliseconds: Int64) -> Tick {
        Tick(milliseconds)
    }
    
    public var seconds: TimeInterval { 
        Double(value) / 1000.0 
    }
    
    public var milliseconds: Int64 {
        value
    }
    
    public static func <(l: Tick, r: Tick) -> Bool { 
        l.value < r.value 
    }
    
    public static func +(l: Tick, r: Tick) -> Tick {
        Tick(l.value + r.value)
    }
    
    public static func -(l: Tick, r: Tick) -> Tick {
        Tick(l.value - r.value)
    }
    
    public static func *(l: Tick, r: Double) -> Tick {
        Tick(Int64(Double(l.value) * r))
    }
    
    public static let zero = Tick(0)
}

public struct TickRange: Hashable, Codable {
    public let start: Tick
    public let end: Tick
    
    public init(start: Tick, end: Tick) {
        self.start = start
        self.end = end
    }
    
    public var duration: Tick {
        end - start
    }
    
    public func contains(_ tick: Tick) -> Bool {
        tick >= start && tick <= end
    }
    
    public func overlaps(_ other: TickRange) -> Bool {
        start < other.end && end > other.start
    }
    
    public func union(_ other: TickRange) -> TickRange {
        TickRange(start: min(start, other.start), end: max(end, other.end))
    }
    
    public func intersection(_ other: TickRange) -> TickRange? {
        let newStart = max(start, other.start)
        let newEnd = min(end, other.end)
        guard newStart < newEnd else { return nil }
        return TickRange(start: newStart, end: newEnd)
    }
}

public struct Timebase {
    public let framesPerSecond: Double
    
    public init(fps: Double = 30.0) {
        self.framesPerSecond = fps
    }
    
    public func tickToFrame(_ tick: Tick) -> Int64 {
        Int64(tick.seconds * framesPerSecond)
    }
    
    public func frameToTick(_ frame: Int64) -> Tick {
        Tick.from(seconds: Double(frame) / framesPerSecond)
    }
    
    public func snapToFrame(_ tick: Tick) -> Tick {
        frameToTick(tickToFrame(tick))
    }
}