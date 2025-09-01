import Foundation
import SQLite

// MARK: - Timeline Events (Event Sourcing)

public struct Event: Codable {
    public let id: UUID
    public let seq: Int64
    public let timestamp: Date
    public let command: Command
    public let metadata: [String: String]
    
    public init(seq: Int64, command: Command, metadata: [String: String] = [:]) {
        self.id = UUID()
        self.seq = seq
        self.timestamp = Date()
        self.command = command
        self.metadata = metadata
    }
}

// MARK: - Event Store with SQLite WAL

public class EventStore {
    private let db: Connection
    private let events = Table("events")
    private let snapshots = Table("snapshots")
    
    // Event columns
    private let id = Expression<String>("id")
    private let seq = Expression<Int64>("seq")
    private let ts = Expression<Date>("ts")
    private let commandData = Expression<Data>("command_data")
    private let metadata = Expression<String?>("metadata")
    
    // Snapshot columns
    private let version = Expression<Int64>("version")
    private let snapshotData = Expression<Data>("snapshot_data")
    
    private var currentSeq: Int64 = 0
    private let snapshotInterval = 5000
    
    public init(path: String = "timeline.db") throws {
        // Enable WAL mode for better concurrency
        db = try Connection(path)
        try db.execute("PRAGMA journal_mode = WAL")
        try db.execute("PRAGMA synchronous = NORMAL")
        
        try createTables()
        currentSeq = try getMaxSeq()
    }
    
    private func createTables() throws {
        // Events table
        try db.run(events.create(ifNotExists: true) { t in
            t.column(id, primaryKey: true)
            t.column(seq, unique: true)
            t.column(ts)
            t.column(commandData)
            t.column(metadata)
        })
        
        // Snapshots table
        try db.run(snapshots.create(ifNotExists: true) { t in
            t.column(version, primaryKey: true)
            t.column(snapshotData)
            t.column(ts)
        })
        
        // Indexes
        try db.run(events.createIndex(seq, ifNotExists: true))
        try db.run(events.createIndex(ts, ifNotExists: true))
    }
    
    public func append(_ command: Command, metadata: [String: String] = [:]) throws -> Event {
        currentSeq += 1
        let event = Event(seq: currentSeq, command: command, metadata: metadata)
        
        let encoder = JSONEncoder()
        let commandData = try encoder.encode(command)
        let metadataJSON = try encoder.encode(metadata)
        
        let insert = events.insert(
            id <- event.id.uuidString,
            seq <- event.seq,
            ts <- event.timestamp,
            self.commandData <- commandData,
            self.metadata <- String(data: metadataJSON, encoding: .utf8)
        )
        
        try db.run(insert)
        
        // Create snapshot every N events
        if currentSeq % Int64(snapshotInterval) == 0 {
            try createSnapshot()
        }
        
        return event
    }
    
    public func replay(from: Int64 = 0, to: Int64? = nil) throws -> [Event] {
        let query = events
            .filter(seq >= from)
            .filter(to.map { seq <= $0 } ?? true)
            .order(seq.asc)
        
        let decoder = JSONDecoder()
        var result: [Event] = []
        
        for row in try db.prepare(query) {
            let command = try decoder.decode(Command.self, from: row[commandData])
            var eventMetadata: [String: String] = [:]
            
            if let metadataString = row[metadata],
               let data = metadataString.data(using: .utf8) {
                eventMetadata = (try? decoder.decode([String: String].self, from: data)) ?? [:]
            }
            
            let event = Event(
                seq: row[seq],
                command: command,
                metadata: eventMetadata
            )
            result.append(event)
        }
        
        return result
    }
    
    private func createSnapshot() throws {
        // Implement snapshot creation
        let snapshotData = try JSONEncoder().encode(currentSeq)
        
        let insert = snapshots.insert(
            version <- currentSeq,
            self.snapshotData <- snapshotData,
            ts <- Date()
        )
        
        try db.run(insert)
    }
    
    public func compact() throws {
        // Snapshot-safe compaction (from blueprint)
        let sql = """
            WITH s AS (SELECT COALESCE(MAX(version),0) v FROM snapshots)
            DELETE FROM events WHERE seq <= (SELECT v FROM s)
                                AND ts < strftime('%s','now','-90 day');
        """
        try db.execute(sql)
    }
    
    private func getMaxSeq() throws -> Int64 {
        return try db.scalar(events.select(seq.max)) ?? 0
    }
}