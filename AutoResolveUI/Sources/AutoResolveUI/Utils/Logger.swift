// AUTORESOLVE V3.0 - COMPREHENSIVE LOGGING SYSTEM
// Production-grade logging with levels, categories, and persistence

import Foundation
import os.log

/// Comprehensive logging system with categories and persistence
public class Logger {
    public static let shared = Logger()
    
    // Log levels
    public enum Level: Int, Comparable {
        case verbose = 0
        case debug = 1
        case info = 2
        case warning = 3
        case error = 4
        case critical = 5
        
        public static func < (lhs: Level, rhs: Level) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
        
        var emoji: String {
            switch self {
            case .verbose: return "🔍"
            case .debug: return "🐛"
            case .info: return "ℹ️"
            case .warning: return "⚠️"
            case .error: return "❌"
            case .critical: return "🚨"
            }
        }
        
        var osLogType: OSLogType {
            switch self {
            case .verbose, .debug: return .debug
            case .info: return .info
            case .warning: return .default
            case .error: return .error
            case .critical: return .fault
            }
        }
    }
    
    // Categories
    public enum Category: String {
        case general = "General"
        case ui = "UI"
        case network = "Network"
        case backend = "Backend"
        case pipeline = "Pipeline"
        case timeline = "Timeline"
        case media = "Media"
        case performance = "Performance"
        case storage = "Storage"
        case websocket = "WebSocket"
        
        var osLog: OSLog {
            OSLog(subsystem: "com.autoresolve.app", category: rawValue)
        }
    }
    
    // Configuration
    public var minimumLevel: Level = .debug
    public var enableConsoleLogging = true
    public var enableFileLogging = true
    public var enableOSLogging = true
    public var maxLogFileSize: Int = 10 * 1024 * 1024 // 10MB
    
    // File logging
    private let logFileURL: URL
    private let logQueue = DispatchQueue(label: "logger.queue", qos: .utility)
    private var logFileHandle: FileHandle?
    
    // Performance tracking
    private var performanceMarkers: [String: Date] = [:]
    
    private init() {
        // Setup log file
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let logsDirectory = documentsPath.appendingPathComponent("Logs")
        
        try? FileManager.default.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        let fileName = "autoresolve-\(dateFormatter.string(from: Date())).log"
        
        logFileURL = logsDirectory.appendingPathComponent(fileName)
        
        setupFileHandle()
        rotateLogsIfNeeded()
    }
    
    // MARK: - Public Logging Methods
    
    public func verbose(_ message: String, category: Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .verbose, category: category, file: file, function: function, line: line)
    }
    
    public func debug(_ message: String, category: Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .debug, category: category, file: file, function: function, line: line)
    }
    
    public func info(_ message: String, category: Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .info, category: category, file: file, function: function, line: line)
    }
    
    public func warning(_ message: String, category: Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, level: .warning, category: category, file: file, function: function, line: line)
    }
    
    public func error(_ message: String, error: Error? = nil, category: Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
        var fullMessage = message
        if let error = error {
            fullMessage += " | Error: \(error.localizedDescription)"
        }
        log(fullMessage, level: .error, category: category, file: file, function: function, line: line)
    }
    
    public func critical(_ message: String, error: Error? = nil, category: Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
        var fullMessage = message
        if let error = error {
            fullMessage += " | Error: \(error.localizedDescription)"
        }
        log(fullMessage, level: .critical, category: category, file: file, function: function, line: line)
    }
    
    // MARK: - Performance Logging
    
    public func startPerformanceMarker(_ identifier: String) {
        performanceMarkers[identifier] = Date()
        verbose("Performance marker started: \(identifier)", category: .performance)
    }
    
    public func endPerformanceMarker(_ identifier: String) {
        guard let startTime = performanceMarkers[identifier] else {
            warning("Performance marker not found: \(identifier)", category: .performance)
            return
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        performanceMarkers.removeValue(forKey: identifier)
        
        let message = String(format: "Performance marker '\(identifier)' completed in %.3f seconds", elapsed)
        
        if elapsed > 1.0 {
            warning(message, category: .performance)
        } else {
            info(message, category: .performance)
        }
    }
    
    // MARK: - Network Logging
    
    public func logNetworkRequest(_ request: URLRequest) {
        var message = "→ \(request.httpMethod ?? "GET") \(request.url?.absoluteString ?? "unknown")"
        
        if let headers = request.allHTTPHeaderFields, !headers.isEmpty {
            message += " | Headers: \(headers.count)"
        }
        
        if let body = request.httpBody {
            message += " | Body: \(body.count) bytes"
        }
        
        debug(message, category: .network)
    }
    
    public func logNetworkResponse(_ response: URLResponse?, data: Data?, error: Error?) {
        if let error = error {
            self.error("← Network error: \(error.localizedDescription)", category: .network)
            return
        }
        
        guard let httpResponse = response as? HTTPURLResponse else {
            warning("← Invalid response type", category: .network)
            return
        }
        
        let statusEmoji = httpResponse.statusCode < 300 ? "✅" : "⚠️"
        var message = "← \(statusEmoji) \(httpResponse.statusCode) \(httpResponse.url?.absoluteString ?? "unknown")"
        
        if let data = data {
            message += " | \(data.count) bytes"
        }
        
        if httpResponse.statusCode >= 400 {
            warning(message, category: .network)
        } else {
            debug(message, category: .network)
        }
    }
    
    // MARK: - Private Methods
    
    private func log(_ message: String, level: Level, category: Category, file: String, function: String, line: Int) {
        guard level >= minimumLevel else { return }
        
        let fileName = URL(fileURLWithPath: file).lastPathComponent
        let timestamp = ISO8601DateFormatter().string(from: Date())
        
        let logEntry = LogEntry(
            timestamp: timestamp,
            level: level,
            category: category,
            message: message,
            file: fileName,
            function: function,
            line: line
        )
        
        // Console logging
        if enableConsoleLogging {
            logToConsole(logEntry)
        }
        
        // OS logging
        if enableOSLogging {
            logToOS(logEntry)
        }
        
        // File logging
        if enableFileLogging {
            logToFile(logEntry)
        }
    }
    
    private func logToConsole(_ entry: LogEntry) {
        let formatted = "\(entry.level.emoji) [\(entry.category.rawValue)] \(entry.message) (\(entry.file):\(entry.line))"
        print(formatted)
    }
    
    private func logToOS(_ entry: LogEntry) {
        os_log("%{public}@", log: entry.category.osLog, type: entry.level.osLogType, entry.message)
    }
    
    private func logToFile(_ entry: LogEntry) {
        logQueue.async { [weak self] in
            guard let self = self else { return }
            
            let line = "\(entry.timestamp) [\(entry.level)] [\(entry.category.rawValue)] \(entry.message) | \(entry.file):\(entry.function):\(entry.line)\n"
            
            guard let data = line.data(using: .utf8) else { return }
            
            if self.logFileHandle == nil {
                self.setupFileHandle()
            }
            
            self.logFileHandle?.write(data)
        }
    }
    
    private func setupFileHandle() {
        if !FileManager.default.fileExists(atPath: logFileURL.path) {
            FileManager.default.createFile(atPath: logFileURL.path, contents: nil)
        }
        
        logFileHandle = try? FileHandle(forWritingTo: logFileURL)
        logFileHandle?.seekToEndOfFile()
    }
    
    private func rotateLogsIfNeeded() {
        logQueue.async { [weak self] in
            guard let self = self else { return }
            
            let attrs = try? FileManager.default.attributesOfItem(atPath: self.logFileURL.path)
            let fileSize = attrs?[.size] as? Int ?? 0
            
            if fileSize > self.maxLogFileSize {
                self.logFileHandle?.closeFile()
                self.logFileHandle = nil
                
                // Archive current log
                let archiveURL = self.logFileURL.appendingPathExtension("old")
                try? FileManager.default.removeItem(at: archiveURL)
                try? FileManager.default.moveItem(at: self.logFileURL, to: archiveURL)
                
                // Create new log file
                self.setupFileHandle()
            }
        }
    }
    
    // MARK: - Log Management
    
    public func exportLogs() -> URL? {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("autoresolve-logs.txt")
        try? FileManager.default.copyItem(at: logFileURL, to: tempURL)
        return tempURL
    }
    
    public func clearLogs() {
        logQueue.async { [weak self] in
            self?.logFileHandle?.closeFile()
            self?.logFileHandle = nil
            try? FileManager.default.removeItem(at: self?.logFileURL ?? URL(fileURLWithPath: ""))
            self?.setupFileHandle()
        }
    }
}

// MARK: - Supporting Types

struct LogEntry {
    let timestamp: String
    let level: Logger.Level
    let category: Logger.Category
    let message: String
    let file: String
    let function: String
    let line: Int
}

// MARK: - Global Convenience Functions

public func logVerbose(_ message: String, category: Logger.Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
    Logger.shared.verbose(message, category: category, file: file, function: function, line: line)
}

public func logDebug(_ message: String, category: Logger.Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
    Logger.shared.debug(message, category: category, file: file, function: function, line: line)
}

public func logInfo(_ message: String, category: Logger.Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
    Logger.shared.info(message, category: category, file: file, function: function, line: line)
}

public func logWarning(_ message: String, category: Logger.Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
    Logger.shared.warning(message, category: category, file: file, function: function, line: line)
}

public func logError(_ message: String, error: Error? = nil, category: Logger.Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
    Logger.shared.error(message, error: error, category: category, file: file, function: function, line: line)
}

public func logCritical(_ message: String, error: Error? = nil, category: Logger.Category = .general, file: String = #file, function: String = #function, line: Int = #line) {
    Logger.shared.critical(message, error: error, category: category, file: file, function: function, line: line)
}
