// AUTORESOLVE V3.0 - COMMON PATTERNS
// Reusable patterns and helpers for consistent code

import Foundation
import SwiftUI
import Combine
import AVFoundation

// MARK: - Async Helpers

/// Execute work on background thread and return to main
public func performBackgroundWork<T>(
    priority: _Concurrency.TaskPriority = .userInitiated,
    _ work: @escaping () async throws -> T
) async throws -> T {
    try await Task.detached(priority: priority) {
        try await work()
    }.value
}

/// Execute work with timeout
public func withTimeout<T>(
    seconds: TimeInterval,
    operation: @escaping () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask {
            try await operation()
        }
        
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw AutoResolveUILib.AppNetworkError.timeout
        }
        
        guard let result = try await group.next() else {
            throw AutoResolveUILib.AppNetworkError.timeout
        }
        
        group.cancelAll()
        return result
    }
}

// MARK: - Thread Safety

/// Thread-safe value wrapper
@propertyWrapper
public final class ThreadSafe<Value> {
    private var value: Value
    private let queue = DispatchQueue(label: "ThreadSafe", attributes: .concurrent)
    
    public init(wrappedValue: Value) {
        self.value = wrappedValue
    }
    
    public var wrappedValue: Value {
        get {
            queue.sync { value }
        }
        set {
            queue.async(flags: .barrier) {
                self.value = newValue
            }
        }
    }
    
    public func mutate(_ mutation: @escaping (inout Value) -> Void) {
        queue.async(flags: .barrier) {
            mutation(&self.value)
        }
    }
}

// MARK: - Debouncing

/// Debounce publisher for search and rapid updates
public extension Publisher where Failure == Never {
    func debounce(for seconds: TimeInterval) -> AnyPublisher<Output, Never> {
        self.debounce(for: .seconds(seconds), scheduler: RunLoop.main)
            .eraseToAnyPublisher()
    }
}

/// Debounced action executor
public class DebouncedAction {
    private var workItem: DispatchWorkItem?
    private let queue: DispatchQueue
    private let delay: TimeInterval
    
    public init(delay: TimeInterval, queue: DispatchQueue = .main) {
        self.delay = delay
        self.queue = queue
    }
    
    public func execute(_ action: @escaping () -> Void) {
        workItem?.cancel()
        
        let item = DispatchWorkItem(block: action)
        workItem = item
        
        queue.asyncAfter(deadline: .now() + delay, execute: item)
    }
    
    public func cancel() {
        workItem?.cancel()
        workItem = nil
    }
}

// MARK: - Result Builders

/// Result builder for creating validation rules
@resultBuilder
public struct ValidationBuilder {
    public static func buildBlock(_ components: ValidationRule...) -> [ValidationRule] {
        components
    }
}

public struct ValidationRule {
    let validate: () -> Bool
    let error: String
    
    public init(_ error: String, validate: @escaping () -> Bool) {
        self.error = error
        self.validate = validate
    }
}

public func validate(@ValidationBuilder _ builder: () -> [ValidationRule]) -> AutoResolveUILib.AppValidationResult {
    let rules = builder()
    let errors = rules.compactMap { rule in
        rule.validate() ? nil : AutoResolveUILib.AppValidationError.invalidFormat(field: "validation", expected: rule.error)
    }
    
    return errors.isEmpty ? .valid : .invalid(errors)
}

// MARK: - File Operations

/// Safe file operation with automatic cleanup
public func withTemporaryFile<T>(
    extension ext: String = "tmp",
    _ operation: (URL) async throws -> T
) async throws -> T {
    let tempURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension(ext)
    
    defer {
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    return try await operation(tempURL)
}

/// Safe directory operation with cleanup
public func withTemporaryDirectory<T>(
    _ operation: (URL) async throws -> T
) async throws -> T {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
    
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    
    defer {
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    return try await operation(tempDir)
}

// MARK: - Resource Management

/// Auto-release resource wrapper
public class Resource<T> {
    private let resource: T
    private let cleanup: (T) -> Void
    
    public init(_ resource: T, cleanup: @escaping (T) -> Void) {
        self.resource = resource
        self.cleanup = cleanup
    }
    
    deinit {
        cleanup(resource)
    }
    
    public func use<R>(_ operation: (T) throws -> R) rethrows -> R {
        try operation(resource)
    }
}

// MARK: - Retry Logic

/// Retry operation with exponential backoff
public func retry<T>(
    maxAttempts: Int = 3,
    delay: TimeInterval = 1.0,
    multiplier: Double = 2.0,
    _ operation: @escaping () async throws -> T
) async throws -> T {
    var currentDelay = delay
    var lastError: Error?
    
    for attempt in 1...maxAttempts {
        do {
            return try await operation()
        } catch {
            lastError = error
            
            // Check if error is retryable
            if let autoResolveError = error as? AutoResolveUILib.AutoResolveError,
               !autoResolveError.isRetryable {
                throw error
            }
            
            if attempt < maxAttempts {
                try await Task.sleep(nanoseconds: UInt64(currentDelay * 1_000_000_000))
                currentDelay *= multiplier
            }
        }
    }
    
    throw lastError ?? AutoResolveUILib.AutoResolveError.network(.timeout)
}

// MARK: - Batch Processing

/// Process items in batches
public func processBatch<T, R>(
    items: [T],
    batchSize: Int = 10,
    process: @escaping ([T]) async throws -> [R]
) async throws -> [R] {
    var results: [R] = []
    
    for index in stride(from: 0, to: items.count, by: batchSize) {
        let endIndex = min(index + batchSize, items.count)
        let batch = Array(items[index..<endIndex])
        let batchResults = try await process(batch)
        results.append(contentsOf: batchResults)
    }
    
    return results
}

/// Process items concurrently with limit
public func processConcurrently<T, R>(
    items: [T],
    maxConcurrency: Int = 4,
    process: @escaping (T) async throws -> R
) async throws -> [R] {
    try await withThrowingTaskGroup(of: (Int, R).self) { group in
        var results: [R?] = Array(repeating: nil, count: items.count)
        
        // Add initial tasks
        for (index, item) in items.prefix(maxConcurrency).enumerated() {
            group.addTask {
                let result = try await process(item)
                return (index, result)
            }
        }
        
        var nextIndex = maxConcurrency
        
        // Process results and add new tasks
        for try await (index, result) in group {
            results[index] = result
            
            if nextIndex < items.count {
                let itemIndex = nextIndex
                let item = items[itemIndex]
                nextIndex += 1
                
                group.addTask {
                    let result = try await process(item)
                    return (itemIndex, result)
                }
            }
        }
        
        return results.compactMap { $0 }
    }
}

// MARK: - Observable Helpers

/// Combine latest values from multiple publishers
public func combineLatest<T1, T2, T3>(
    _ p1: AnyPublisher<T1, Never>,
    _ p2: AnyPublisher<T2, Never>,
    _ p3: AnyPublisher<T3, Never>
) -> AnyPublisher<(T1, T2, T3), Never> {
    Publishers.CombineLatest3(p1, p2, p3)
        .eraseToAnyPublisher()
}

// MARK: - SwiftUI Helpers

/// Conditional view modifier
public extension View {
    @ViewBuilder
    func `if`<Content: View>(_ condition: Bool, transform: (Self) -> Content) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }
    
    func onAppearAsync(perform action: @escaping () async -> Void) -> some View {
        self.onAppear {
            Task {
                await action()
            }
        }
    }
}

// MARK: - Collection Helpers

public extension Collection {
    /// Safe subscript access
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
    
    /// Chunk collection into smaller arrays
    func chunked(size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self.dropFirst($0).prefix(size))
        }
    }
}

public extension Array {
    /// Remove duplicates while preserving order
    func removingDuplicates<T: Hashable>(by keyPath: KeyPath<Element, T>) -> [Element] {
        var seen = Set<T>()
        return filter { element in
            let key = element[keyPath: keyPath]
            guard !seen.contains(key) else { return false }
            seen.insert(key)
            return true
        }
    }
}

// MARK: - URL Helpers

public extension URL {
    /// Check if URL is a directory
    var isDirectory: Bool {
        (try? resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
    }
    
    /// Get file size
    var fileSize: Int64? {
        (try? resourceValues(forKeys: [.fileSizeKey]))?.fileSize.map { Int64($0) }
    }
    
    /// Get creation date
    var creationDate: Date? {
        (try? resourceValues(forKeys: [.creationDateKey]))?.creationDate
    }
    
    /// Get modification date
    var modificationDate: Date? {
        (try? resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
    }
}

// MARK: - String Helpers

public extension String {
    /// Truncate string with ellipsis
    func truncated(to length: Int, addEllipsis: Bool = true) -> String {
        guard count > length else { return self }
        
        let truncated = String(prefix(length))
        return addEllipsis ? truncated + "..." : truncated
    }
    
    /// Convert to safe filename
    var safeFilename: String {
        let invalidCharacters = CharacterSet(charactersIn: "/\\?%*|\"<>:")
        return components(separatedBy: invalidCharacters).joined(separator: "_")
    }
}

// MARK: - Date Helpers

public extension Date {
    /// Check if date is today
    var isToday: Bool {
        Calendar.current.isDateInToday(self)
    }
    
    /// Check if date is yesterday
    var isYesterday: Bool {
        Calendar.current.isDateInYesterday(self)
    }
    
    /// Get time interval until now
    var timeIntervalUntilNow: TimeInterval {
        -timeIntervalSinceNow
    }
}

// MARK: - CMTime Helpers

public extension CMTime {
    /// Convert to TimeInterval safely
    var timeInterval: TimeInterval? {
        guard isValid && isNumeric else { return nil }
        let seconds = CMTimeGetSeconds(self)
        guard seconds.isFinite else { return nil }
        return seconds
    }
    
    /// Create from TimeInterval
    static func from(seconds: TimeInterval, preferredTimescale: CMTimeScale = 600) -> CMTime {
        CMTime(seconds: seconds, preferredTimescale: preferredTimescale)
    }
}