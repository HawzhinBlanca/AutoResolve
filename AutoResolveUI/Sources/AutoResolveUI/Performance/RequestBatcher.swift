// AUTORESOLVE V3.0 - REQUEST BATCHER
// Intelligent request batching to reduce backend calls

import Foundation
import Combine
import os.log

// MARK: - Request Batcher
@MainActor
final class RequestBatcher {
    static let shared = RequestBatcher()
    
    // Configuration
    private let debounceInterval: TimeInterval = 0.05 // 50ms
    private let maxBatchSize = 100
    private let flushInterval: TimeInterval = 1.0 // Auto-flush every second
    
    // Request queues by priority
    private var immediateQueue: [BatchableRequest] = []
    private var highQueue: [BatchableRequest] = []
    private var normalQueue: [BatchableRequest] = []
    private var lowQueue: [BatchableRequest] = []
    
    // Batching state
    private var pendingBatches: [String: RequestBatch] = [:]
    private var debounceTimers: [String: Timer] = [:]
    private var flushTimer: Timer?
    
    // Statistics
    @Published var totalRequests = 0
    @Published var batchedRequests = 0
    @Published var sentBatches = 0
    @Published var averageBatchSize: Double = 0
    @Published var networkCalls = 0
    @Published var savedCalls = 0
    
    private let logger = Logger.shared
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Request Priority
    enum Priority: Int, Comparable {
        case immediate = 0
        case high = 1
        case normal = 2
        case low = 3
        
        static func < (lhs: Priority, rhs: Priority) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }
    
    // MARK: - Initialization
    private init() {
        startFlushTimer()
    }
    
    // MARK: - Request Submission
    func submit<T: BatchableRequest>(_ request: T, priority: Priority = .normal) {
        totalRequests += 1
        
        // Immediate requests bypass batching
        if priority == .immediate {
            Task {
                await executeImmediate(request)
            }
            return
        }
        
        // Add to appropriate queue
        enqueue(request, priority: priority)
        
        // Check if we should batch
        let batchKey = request.batchKey
        if let existingBatch = pendingBatches[batchKey] {
            existingBatch.add(request)
            
            // Check batch size limit
            if existingBatch.requests.count >= maxBatchSize {
                Task {
                    await flushBatch(batchKey)
                }
            }
        } else {
            // Create new batch
            let batch = RequestBatch(key: batchKey)
            batch.add(request)
            pendingBatches[batchKey] = batch
            
            // Start debounce timer
            startDebounceTimer(for: batchKey)
        }
        
        logger.debug("Request submitted: \(request.identifier) with priority \(String(describing: priority))")
    }
    
    // MARK: - Request Coalescing
    func coalesce<T: CoalescableRequest>(_ requests: [T]) -> [T] {
        var coalescedRequests: [String: T] = [:]
        
        for request in requests {
            let key = request.coalescingKey
            if let existing = coalescedRequests[key] {
                // Merge with existing request
                let merged = existing.merge(with: request)
                coalescedRequests[key] = merged
            } else {
                coalescedRequests[key] = request
            }
        }
        
        let result = Array(coalescedRequests.values)
        if result.count < requests.count {
            savedCalls += requests.count - result.count
            logger.info("Coalesced \(requests.count) requests into \(result.count)")
        }
        
        return result
    }
    
    // MARK: - Queue Management
    private func enqueue(_ request: BatchableRequest, priority: Priority) {
        switch priority {
        case .immediate:
            immediateQueue.append(request)
        case .high:
            highQueue.append(request)
        case .normal:
            normalQueue.append(request)
        case .low:
            lowQueue.append(request)
        }
    }
    
    private func dequeueNext() -> BatchableRequest? {
        if !immediateQueue.isEmpty {
            return immediateQueue.removeFirst()
        } else if !highQueue.isEmpty {
            return highQueue.removeFirst()
        } else if !normalQueue.isEmpty {
            return normalQueue.removeFirst()
        } else if !lowQueue.isEmpty {
            return lowQueue.removeFirst()
        }
        return nil
    }
    
    // MARK: - Batch Execution
    private func executeImmediate(_ request: BatchableRequest) async {
        networkCalls += 1
        
        do {
            try await request.execute()
            logger.debug("Immediate request executed: \(request.identifier)")
        } catch {
            logger.error("Failed to execute immediate request: \(error)")
            await request.handleError(error)
        }
    }
    
    private func flushBatch(_ batchKey: String) async {
        guard let batch = pendingBatches.removeValue(forKey: batchKey) else { return }
        
        // Cancel debounce timer
        debounceTimers[batchKey]?.invalidate()
        debounceTimers.removeValue(forKey: batchKey)
        
        // Coalesce if possible
        let requests = batch.requests
        let finalRequests: [BatchableRequest]
        
        // Check if requests can be coalesced
        finalRequests = requests
        
        // Execute batch
        if finalRequests.count == 1 {
            // Single request - execute directly
            networkCalls += 1
            do {
                try await finalRequests[0].execute()
            } catch {
                await finalRequests[0].handleError(error)
            }
        } else {
            // Multiple requests - execute as batch
            await executeBatch(finalRequests)
        }
        
        // Update statistics
        batchedRequests += requests.count
        sentBatches += 1
        updateAverageBatchSize()
        
        logger.info("Batch flushed: \(batchKey) with \(finalRequests.count) requests")
    }
    
    private func executeBatch(_ requests: [BatchableRequest]) async {
        networkCalls += 1
        
        // Group requests by endpoint
        let grouped = Dictionary(grouping: requests) { $0.endpoint }
        
        for (endpoint, endpointRequests) in grouped {
            // Create batch request
            let batchRequest = BatchRequest(
                endpoint: endpoint,
                requests: endpointRequests
            )
            
            do {
                let responses = try await batchRequest.execute()
                
                // Distribute responses
                for (index, request) in endpointRequests.enumerated() {
                    if index < responses.count {
                        await request.handleResponse(responses[index])
                    }
                }
            } catch {
                // Handle batch error
                for request in endpointRequests {
                    await request.handleError(error)
                }
            }
        }
    }
    
    // MARK: - Timers
    private func startDebounceTimer(for batchKey: String) {
        debounceTimers[batchKey]?.invalidate()
        
        debounceTimers[batchKey] = Timer.scheduledTimer(withTimeInterval: debounceInterval, repeats: false) { _ in
            Task { @MainActor in
                await self.flushBatch(batchKey)
            }
        }
    }
    
    private func startFlushTimer() {
        flushTimer = Timer.scheduledTimer(withTimeInterval: flushInterval, repeats: true) { _ in
            Task { @MainActor in
                await self.flushAllBatches()
            }
        }
    }
    
    private func flushAllBatches() async {
        let batchKeys = Array(pendingBatches.keys)
        
        for key in batchKeys {
            await flushBatch(key)
        }
        
        logger.debug("Auto-flush completed: \(batchKeys.count) batches")
    }
    
    // MARK: - Statistics
    private func updateAverageBatchSize() {
        guard sentBatches > 0 else { return }
        averageBatchSize = Double(batchedRequests) / Double(sentBatches)
    }
    
    func resetStatistics() {
        totalRequests = 0
        batchedRequests = 0
        sentBatches = 0
        averageBatchSize = 0
        networkCalls = 0
        savedCalls = 0
    }
    
    var efficiency: Double {
        guard totalRequests > 0 else { return 0 }
        return Double(savedCalls) / Double(totalRequests)
    }
    
    var compressionRatio: Double {
        guard batchedRequests > 0 else { return 1 }
        return Double(batchedRequests) / Double(networkCalls)
    }
}

// MARK: - Request Protocols
protocol BatchableRequest {
    var identifier: String { get }
    var batchKey: String { get }
    var endpoint: String { get }
    
    func execute() async throws
    func handleResponse(_ response: Any) async
    func handleError(_ error: Error) async
}

protocol CoalescableRequest: BatchableRequest {
    var coalescingKey: String { get }
    func merge(with other: Self) -> Self
}

// MARK: - Request Batch
private class RequestBatch {
    let key: String
    private(set) var requests: [BatchableRequest] = []
    let createdAt = Date()
    
    init(key: String) {
        self.key = key
    }
    
    func add(_ request: BatchableRequest) {
        requests.append(request)
    }
    
    var age: TimeInterval {
        Date().timeIntervalSince(createdAt)
    }
}

// MARK: - Batch Request
private struct BatchRequest {
    let endpoint: String
    let requests: [BatchableRequest]
    
    func execute() async throws -> [Any] {
        // Create combined request body
        let requestData = createBatchRequestData()
        
        // Execute batch API call
        let response = try await performBatchAPICall(requestData)
        
        // Parse responses
        return parseBatchResponse(response)
    }
    
    private func createBatchRequestData() -> Data {
        // Combine individual requests into batch format
        var batchData: [[String: Any]] = []
        
        for request in requests {
            // Extract request data (implementation depends on request type)
            batchData.append([
                "id": request.identifier,
                "endpoint": request.endpoint
                // Add request-specific data
            ])
        }
        
        return try! JSONSerialization.data(withJSONObject: batchData)
    }
    
    private func performBatchAPICall(_ data: Data) async throws -> Data {
        // Perform actual network call
        // This is a placeholder - integrate with actual backend
        let url = URL(string: "http://localhost:8000/api/batch")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = data
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let (responseData, _) = try await URLSession.shared.data(for: request)
        return responseData
    }
    
    private func parseBatchResponse(_ data: Data) -> [Any] {
        // Parse batch response into individual responses
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return []
        }
        
        return json
    }
}

// MARK: - Example Request Types
struct MetadataRequest: CoalescableRequest {
    let fileURLs: Set<URL>
    
    var identifier: String {
        fileURLs.map { $0.lastPathComponent }.joined(separator: ",")
    }
    
    var batchKey: String { "metadata" }
    var endpoint: String { "/api/metadata" }
    var coalescingKey: String { "metadata-batch" }
    
    func merge(with other: MetadataRequest) -> MetadataRequest {
        MetadataRequest(fileURLs: fileURLs.union(other.fileURLs))
    }
    
    func execute() async throws {
        // Execute metadata fetch
    }
    
    func handleResponse(_ response: Any) async {
        // Handle metadata response
    }
    
    func handleError(_ error: Error) async {
        // Handle error
    }
}

struct ThumbnailRequest: BatchableRequest {
    let fileURL: URL
    let size: CGSize
    
    var identifier: String { fileURL.lastPathComponent }
    var batchKey: String { "thumbnail-\(Int(size.width))x\(Int(size.height))" }
    var endpoint: String { "/api/thumbnail" }
    
    func execute() async throws {
        // Execute thumbnail generation
    }
    
    func handleResponse(_ response: Any) async {
        // Handle thumbnail response
    }
    
    func handleError(_ error: Error) async {
        // Handle error
    }
}

// MARK: - Request Builder
@MainActor
class RequestBuilder {
    private let batcher = RequestBatcher.shared
    
    func fetchMetadata(for urls: [URL], priority: RequestBatcher.Priority = .normal) {
        let request = MetadataRequest(fileURLs: Set(urls))
        batcher.submit(request, priority: priority)
    }
    
    func generateThumbnail(for url: URL, size: CGSize, priority: RequestBatcher.Priority = .normal) {
        let request = ThumbnailRequest(fileURL: url, size: size)
        batcher.submit(request, priority: priority)
    }
    
    func preloadData(for urls: [URL]) {
        // Batch preload requests
        for url in urls {
            fetchMetadata(for: [url], priority: .low)
            generateThumbnail(for: url, size: CGSize(width: 280, height: 160), priority: .low)
        }
    }
}
