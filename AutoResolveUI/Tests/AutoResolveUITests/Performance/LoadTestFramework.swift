//
//  LoadTestFramework.swift
//  AutoResolveUITests
//
//  Created by AutoResolve on 8/23/25.
//

import XCTest
import Foundation
import AVFoundation
import Combine
@testable import AutoResolveUI

/// Comprehensive load testing framework for AutoResolve enterprise performance validation
/// Tests system behavior under high load, concurrent users, and resource constraints
@MainActor
class LoadTestFramework: XCTestCase {
    
    // MARK: - Test Configuration
    
    private struct LoadTestConfig {
        let concurrentUsers: Int
        let testDuration: TimeInterval
        let operationsPerSecond: Double
        let memoryLimitMB: Int
        let cpuLimitPercent: Double
    }
    
    private let lightLoad = LoadTestConfig(
        concurrentUsers: 5,
        testDuration: 30.0,
        operationsPerSecond: 10.0,
        memoryLimitMB: 100,
        cpuLimitPercent: 50.0
    )
    
    private let mediumLoad = LoadTestConfig(
        concurrentUsers: 15,
        testDuration: 60.0,
        operationsPerSecond: 25.0,
        memoryLimitMB: 200,
        cpuLimitPercent: 70.0
    )
    
    private let heavyLoad = LoadTestConfig(
        concurrentUsers: 50,
        testDuration: 120.0,
        operationsPerSecond: 50.0,
        memoryLimitMB: 500,
        cpuLimitPercent: 85.0
    )
    
    // MARK: - Test Infrastructure
    
    private var performanceMonitor: PerformanceMonitor!
    private var loadTestResults: [LoadTestResult] = []
    private var cancellables: Set<AnyCancellable> = []
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        
        performanceMonitor = PerformanceMonitor()
        loadTestResults.removeAll()
        
        // Setup performance monitoring
        setupPerformanceMonitoring()
    }
    
    override func tearDownWithError() throws {
        // Generate load test report
        generateLoadTestReport()
        
        performanceMonitor = nil
        cancellables.removeAll()
        
        try super.tearDownWithError()
    }
    
    private func setupPerformanceMonitoring() {
        performanceMonitor.startMonitoring()
    }
    
    // MARK: - Database Load Tests
    
    func testDatabaseLoadLightLoad() async throws {
        try await runDatabaseLoadTest(config: lightLoad)
    }
    
    func testDatabaseLoadMediumLoad() async throws {
        try await runDatabaseLoadTest(config: mediumLoad)
    }
    
    func testDatabaseLoadHeavyLoad() async throws {
        try await runDatabaseLoadTest(config: heavyLoad)
    }
    
    private func runDatabaseLoadTest(config: LoadTestConfig) async throws {
        let expectation = expectation(description: "Database load test - \(config.concurrentUsers) users")
        expectation.expectedFulfillmentCount = config.concurrentUsers
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let databaseManager = DatabaseManager.shared
        let testResults = ThreadSafeArray<DatabaseOperationResult>()
        
        // Launch concurrent database operations
        for userId in 0..<config.concurrentUsers {
            Task {
                do {
                    let userResults = try await performDatabaseOperations(
                        userId: userId,
                        duration: config.testDuration,
                        operationsPerSecond: config.operationsPerSecond
                    )
                    testResults.append(contentsOf: userResults)
                    expectation.fulfill()
                } catch {
                    XCTFail("Database load test failed for user \(userId): \(error)")
                    expectation.fulfill()
                }
            }
        }
        
        await fulfillment(of: [expectation], timeout: config.testDuration + 30.0)
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let totalDuration = endTime - startTime
        
        // Analyze results
        let results = testResults.elements
        let successfulOperations = results.filter { $0.success }.count
        let failedOperations = results.filter { !$0.success }.count
        let averageResponseTime = results.map { $0.responseTime }.reduce(0, +) / Double(results.count)
        let operationsPerSecond = Double(results.count) / totalDuration
        
        // Record test results
        let loadTestResult = LoadTestResult(
            testName: "Database Load Test",
            configuration: config,
            totalOperations: results.count,
            successfulOperations: successfulOperations,
            failedOperations: failedOperations,
            averageResponseTime: averageResponseTime,
            operationsPerSecond: operationsPerSecond,
            memoryUsage: performanceMonitor.getCurrentMemoryUsage(),
            cpuUsage: performanceMonitor.getCurrentCPUUsage()
        )
        
        loadTestResults.append(loadTestResult)
        
        // Assertions
        XCTAssertGreaterThan(successfulOperations, results.count * 95 / 100, "Should have >95% success rate")
        XCTAssertLessThan(averageResponseTime, 1.0, "Average response time should be <1s")
        XCTAssertLessThan(loadTestResult.memoryUsage, Double(config.memoryLimitMB * 1024 * 1024), "Memory usage within limits")
        XCTAssertLessThan(loadTestResult.cpuUsage, config.cpuLimitPercent, "CPU usage within limits")
    }
    
    private func performDatabaseOperations(userId: Int, duration: TimeInterval, operationsPerSecond: Double) async throws -> [DatabaseOperationResult] {
        let databaseManager = DatabaseManager.shared
        var results: [DatabaseOperationResult] = []
        
        let endTime = Date().addingTimeInterval(duration)
        let operationInterval = 1.0 / operationsPerSecond
        
        while Date() < endTime {
            let operationStart = CFAbsoluteTimeGetCurrent()
            
            do {
                // Mix of different database operations
                let operation = Int.random(in: 0...4)
                
                switch operation {
                case 0:
                    // Create project
                    let project = VideoProject(name: "Load Test Project \(userId)-\(Date().timeIntervalSince1970)")
                    try await databaseManager.saveProject(project)
                    
                case 1:
                    // Load projects
                    _ = try await databaseManager.loadAllProjects()
                    
                case 2:
                    // Update project
                    let projects = try await databaseManager.loadAllProjects()
                    if let project = projects.first {
                        project.description = "Updated at \(Date())"
                        try await databaseManager.saveProject(project)
                    }
                    
                case 3:
                    // Create and save media item
                    let mediaItem = MediaItem(url: createTestMediaURL(), type: .video)
                    try await databaseManager.saveMediaItem(mediaItem)
                    
                case 4:
                    // Load media items
                    _ = try await databaseManager.loadAllMediaItems()
                }
                
                let operationEnd = CFAbsoluteTimeGetCurrent()
                let responseTime = operationEnd - operationStart
                
                results.append(DatabaseOperationResult(
                    operation: "database_operation",
                    success: true,
                    responseTime: responseTime,
                    userId: userId
                ))
                
            } catch {
                let operationEnd = CFAbsoluteTimeGetCurrent()
                let responseTime = operationEnd - operationStart
                
                results.append(DatabaseOperationResult(
                    operation: "database_operation",
                    success: false,
                    responseTime: responseTime,
                    userId: userId,
                    error: error.localizedDescription
                ))
            }
            
            // Wait for next operation
            try await Task.sleep(nanoseconds: UInt64(operationInterval * 1_000_000_000))
        }
        
        return results
    }
    
    // MARK: - Media Processing Load Tests
    
    func testMediaProcessingLoadLightLoad() async throws {
        try await runMediaProcessingLoadTest(config: lightLoad)
    }
    
    func testMediaProcessingLoadMediumLoad() async throws {
        try await runMediaProcessingLoadTest(config: mediumLoad)
    }
    
    func testMediaProcessingLoadHeavyLoad() async throws {
        try await runMediaProcessingLoadTest(config: heavyLoad)
    }
    
    private func runMediaProcessingLoadTest(config: LoadTestConfig) async throws {
        let expectation = expectation(description: "Media processing load test - \(config.concurrentUsers) users")
        expectation.expectedFulfillmentCount = config.concurrentUsers
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let testResults = ThreadSafeArray<MediaProcessingResult>()
        
        // Launch concurrent media processing operations
        for userId in 0..<config.concurrentUsers {
            Task {
                do {
                    let userResults = try await performMediaProcessingOperations(
                        userId: userId,
                        duration: config.testDuration,
                        operationsPerSecond: config.operationsPerSecond
                    )
                    testResults.append(contentsOf: userResults)
                    expectation.fulfill()
                } catch {
                    XCTFail("Media processing load test failed for user \(userId): \(error)")
                    expectation.fulfill()
                }
            }
        }
        
        await fulfillment(of: [expectation], timeout: config.testDuration + 60.0)
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let totalDuration = endTime - startTime
        
        // Analyze results
        let results = testResults.elements
        let successfulOperations = results.filter { $0.success }.count
        let failedOperations = results.filter { !$0.success }.count
        let averageProcessingTime = results.map { $0.processingTime }.reduce(0, +) / Double(results.count)
        
        // Record test results
        let loadTestResult = LoadTestResult(
            testName: "Media Processing Load Test",
            configuration: config,
            totalOperations: results.count,
            successfulOperations: successfulOperations,
            failedOperations: failedOperations,
            averageResponseTime: averageProcessingTime,
            operationsPerSecond: Double(results.count) / totalDuration,
            memoryUsage: performanceMonitor.getCurrentMemoryUsage(),
            cpuUsage: performanceMonitor.getCurrentCPUUsage()
        )
        
        loadTestResults.append(loadTestResult)
        
        // Assertions for media processing
        XCTAssertGreaterThan(successfulOperations, results.count * 90 / 100, "Should have >90% success rate for media processing")
        XCTAssertLessThan(averageProcessingTime, 5.0, "Average processing time should be <5s")
    }
    
    private func performMediaProcessingOperations(userId: Int, duration: TimeInterval, operationsPerSecond: Double) async throws -> [MediaProcessingResult] {
        let audioEngine = AudioEngine()
        var results: [MediaProcessingResult] = []
        
        let endTime = Date().addingTimeInterval(duration)
        let operationInterval = 1.0 / operationsPerSecond
        
        while Date() < endTime {
            let operationStart = CFAbsoluteTimeGetCurrent()
            
            do {
                let testMediaURL = createTestMediaURL()
                let operation = Int.random(in: 0...3)
                
                switch operation {
                case 0:
                    // Load audio
                    let success = await audioEngine.loadAudio(from: testMediaURL)
                    if !success { throw MediaProcessingError.loadFailed }
                    
                case 1:
                    // Generate waveform
                    _ = try await audioEngine.generateWaveformData()
                    
                case 2:
                    // Detect silence
                    _ = try await audioEngine.detectSilence(threshold: -40.0, minimumDuration: 0.5)
                    
                case 3:
                    // Generate thumbnail
                    let cache = OptimizedCache.shared
                    _ = try await cache.generateThumbnail(for: testMediaURL, at: 1.0)
                    
                default:
                    break
                }
                
                let operationEnd = CFAbsoluteTimeGetCurrent()
                let processingTime = operationEnd - operationStart
                
                results.append(MediaProcessingResult(
                    operation: "media_processing",
                    success: true,
                    processingTime: processingTime,
                    userId: userId
                ))
                
            } catch {
                let operationEnd = CFAbsoluteTimeGetCurrent()
                let processingTime = operationEnd - operationStart
                
                results.append(MediaProcessingResult(
                    operation: "media_processing",
                    success: false,
                    processingTime: processingTime,
                    userId: userId,
                    error: error.localizedDescription
                ))
            }
            
            // Wait for next operation
            try await Task.sleep(nanoseconds: UInt64(operationInterval * 1_000_000_000))
        }
        
        return results
    }
    
    // MARK: - Collaboration Load Tests
    
    func testCollaborationLoadLightLoad() async throws {
        try await runCollaborationLoadTest(config: lightLoad)
    }
    
    func testCollaborationLoadMediumLoad() async throws {
        try await runCollaborationLoadTest(config: mediumLoad)
    }
    
    func testCollaborationLoadHeavyLoad() async throws {
        try await runCollaborationLoadTest(config: heavyLoad)
    }
    
    private func runCollaborationLoadTest(config: LoadTestConfig) async throws {
        let expectation = expectation(description: "Collaboration load test - \(config.concurrentUsers) users")
        expectation.expectedFulfillmentCount = config.concurrentUsers
        
        let collaborationManager = CollaborationManager.shared
        let testResults = ThreadSafeArray<CollaborationResult>()
        
        // Create shared session
        let sessionId = UUID()
        let hostUserId = UUID()
        
        let session = try await collaborationManager.createSession(
            projectId: UUID(),
            createdBy: hostUserId,
            isEncrypted: true
        )
        
        // Launch concurrent users
        for userId in 0..<config.concurrentUsers {
            Task {
                do {
                    let userResults = try await performCollaborationOperations(
                        sessionId: session.id,
                        userId: UUID(),
                        duration: config.testDuration,
                        operationsPerSecond: config.operationsPerSecond
                    )
                    testResults.append(contentsOf: userResults)
                    expectation.fulfill()
                } catch {
                    XCTFail("Collaboration load test failed for user \(userId): \(error)")
                    expectation.fulfill()
                }
            }
        }
        
        await fulfillment(of: [expectation], timeout: config.testDuration + 30.0)
        
        // Analyze results
        let results = testResults.elements
        let successfulOperations = results.filter { $0.success }.count
        let failedOperations = results.filter { !$0.success }.count
        let averageLatency = results.map { $0.latency }.reduce(0, +) / Double(results.count)
        
        // Record test results
        let loadTestResult = LoadTestResult(
            testName: "Collaboration Load Test",
            configuration: config,
            totalOperations: results.count,
            successfulOperations: successfulOperations,
            failedOperations: failedOperations,
            averageResponseTime: averageLatency,
            operationsPerSecond: Double(results.count) / config.testDuration,
            memoryUsage: performanceMonitor.getCurrentMemoryUsage(),
            cpuUsage: performanceMonitor.getCurrentCPUUsage()
        )
        
        loadTestResults.append(loadTestResult)
        
        // Assertions for collaboration
        XCTAssertGreaterThan(successfulOperations, results.count * 95 / 100, "Should have >95% success rate for collaboration")
        XCTAssertLessThan(averageLatency, 0.5, "Average latency should be <500ms")
    }
    
    private func performCollaborationOperations(sessionId: UUID, userId: UUID, duration: TimeInterval, operationsPerSecond: Double) async throws -> [CollaborationResult] {
        let collaborationManager = CollaborationManager.shared
        var results: [CollaborationResult] = []
        
        // Join session
        let joinResult = try await collaborationManager.joinSession(sessionId: sessionId, userId: userId)
        XCTAssertTrue(joinResult.success)
        
        let endTime = Date().addingTimeInterval(duration)
        let operationInterval = 1.0 / operationsPerSecond
        
        while Date() < endTime {
            let operationStart = CFAbsoluteTimeGetCurrent()
            
            do {
                // Send collaboration message
                let message = CollaborationMessage(
                    type: .edit,
                    content: "Load test operation \(Date().timeIntervalSince1970)",
                    senderId: userId,
                    timestamp: Date()
                )
                
                let encryptedData = try JSONEncoder().encode(message)
                
                try await collaborationManager.sendMessage(
                    sessionId: sessionId,
                    encryptedData: encryptedData,
                    senderId: userId
                )
                
                let operationEnd = CFAbsoluteTimeGetCurrent()
                let latency = operationEnd - operationStart
                
                results.append(CollaborationResult(
                    operation: "send_message",
                    success: true,
                    latency: latency,
                    userId: userId
                ))
                
            } catch {
                let operationEnd = CFAbsoluteTimeGetCurrent()
                let latency = operationEnd - operationStart
                
                results.append(CollaborationResult(
                    operation: "send_message",
                    success: false,
                    latency: latency,
                    userId: userId,
                    error: error.localizedDescription
                ))
            }
            
            // Wait for next operation
            try await Task.sleep(nanoseconds: UInt64(operationInterval * 1_000_000_000))
        }
        
        return results
    }
    
    // MARK: - Memory Stress Tests
    
    func testMemoryStressTest() async throws {
        let expectation = expectation(description: "Memory stress test")
        
        let initialMemory = performanceMonitor.getCurrentMemoryUsage()
        var allocatedObjects: [Data] = []
        
        do {
            // Allocate memory progressively
            for i in 0..<1000 {
                let chunkSize = 1024 * 1024 // 1MB chunks
                let data = Data(count: chunkSize)
                allocatedObjects.append(data)
                
                let currentMemory = performanceMonitor.getCurrentMemoryUsage()
                let memoryIncrease = currentMemory - initialMemory
                
                // Monitor memory pressure
                if memoryIncrease > 500 * 1024 * 1024 { // 500MB limit
                    break
                }
                
                if i % 100 == 0 {
                    // Test system responsiveness under memory pressure
                    let responseTest = try await performQuickDatabaseOperation()
                    XCTAssertTrue(responseTest, "System should remain responsive under memory pressure")
                }
            }
            
            // Test memory cleanup
            allocatedObjects.removeAll()
            
            // Force garbage collection
            for _ in 0..<5 {
                autoreleasepool {
                    _ = Data(count: 1024)
                }
            }
            
            // Wait for cleanup
            try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
            
            let finalMemory = performanceMonitor.getCurrentMemoryUsage()
            let memoryAfterCleanup = finalMemory - initialMemory
            
            XCTAssertLessThan(memoryAfterCleanup, 100 * 1024 * 1024, "Memory should be cleaned up after object deallocation")
            
            expectation.fulfill()
            
        } catch {
            XCTFail("Memory stress test failed: \(error)")
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: 60.0)
    }
    
    // MARK: - CPU Stress Tests
    
    func testCPUStressTest() async throws {
        let expectation = expectation(description: "CPU stress test")
        expectation.expectedFulfillmentCount = 8 // 8 concurrent tasks
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Launch CPU-intensive tasks
        for i in 0..<8 {
            Task {
                do {
                    try await performCPUIntensiveTask(taskId: i, duration: 30.0)
                    expectation.fulfill()
                } catch {
                    XCTFail("CPU stress test failed for task \(i): \(error)")
                    expectation.fulfill()
                }
            }
        }
        
        await fulfillment(of: [expectation], timeout: 45.0)
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let totalDuration = endTime - startTime
        
        let finalCPUUsage = performanceMonitor.getCurrentCPUUsage()
        
        // Verify system remained stable under CPU load
        XCTAssertLessThan(totalDuration, 35.0, "CPU stress test should complete within reasonable time")
        XCTAssertLessThan(finalCPUUsage, 95.0, "CPU usage should not max out completely")
    }
    
    private func performCPUIntensiveTask(taskId: Int, duration: TimeInterval) async throws {
        let endTime = Date().addingTimeInterval(duration)
        var iterations = 0
        
        while Date() < endTime {
            // CPU-intensive calculation
            var result = 0.0
            for i in 0..<10000 {
                result += sqrt(Double(i)) * sin(Double(i))
            }
            
            iterations += 1
            
            // Yield control periodically
            if iterations % 100 == 0 {
                try await Task.sleep(nanoseconds: 1_000_000) // 1ms
                
                // Test system responsiveness
                let quickTest = try await performQuickDatabaseOperation()
                if !quickTest {
                    throw LoadTestError.systemUnresponsive
                }
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func createTestMediaURL() -> URL {
        let tempDirectory = FileManager.default.temporaryDirectory
        let testFileName = "load_test_\(UUID().uuidString).mp4"
        return tempDirectory.appendingPathComponent(testFileName)
    }
    
    private func performQuickDatabaseOperation() async throws -> Bool {
        let databaseManager = DatabaseManager.shared
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            _ = try await databaseManager.loadAllProjects()
            let endTime = CFAbsoluteTimeGetCurrent()
            let responseTime = endTime - startTime
            
            return responseTime < 2.0 // Should respond within 2 seconds
        } catch {
            return false
        }
    }
    
    private func generateLoadTestReport() {
        guard !loadTestResults.isEmpty else { return }
        
        let reportURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadTestReport_\(Date().timeIntervalSince1970).json")
        
        do {
            let reportData = try JSONEncoder().encode(loadTestResults)
            try reportData.write(to: reportURL)
            print("Load test report generated at: \(reportURL.path)")
        } catch {
            print("Failed to generate load test report: \(error)")
        }
    }
}

// MARK: - Supporting Types

private struct LoadTestResult: Codable {
    let testName: String
    let configuration: LoadTestConfig
    let totalOperations: Int
    let successfulOperations: Int
    let failedOperations: Int
    let averageResponseTime: TimeInterval
    let operationsPerSecond: Double
    let memoryUsage: Double
    let cpuUsage: Double
    let timestamp: Date = Date()
}

private struct DatabaseOperationResult {
    let operation: String
    let success: Bool
    let responseTime: TimeInterval
    let userId: Int
    let error: String?
    
    init(operation: String, success: Bool, responseTime: TimeInterval, userId: Int, error: String? = nil) {
        self.operation = operation
        self.success = success
        self.responseTime = responseTime
        self.userId = userId
        self.error = error
    }
}

private struct MediaProcessingResult {
    let operation: String
    let success: Bool
    let processingTime: TimeInterval
    let userId: Int
    let error: String?
    
    init(operation: String, success: Bool, processingTime: TimeInterval, userId: Int, error: String? = nil) {
        self.operation = operation
        self.success = success
        self.processingTime = processingTime
        self.userId = userId
        self.error = error
    }
}

private struct CollaborationResult {
    let operation: String
    let success: Bool
    let latency: TimeInterval
    let userId: UUID
    let error: String?
    
    init(operation: String, success: Bool, latency: TimeInterval, userId: UUID, error: String? = nil) {
        self.operation = operation
        self.success = success
        self.latency = latency
        self.userId = userId
        self.error = error
    }
}

private enum LoadTestError: Error {
    case systemUnresponsive
    case resourceExhausted
    case operationTimeout
}

private enum MediaProcessingError: Error {
    case loadFailed
    case processingFailed
    case unsupportedFormat
}

extension LoadTestConfig: Codable {}

// MARK: - Thread-Safe Array

private class ThreadSafeArray<Element> {
    private var array: [Element] = []
    private let queue = DispatchQueue(label: "ThreadSafeArray", attributes: .concurrent)
    
    func append(_ element: Element) {
        queue.async(flags: .barrier) {
            self.array.append(element)
        }
    }
    
    func append(contentsOf elements: [Element]) {
        queue.async(flags: .barrier) {
            self.array.append(contentsOf: elements)
        }
    }
    
    var elements: [Element] {
        return queue.sync {
            return Array(array)
        }
    }
}

// MARK: - Performance Monitor

private class PerformanceMonitor {
    private var isMonitoring = false
    private var monitoringQueue = DispatchQueue(label: "PerformanceMonitor", qos: .background)
    
    func startMonitoring() {
        isMonitoring = true
    }
    
    func stopMonitoring() {
        isMonitoring = false
    }
    
    func getCurrentMemoryUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size)
        } else {
            return 0
        }
    }
    
    func getCurrentCPUUsage() -> Double {
        var cpuInfo: processor_info_array_t!
        var numCpuInfo: mach_msg_type_number_t = 0
        var numCpus: natural_t = 0
        
        let result = host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &numCpus, &cpuInfo, &numCpuInfo)
        
        if result == KERN_SUCCESS {
            let cpuLoad = cpuInfo.withMemoryRebound(to: processor_cpu_load_info_t.self, capacity: Int(numCpus)) {
                return $0[0]
            }
            
            let user = Double(cpuLoad.cpu_ticks.0)
            let system = Double(cpuLoad.cpu_ticks.1)
            let idle = Double(cpuLoad.cpu_ticks.2)
            let nice = Double(cpuLoad.cpu_ticks.3)
            
            let total = user + system + idle + nice
            let usage = total > 0 ? ((user + system) / total) * 100.0 : 0.0
            
            vm_deallocate(mach_task_self_, vm_address_t(bitPattern: cpuInfo), vm_size_t(numCpuInfo))
            
            return usage
        }
        
        return 0.0
    }
}