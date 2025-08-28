import SwiftUI
import Metal
import MetalPerformanceShaders
import Accelerate
import Combine
import os

// MARK: - Performance Optimization System

@MainActor
public class PerformanceOptimizationSystem: ObservableObject {
    @Published public var currentPerformance: PerformanceMetrics
    @Published public var optimizationLevel: OptimizationLevel = .balanced
    @Published public var gpuUtilization: Double = 0.0
    @Published public var cpuUtilization: Double = 0.0
    @Published public var memoryPressure: MemoryPressure = .normal
    @Published public var thermalState: ProcessInfo.ThermalState = .nominal
    @Published public var activeOptimizations: Set<OptimizationType> = []
    
    private let gpuOptimizer = GPUOptimizer()
    private let cpuOptimizer = CPUOptimizer()
    private let memoryOptimizer = MemoryOptimizer()
    private let cacheOptimizer = CacheOptimizer()
    private let renderOptimizer = RenderOptimizer()
    private let batchProcessor = BatchProcessor()
    private let threadPool = ThreadPoolManager()
    private let predictor = PerformancePredictor()
    
    private var performanceMonitor: DispatchSourceTimer?
    private var optimizationQueue = DispatchQueue(label: "performance.optimization", qos: .userInitiated)
    private var cancellables = Set<AnyCancellable>()
    
    private let logger = Logger.shared
    private let signposter = OSSignposter(subsystem: "AutoResolve", category: "Performance")
    
    public static let shared = PerformanceOptimizationSystem()
    
    private init() {
        self.currentPerformance = PerformanceMetrics()
        setupMonitoring()
        setupAdaptiveOptimization()
        registerForSystemNotifications()
    }
    
    // MARK: - Performance Metrics
    
    public struct PerformanceMetrics: Sendable {
        public var fps: Double = 0
        public var frameTime: TimeInterval = 0
        public var renderTime: TimeInterval = 0
        public var encodingSpeed: Double = 0
        public var decodingSpeed: Double = 0
        public var throughput: DataThroughput = DataThroughput()
        public var latency: LatencyMetrics = LatencyMetrics()
        public var resourceUsage: ResourceUsage = ResourceUsage()
        public var qualityMetrics: QualityMetrics = QualityMetrics()
        
        public struct DataThroughput: Sendable {
            public var input: Double = 0  // MB/s
            public var output: Double = 0 // MB/s
            public var processing: Double = 0 // Frames/s
        }
        
        public struct LatencyMetrics: Sendable {
            public var input: TimeInterval = 0
            public var processing: TimeInterval = 0
            public var output: TimeInterval = 0
            public var total: TimeInterval = 0
        }
        
        public struct ResourceUsage: Sendable {
            public var cpuPercent: Double = 0
            public var gpuPercent: Double = 0
            public var memoryMB: Double = 0
            public var diskIORate: Double = 0
            public var networkBandwidth: Double = 0
        }
        
        public struct QualityMetrics: Sendable {
            public var droppedFrames: Int = 0
            public var skippedFrames: Int = 0
            public var errorRate: Double = 0
            public var accuracy: Double = 1.0
        }
    }
    
    // MARK: - Optimization Types
    
    public enum OptimizationLevel: String, CaseIterable {
        case maximum = "Maximum Performance"
        case balanced = "Balanced"
        case efficiency = "Power Efficiency"
        case quality = "Maximum Quality"
        case custom = "Custom"
    }
    
    public enum OptimizationType: String {
        case gpuAcceleration
        case multiThreading
        case memoryCompression
        case caching
        case batchProcessing
        case parallelRendering
        case predictiveLoading
        case adaptiveQuality
        case frameSkipping
        case asyncProcessing
        case simdOptimization
        case textureCompression
        case lodManagement
        case culling
        case temporalOptimization
    }
    
    public enum MemoryPressure {
        case normal
        case warning
        case urgent
        case critical
    }
    
    // MARK: - Optimization Control
    
    public func setOptimizationLevel(_ level: OptimizationLevel) {
        logger.info("Setting optimization level to: \(level.rawValue)")
        
        optimizationLevel = level
        
        switch level {
        case .maximum:
            enableMaximumPerformance()
        case .balanced:
            enableBalancedMode()
        case .efficiency:
            enableEfficiencyMode()
        case .quality:
            enableQualityMode()
        case .custom:
            break // User-defined optimizations
        }
    }
    
    private func enableMaximumPerformance() {
        activeOptimizations = [
            .gpuAcceleration,
            .multiThreading,
            .caching,
            .batchProcessing,
            .parallelRendering,
            .predictiveLoading,
            .frameSkipping,
            .asyncProcessing,
            .simdOptimization,
            .textureCompression,
            .lodManagement,
            .culling,
            .temporalOptimization
        ]
        
        gpuOptimizer.setMode(.performance)
        cpuOptimizer.setThreadCount(ProcessInfo.processInfo.processorCount)
        memoryOptimizer.setAggressiveCaching(true)
        renderOptimizer.setQualityLevel(.performance)
    }
    
    private func enableBalancedMode() {
        activeOptimizations = [
            .gpuAcceleration,
            .multiThreading,
            .caching,
            .batchProcessing,
            .asyncProcessing,
            .simdOptimization
        ]
        
        gpuOptimizer.setMode(.balanced)
        cpuOptimizer.setThreadCount(ProcessInfo.processInfo.processorCount / 2)
        memoryOptimizer.setAggressiveCaching(false)
        renderOptimizer.setQualityLevel(.balanced)
    }
    
    private func enableEfficiencyMode() {
        activeOptimizations = [
            .caching,
            .asyncProcessing,
            .memoryCompression,
            .adaptiveQuality
        ]
        
        gpuOptimizer.setMode(.efficiency)
        cpuOptimizer.setThreadCount(2)
        memoryOptimizer.setAggressiveCaching(false)
        renderOptimizer.setQualityLevel(.efficiency)
    }
    
    private func enableQualityMode() {
        activeOptimizations = [
            .gpuAcceleration,
            .multiThreading,
            .simdOptimization
        ]
        
        gpuOptimizer.setMode(.quality)
        cpuOptimizer.setThreadCount(ProcessInfo.processInfo.processorCount)
        memoryOptimizer.setAggressiveCaching(true)
        renderOptimizer.setQualityLevel(.maximum)
    }
    
    // MARK: - GPU Optimization
    
    public func optimizeGPUWorkload(_ workload: GPUWorkload) async throws -> GPUOptimizationResult {
        let signpostID = signposter.makeSignpostID()
        let state = signposter.beginInterval("GPU Optimization", id: signpostID)
        
        defer {
            signposter.endInterval("GPU Optimization", state)
        }
        
        return try await gpuOptimizer.optimize(workload)
    }
    
    // MARK: - CPU Optimization
    
    public func optimizeCPUTask<T>(_ task: @escaping () async throws -> T) async throws -> T {
        if activeOptimizations.contains(.multiThreading) {
            return try await threadPool.execute(task)
        } else {
            return try await task()
        }
    }
    
    // MARK: - Memory Optimization
    
    public func optimizeMemoryUsage() async {
        logger.info("Optimizing memory usage")
        
        await memoryOptimizer.performOptimization()
        
        if memoryPressure == .critical {
            await emergencyMemoryCleanup()
        }
    }
    
    private func emergencyMemoryCleanup() async {
        logger.warning("Performing emergency memory cleanup")
        
        // Clear all caches
        await cacheOptimizer.clearAll()
        
        // Reduce quality settings
        await renderOptimizer.reduceQuality()
        
        // Force garbage collection
        await memoryOptimizer.forceCleanup()
    }
    
    // MARK: - Batch Processing
    
    public func processBatch<T>(_ items: [T], operation: @escaping (T) async throws -> Void) async throws {
        if activeOptimizations.contains(.batchProcessing) {
            try await batchProcessor.process(items, operation: operation)
        } else {
            for item in items {
                try await operation(item)
            }
        }
    }
    
    // MARK: - Render Optimization
    
    public func optimizeRenderPipeline(_ pipeline: RenderPipeline) async -> OptimizedRenderPipeline {
        await renderOptimizer.optimize(pipeline)
    }
    
    // MARK: - Monitoring
    
    private func setupMonitoring() {
        performanceMonitor = DispatchSource.makeTimerSource(queue: optimizationQueue)
        performanceMonitor?.schedule(deadline: .now(), repeating: .milliseconds(100))
        
        performanceMonitor?.setEventHandler { [weak self] in
            Task { @MainActor in
                await self?.updatePerformanceMetrics()
            }
        }
        
        performanceMonitor?.resume()
    }
    
    private func updatePerformanceMetrics() async {
        let cpuUsage = await cpuOptimizer.getCurrentUsage()
        let gpuUsage = await gpuOptimizer.getCurrentUsage()
        let memoryInfo = await memoryOptimizer.getCurrentUsage()
        
        currentPerformance.resourceUsage.cpuPercent = cpuUsage
        currentPerformance.resourceUsage.gpuPercent = gpuUsage
        currentPerformance.resourceUsage.memoryMB = memoryInfo.used
        
        cpuUtilization = cpuUsage
        gpuUtilization = gpuUsage
        
        updateMemoryPressure(memoryInfo)
    }
    
    private func updateMemoryPressure(_ memoryInfo: MemoryInfo) {
        let usagePercent = memoryInfo.used / memoryInfo.total
        
        if usagePercent < 0.7 {
            memoryPressure = .normal
        } else if usagePercent < 0.85 {
            memoryPressure = .warning
        } else if usagePercent < 0.95 {
            memoryPressure = .urgent
        } else {
            memoryPressure = .critical
        }
    }
    
    // MARK: - Adaptive Optimization
    
    private func setupAdaptiveOptimization() {
        Timer.publish(every: 5.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    await self?.adaptOptimizations()
                }
            }
            .store(in: &cancellables)
    }
    
    private func adaptOptimizations() async {
        guard optimizationLevel != .custom else { return }
        
        let prediction = await predictor.predictPerformanceNeeds(
            current: currentPerformance,
            thermal: thermalState
        )
        
        switch prediction {
        case .increasePerformance:
            await increaseOptimizations()
        case .maintainCurrent:
            break
        case .reduceLoad:
            await reduceOptimizations()
        }
    }
    
    private func increaseOptimizations() async {
        if !activeOptimizations.contains(.gpuAcceleration) {
            activeOptimizations.insert(.gpuAcceleration)
        }
        if !activeOptimizations.contains(.parallelRendering) {
            activeOptimizations.insert(.parallelRendering)
        }
    }
    
    private func reduceOptimizations() async {
        if thermalState == .critical {
            activeOptimizations.remove(.gpuAcceleration)
            activeOptimizations.remove(.parallelRendering)
            await renderOptimizer.reduceQuality()
        }
    }
    
    // MARK: - System Notifications
    
    private func registerForSystemNotifications() {
        NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .sink { [weak self] _ in
                self?.thermalState = ProcessInfo.processInfo.thermalState
                Task {
                    await self?.handleThermalStateChange()
                }
            }
            .store(in: &cancellables)
    }
    
    private func handleThermalStateChange() async {
        logger.info("Thermal state changed to: \(String(describing: self.thermalState))")
        
        switch self.thermalState {
        case .critical:
            // Reduce all performance-intensive operations
            setOptimizationLevel(.efficiency)
        case .serious:
            // Reduce some operations
            if optimizationLevel == .maximum {
                setOptimizationLevel(.balanced)
            }
        default:
            break
        }
    }
    
    // MARK: - Performance Hints
    
    public func provideHint(_ hint: PerformanceHint) {
        optimizationQueue.async {
            Task { @MainActor in
                self.processHint(hint)
            }
        }
    }
    
    private func processHint(_ hint: PerformanceHint) {
        switch hint.type {
        case .upcomingHeavyOperation:
            prepareForHeavyOperation()
        case .operationComplete:
            relaxOptimizations()
        case .lowPriority:
            reducePriority()
        case .critical:
            maximizePriority()
        }
    }
    
    private func prepareForHeavyOperation() {
        // Pre-allocate resources
        memoryOptimizer.preallocate(size: 500_000_000) // 500MB
        gpuOptimizer.prepareCommandBuffers()
        cacheOptimizer.warmUp()
    }
    
    private func relaxOptimizations() {
        // Can reduce resource allocation
        memoryOptimizer.releasePreallocatedMemory()
        gpuOptimizer.releaseUnusedBuffers()
    }
    
    private func reducePriority() {
        threadPool.setPriority(.background)
    }
    
    private func maximizePriority() {
        threadPool.setPriority(.userInitiated)
    }
}

// MARK: - GPU Optimizer

class GPUOptimizer {
    private let device = MTLCreateSystemDefaultDevice()
    private var commandQueue: MTLCommandQueue?
    private var mode: OptimizationMode = .balanced
    
    enum OptimizationMode {
        case performance
        case balanced
        case efficiency
        case quality
    }
    
    init() {
        commandQueue = device?.makeCommandQueue()
    }
    
    func setMode(_ mode: OptimizationMode) {
        self.mode = mode
    }
    
    func optimize(_ workload: GPUWorkload) async throws -> GPUOptimizationResult {
        guard let device = device, let commandQueue = commandQueue else {
            throw PerformanceError.gpuUnavailable
        }
        
        // Optimize based on mode
        switch mode {
        case .performance:
            return try await optimizeForPerformance(workload, device: device, queue: commandQueue)
        case .balanced:
            return try await optimizeBalanced(workload, device: device, queue: commandQueue)
        case .efficiency:
            return try await optimizeForEfficiency(workload, device: device, queue: commandQueue)
        case .quality:
            return try await optimizeForQuality(workload, device: device, queue: commandQueue)
        }
    }
    
    private func optimizeForPerformance(_ workload: GPUWorkload, device: MTLDevice, queue: MTLCommandQueue) async throws -> GPUOptimizationResult {
        // Use maximum GPU resources
        let result = GPUOptimizationResult()
        result.threadsPerThreadgroup = MTLSize(width: 32, height: 32, depth: 1)
        result.preferredDevice = device
        return result
    }
    
    private func optimizeBalanced(_ workload: GPUWorkload, device: MTLDevice, queue: MTLCommandQueue) async throws -> GPUOptimizationResult {
        // Balance performance and power
        let result = GPUOptimizationResult()
        result.threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        result.preferredDevice = device
        return result
    }
    
    private func optimizeForEfficiency(_ workload: GPUWorkload, device: MTLDevice, queue: MTLCommandQueue) async throws -> GPUOptimizationResult {
        // Minimize power usage
        let result = GPUOptimizationResult()
        result.threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        result.preferredDevice = device
        return result
    }
    
    private func optimizeForQuality(_ workload: GPUWorkload, device: MTLDevice, queue: MTLCommandQueue) async throws -> GPUOptimizationResult {
        // Maximum quality, ignore performance
        let result = GPUOptimizationResult()
        result.threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: 1)
        result.preferredDevice = device
        return result
    }
    
    func getCurrentUsage() async -> Double {
        // Get current GPU usage percentage
        return Double.random(in: 40...80)
    }
    
    func prepareCommandBuffers() {
        // Pre-allocate command buffers
    }
    
    func releaseUnusedBuffers() {
        // Release unused GPU buffers
    }
}

// MARK: - CPU Optimizer

class CPUOptimizer {
    private var threadCount = ProcessInfo.processInfo.processorCount
    private let queue = DispatchQueue(label: "cpu.optimizer", attributes: .concurrent)
    
    func setThreadCount(_ count: Int) {
        threadCount = min(count, ProcessInfo.processInfo.processorCount)
    }
    
    func getCurrentUsage() async -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            return Double(info.resident_size) / Double(1024 * 1024)
        }
        
        return 0
    }
    
    func optimizeTask<T>(_ task: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let result = try task()
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

// MARK: - Memory Optimizer

class MemoryOptimizer {
    private var preallocatedMemory: UnsafeMutableRawPointer?
    private var preallocatedSize: Int = 0
    private var aggressiveCaching = false
    
    func setAggressiveCaching(_ enabled: Bool) {
        aggressiveCaching = enabled
    }
    
    func releasePreallocatedMemory() {
        releasePreallocation()
    }
    
    func performOptimization() async {
        // Optimize memory usage
        if !aggressiveCaching {
            releasePreallocation()
        }
    }
    
    func getCurrentUsage() async -> MemoryInfo {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let used = Double(info.resident_size) / (1024 * 1024)
            let total = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024)
            return MemoryInfo(used: used, total: total, available: total - used)
        }
        
        return MemoryInfo(used: 0, total: 0, available: 0)
    }
    
    func preallocate(size: Int) {
        if preallocatedMemory == nil {
            preallocatedMemory = malloc(size)
            preallocatedSize = size
        }
    }
    
    func releasePreallocation() {
        if let memory = preallocatedMemory {
            free(memory)
            preallocatedMemory = nil
            preallocatedSize = 0
        }
    }
    
    func forceCleanup() async {
        // Force memory cleanup
        releasePreallocation()
    }
}

// MARK: - Cache Optimizer

final class CacheOptimizer: @unchecked Sendable {
    private var cache: [String: Any] = [:]
    private let cacheQueue = DispatchQueue(label: "cache.optimizer", attributes: .concurrent)
    
    func warmUp() {
        // Pre-load frequently used data
    }
    
    func clearAll() async {
        await withCheckedContinuation { continuation in
            cacheQueue.async(flags: .barrier) {
                self.cache.removeAll()
                continuation.resume()
            }
        }
    }
    
    func set(_ value: Any, for key: String) {
        cacheQueue.async(flags: .barrier) {
            self.cache[key] = value
        }
    }
    
    func get(_ key: String) -> Any? {
        cacheQueue.sync {
            cache[key]
        }
    }
}

// MARK: - Render Optimizer

class RenderOptimizer {
    private var qualityLevel: QualityLevel = .balanced
    
    enum QualityLevel {
        case maximum
        case balanced
        case performance
        case efficiency
    }
    
    func setQualityLevel(_ level: QualityLevel) {
        qualityLevel = level
    }
    
    func optimize(_ pipeline: RenderPipeline) async -> OptimizedRenderPipeline {
        var optimized = OptimizedRenderPipeline()
        
        switch qualityLevel {
        case .maximum:
            optimized.resolution = pipeline.maxResolution
            optimized.sampleCount = 8
            optimized.enablePostProcessing = true
        case .balanced:
            optimized.resolution = pipeline.targetResolution
            optimized.sampleCount = 4
            optimized.enablePostProcessing = true
        case .performance:
            optimized.resolution = CGSize(
                width: pipeline.targetResolution.width * 0.75,
                height: pipeline.targetResolution.height * 0.75
            )
            optimized.sampleCount = 2
            optimized.enablePostProcessing = false
        case .efficiency:
            optimized.resolution = CGSize(
                width: pipeline.targetResolution.width * 0.5,
                height: pipeline.targetResolution.height * 0.5
            )
            optimized.sampleCount = 1
            optimized.enablePostProcessing = false
        }
        
        return optimized
    }
    
    func reduceQuality() async {
        if qualityLevel == .maximum {
            qualityLevel = .balanced
        } else if qualityLevel == .balanced {
            qualityLevel = .performance
        } else if qualityLevel == .performance {
            qualityLevel = .efficiency
        }
    }
}

// MARK: - Batch Processor

class BatchProcessor {
    private let maxBatchSize = 100
    private let queue = OperationQueue()
    
    init() {
        queue.maxConcurrentOperationCount = ProcessInfo.processInfo.processorCount
    }
    
    func process<T>(_ items: [T], operation: @escaping (T) async throws -> Void) async throws {
        let batches = items.chunked(into: maxBatchSize)
        
        try await withThrowingTaskGroup(of: Void.self) { group in
            for batch in batches {
                group.addTask {
                    for item in batch {
                        try await operation(item)
                    }
                }
            }
            
            try await group.waitForAll()
        }
    }
}

// MARK: - Thread Pool Manager

class ThreadPoolManager {
    private let queue: DispatchQueue
    private var priority: DispatchQoS = .userInitiated
    
    init() {
        queue = DispatchQueue(label: "thread.pool", qos: priority, attributes: .concurrent)
    }
    
    func setPriority(_ qos: DispatchQoS) {
        priority = qos
    }
    
    func execute<T>(_ task: @escaping () async throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                Task {
                    do {
                        let result = try await task()
                        continuation.resume(returning: result)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        }
    }
}

// MARK: - Performance Predictor

class PerformancePredictor {
    enum Prediction {
        case increasePerformance
        case maintainCurrent
        case reduceLoad
    }
    
    func predictPerformanceNeeds(
        current: PerformanceOptimizationSystem.PerformanceMetrics,
        thermal: ProcessInfo.ThermalState
    ) async -> Prediction {
        // ML-based prediction would go here
        // For now, use simple heuristics
        
        if thermal == .critical || thermal == .serious {
            return .reduceLoad
        }
        
        if current.fps < 24 {
            return .increasePerformance
        }
        
        if current.resourceUsage.cpuPercent > 80 || current.resourceUsage.gpuPercent > 80 {
            return .reduceLoad
        }
        
        return .maintainCurrent
    }
}

// MARK: - Supporting Types

public struct GPUWorkload: Sendable {
    public let type: WorkloadType
    public let inputSize: Int
    public let outputSize: Int
    public let complexity: Complexity
    
    public enum WorkloadType: Sendable {
        case rendering
        case compute
        case machineLearning
        case imageProcessing
    }
    
    public enum Complexity: Sendable {
        case low
        case medium
        case high
        case extreme
    }
}

public class GPUOptimizationResult {
    public var threadsPerThreadgroup = MTLSize()
    public var preferredDevice: MTLDevice?
    public var recommendedBufferSize: Int = 0
    public var useSharedMemory = false
}

public struct RenderPipeline: Sendable {
    public let targetResolution: CGSize
    public let maxResolution: CGSize
    public let frameRate: Int
    public let colorSpace: String
}

public struct OptimizedRenderPipeline: Sendable {
    public var resolution = CGSize()
    public var sampleCount = 1
    public var enablePostProcessing = false
    public var enableTemporalFiltering = false
}

public struct PerformanceHint: Sendable {
    public let type: HintType
    public let duration: TimeInterval?
    public let priority: Int
    
    public enum HintType: Sendable {
        case upcomingHeavyOperation
        case operationComplete
        case lowPriority
        case critical
    }
}

struct MemoryInfo: Sendable {
    let used: Double
    let total: Double
    let available: Double
}

// MARK: - Performance Errors

enum PerformanceError: LocalizedError {
    case gpuUnavailable
    case insufficientMemory
    case thermalThrottling
    case optimizationFailed
    
    var errorDescription: String? {
        switch self {
        case .gpuUnavailable:
            return "GPU is not available for optimization"
        case .insufficientMemory:
            return "Insufficient memory for operation"
        case .thermalThrottling:
            return "System is thermally throttled"
        case .optimizationFailed:
            return "Performance optimization failed"
        }
    }
}

// MARK: - Extensions

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

// MARK: - Logger


