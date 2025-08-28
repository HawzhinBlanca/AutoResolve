// AUTORESOLVE V3.0 - FRAME MEMORY POOL
import Combine
// High-performance memory pool for video frame buffers

import Foundation
import SwiftUI
import CoreVideo
import AVFoundation
import os.log

// MARK: - Frame Memory Pool
final class FrameMemoryPool {
    static let shared = FrameMemoryPool()
    
    // Pool configuration
    private let maxBuffersPerResolution: [PoolResolution: Int] = [
        .hd720: 20,
        .hd1080: 10,
        .uhd4K: 5,
        .uhd8K: 2
    ]
    
    // Buffer pools by resolution
    private var bufferPools: [PoolResolution: BufferPool] = [:]
    
    // Statistics
    @Published var totalMemoryUsage: Int64 = 0
    @Published var activeBuffers: Int = 0
    @Published var poolHitRate: Double = 0
    @Published var allocations: Int = 0
    @Published var deallocations: Int = 0
    
    private var hits: Int = 0
    private var misses: Int = 0
    
    // Memory pressure handling
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    private let logger = Logger.shared
    
    // MARK: - Resolution Definition
    enum PoolResolution: CaseIterable, Hashable {
        case hd720
        case hd1080
        case uhd4K
        case uhd8K
        case custom(width: Int, height: Int)
        
        var dimensions: (width: Int, height: Int) {
            switch self {
            case .hd720: return (1280, 720)
            case .hd1080: return (1920, 1080)
            case .uhd4K: return (3840, 2160)
            case .uhd8K: return (7680, 4320)
            case .custom(let w, let h): return (w, h)
            }
        }
        
        var bufferSize: Int {
            let (w, h) = dimensions
            return w * h * 4 // RGBA 8-bit per channel
        }
        
        static var allCases: [PoolResolution] {
            [.hd720, .hd1080, .uhd4K, .uhd8K]
        }
    }
    
    // MARK: - Initialization
    private init() {
        setupBufferPools()
        setupMemoryPressureHandling()
        startPeriodicCleanup()
    }
    
    private func setupBufferPools() {
        for resolution in PoolResolution.allCases {
            let maxBuffers = maxBuffersPerResolution[resolution] ?? 5
            bufferPools[resolution] = BufferPool(
                resolution: resolution,
                maxBuffers: maxBuffers
            )
        }
        
        logger.info("Frame memory pool initialized with \(self.bufferPools.count) resolution pools")
    }
    
    // MARK: - Buffer Acquisition
    func acquireBuffer(for resolution: PoolResolution) -> FrameBuffer? {
        // Try to get from pool
        if let pool = bufferPools[resolution],
           let buffer = pool.acquire() {
            hits += 1
            updateStatistics()
            logger.debug("Buffer acquired from pool for \(String(describing: resolution))")
            return buffer
        }
        
        // Handle custom resolution
        if case .custom = resolution {
            return createCustomBuffer(resolution: resolution)
        }
        
        // Pool miss - create new buffer if under limit
        misses += 1
        if let pool = bufferPools[resolution],
           pool.canAllocateMore() {
            let buffer = FrameBuffer(resolution: resolution)
            pool.track(buffer)
            allocations += 1
            updateStatistics()
            logger.debug("New buffer allocated for \(String(describing: resolution))")
            return buffer
        }
        
        // Over limit - try to reclaim
        reclaimUnusedBuffers()
        
        // Try again after reclaim
        if let pool = bufferPools[resolution],
           pool.canAllocateMore() {
            let buffer = FrameBuffer(resolution: resolution)
            pool.track(buffer)
            allocations += 1
            updateStatistics()
            return buffer
        }
        
        logger.warning("Failed to acquire buffer for \(String(describing: resolution)) - pool exhausted")
        return nil
    }
    
    func releaseBuffer(_ buffer: FrameBuffer) {
        guard let pool = bufferPools[buffer.resolution] else {
            // Custom resolution buffer - just deallocate
            buffer.deallocate()
            deallocations += 1
            updateStatistics()
            return
        }
        
        pool.release(buffer)
        updateStatistics()
        logger.debug("Buffer released to pool for \(String(describing: buffer.resolution))")
    }
    
    // MARK: - Custom Resolution Handling
    private func createCustomBuffer(resolution: PoolResolution) -> FrameBuffer? {
        guard case .custom = resolution else { return nil }
        
        // Check if custom size is reasonable
        if resolution.bufferSize > 100_000_000 { // 100MB limit for custom
            logger.error("Custom resolution too large: \(resolution.bufferSize) bytes")
            return nil
        }
        
        let buffer = FrameBuffer(resolution: resolution)
        allocations += 1
        updateStatistics()
        return buffer
    }
    
    // MARK: - Memory Management
    private func setupMemoryPressureHandling() {
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])
        
        memoryPressureSource?.setEventHandler { [weak self] in
            self?.handleMemoryPressure()
        }
        
        memoryPressureSource?.resume()
    }
    
    private func handleMemoryPressure() {
        Task { @MainActor in
            logger.warning("Memory pressure detected - clearing frame pools")
            
            // Clear pools based on pressure level
            let freedMemory = clearPools(keepActive: true)
            
            logger.info("Freed \(freedMemory / 1_048_576)MB from frame pools")
            updateStatistics()
        }
    }
    
    private func clearPools(keepActive: Bool) -> Int64 {
        var freedMemory: Int64 = 0
        
        for (_, pool) in bufferPools {
            freedMemory += pool.clear(keepActive: keepActive)
        }
        
        return freedMemory
    }
    
    private func reclaimUnusedBuffers() {
        for (_, pool) in bufferPools {
            pool.reclaimUnused()
        }
        updateStatistics()
    }
    
    // MARK: - Periodic Cleanup
    private func startPeriodicCleanup() {
        Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { _ in
            Task { @MainActor in
                self.performCleanup()
            }
        }
    }
    
    private func performCleanup() {
        logger.debug("Performing periodic cleanup")
        
        // Remove buffers that haven't been used recently
        for (_, pool) in bufferPools {
            pool.removeStale(olderThan: 60) // 60 seconds
        }
        
        updateStatistics()
    }
    
    // MARK: - Statistics
    private func updateStatistics() {
        var totalMemory: Int64 = 0
        var totalActive = 0
        
        for (_, pool) in bufferPools {
            totalMemory += pool.memoryUsage
            totalActive += pool.activeCount
        }
        
        totalMemoryUsage = totalMemory
        activeBuffers = totalActive
        
        let totalRequests = hits + misses
        poolHitRate = totalRequests > 0 ? Double(hits) / Double(totalRequests) : 0
    }
    
    func resetStatistics() {
        hits = 0
        misses = 0
        allocations = 0
        deallocations = 0
        poolHitRate = 0
    }
    
    // MARK: - Debug Info
    func debugInfo() -> String {
        var info = "Frame Memory Pool Status:\n"
        info += "Total Memory: \(totalMemoryUsage / 1_048_576)MB\n"
        info += "Active Buffers: \(activeBuffers)\n"
        info += "Hit Rate: \(String(format: "%.1f%%", poolHitRate * 100))\n"
        info += "Allocations: \(allocations)\n"
        info += "Deallocations: \(deallocations)\n\n"
        
        for (resolution, pool) in bufferPools {
            info += "\(resolution): \(pool.debugInfo())\n"
        }
        
        return info
    }
}

// MARK: - Buffer Pool
private class BufferPool {
    let resolution: FrameMemoryPool.PoolResolution
    let maxBuffers: Int
    
    private var availableBuffers: [FrameBuffer] = []
    private var activeBuffers: Set<ObjectIdentifier> = []
    private var allBuffers: [ObjectIdentifier: FrameBuffer] = [:]
    private let lock = NSLock()
    
    var memoryUsage: Int64 {
        Int64(allBuffers.count * resolution.bufferSize)
    }
    
    var activeCount: Int {
        activeBuffers.count
    }
    
    init(resolution: FrameMemoryPool.PoolResolution, maxBuffers: Int) {
        self.resolution = resolution
        self.maxBuffers = maxBuffers
        
        // Pre-allocate some buffers
        let preAllocateCount = min(2, maxBuffers)
        for _ in 0..<preAllocateCount {
            let buffer = FrameBuffer(resolution: resolution)
            availableBuffers.append(buffer)
            allBuffers[ObjectIdentifier(buffer)] = buffer
        }
    }
    
    func acquire() -> FrameBuffer? {
        lock.lock()
        defer { lock.unlock() }
        
        if let buffer = availableBuffers.popLast() {
            buffer.lastUsed = Date()
            activeBuffers.insert(ObjectIdentifier(buffer))
            return buffer
        }
        
        return nil
    }
    
    func release(_ buffer: FrameBuffer) {
        lock.lock()
        defer { lock.unlock() }
        
        let id = ObjectIdentifier(buffer)
        activeBuffers.remove(id)
        
        // Reset buffer before returning to pool
        buffer.reset()
        
        // Only return to pool if we're tracking it
        if allBuffers[id] != nil {
            availableBuffers.append(buffer)
        }
    }
    
    func track(_ buffer: FrameBuffer) {
        lock.lock()
        defer { lock.unlock() }
        
        let id = ObjectIdentifier(buffer)
        allBuffers[id] = buffer
        activeBuffers.insert(id)
    }
    
    func canAllocateMore() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        
        return allBuffers.count < maxBuffers
    }
    
    func clear(keepActive: Bool) -> Int64 {
        lock.lock()
        defer { lock.unlock() }
        
        var freedMemory: Int64 = 0
        
        if keepActive {
            // Only clear available buffers
            for buffer in availableBuffers {
                buffer.deallocate()
                allBuffers.removeValue(forKey: ObjectIdentifier(buffer))
                freedMemory += Int64(resolution.bufferSize)
            }
            availableBuffers.removeAll()
        } else {
            // Clear all buffers
            for (id, buffer) in allBuffers {
                if !activeBuffers.contains(id) {
                    buffer.deallocate()
                    freedMemory += Int64(resolution.bufferSize)
                }
            }
            
            // Remove non-active buffers
            allBuffers = allBuffers.filter { activeBuffers.contains($0.key) }
            availableBuffers.removeAll()
        }
        
        return freedMemory
    }
    
    func reclaimUnused() {
        lock.lock()
        defer { lock.unlock() }
        
        // Find inactive buffers that can be reclaimed
        let inactiveIDs = Set(allBuffers.keys).subtracting(activeBuffers)
        
        for id in inactiveIDs {
            if let buffer = allBuffers[id],
               !availableBuffers.contains(where: { ObjectIdentifier($0) == id }) {
                availableBuffers.append(buffer)
            }
        }
    }
    
    func removeStale(olderThan seconds: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        
        let cutoff = Date().addingTimeInterval(-seconds)
        
        availableBuffers.removeAll { buffer in
            if buffer.lastUsed < cutoff {
                buffer.deallocate()
                allBuffers.removeValue(forKey: ObjectIdentifier(buffer))
                return true
            }
            return false
        }
    }
    
    func debugInfo() -> String {
        lock.lock()
        defer { lock.unlock() }
        
        return "Total: \(allBuffers.count)/\(maxBuffers), Active: \(activeBuffers.count), Available: \(availableBuffers.count)"
    }
}

// MARK: - Frame Buffer
final class FrameBuffer {
    let resolution: FrameMemoryPool.PoolResolution
    let pixelBuffer: CVPixelBuffer?
    var lastUsed: Date
    private var isAllocated: Bool = true
    
    init(resolution: FrameMemoryPool.PoolResolution) {
        self.resolution = resolution
        self.lastUsed = Date()
        
        let (width, height) = resolution.dimensions
        
        let attributes: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        if status == kCVReturnSuccess {
            self.pixelBuffer = pixelBuffer
        } else {
            self.pixelBuffer = nil
        }
    }
    
    func reset() {
        // Clear buffer contents if needed
        if let pixelBuffer = pixelBuffer {
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            if let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) {
                let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
                let height = CVPixelBufferGetHeight(pixelBuffer)
                memset(baseAddress, 0, bytesPerRow * height)
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        }
        
        lastUsed = Date()
    }
    
    func deallocate() {
        isAllocated = false
        // CVPixelBuffer is automatically released when reference count drops
    }
    
    deinit {
        if isAllocated {
            deallocate()
        }
    }
}

// MARK: - Frame Buffer Context
struct FrameBufferContext {
    let buffer: FrameBuffer
    private let pool: FrameMemoryPool
    
    init?(resolution: FrameMemoryPool.PoolResolution) {
        guard let buffer = FrameMemoryPool.shared.acquireBuffer(for: resolution) else {
            return nil
        }
        self.buffer = buffer
        self.pool = FrameMemoryPool.shared
    }
    
    func release() {
        pool.releaseBuffer(buffer)
    }
}

// MARK: - Performance Extensions
extension FrameMemoryPool {
    var formattedMemoryUsage: String {
        let mb = Double(totalMemoryUsage) / 1_048_576
        return String(format: "%.1f MB", mb)
    }
    
    var formattedHitRate: String {
        String(format: "%.1f%%", poolHitRate * 100)
    }
    
    var efficiency: Double {
        guard allocations > 0 else { return 0 }
        return Double(hits) / Double(allocations)
    }
}
