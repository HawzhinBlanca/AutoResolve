// AUTORESOLVE V3.0 - RESOURCE MANAGER
// Centralized resource management with automatic cleanup

import Foundation
import SwiftUI
import Combine
import os.log

// MARK: - Resource Manager
@MainActor
final class ResourceManager: ObservableObject {
    static let shared = ResourceManager()
    
    // Resource tracking
    private var managedResources: [ResourceIdentifier: ManagedResource] = [:]
    private var resourceGroups: [String: Set<ResourceIdentifier>] = [:]
    private var cleanupTimers: [ResourceIdentifier: Timer] = [:]
    
    // Statistics
    @Published var totalResources = 0
    @Published var activeResources = 0
    @Published var memoryFootprint: Int64 = 0
    @Published var lastCleanup = Date()
    @Published var leaksDetected = 0
    
    // Configuration
    private let cleanupInterval: TimeInterval = 60 // 1 minute
    private let resourceTimeout: TimeInterval = 300 // 5 minutes
    private let memoryLimit: Int64 = 500_000_000 // 500MB
    
    // Cleanup
    private var periodicCleanupTimer: Timer?
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    
    private let logger = Logger.shared
    
    // MARK: - Resource Types
    enum ResourceType {
        case image
        case video
        case audio
        case cache
        case buffer
        case network
        case file
        
        var defaultTimeout: TimeInterval {
            switch self {
            case .image, .cache: return 60
            case .video, .audio: return 120
            case .buffer: return 30
            case .network: return 300
            case .file: return 600
            }
        }
    }
    
    // MARK: - Initialization
    private init() {
        setupMemoryPressureHandling()
        startPeriodicCleanup()
        
        #if DEBUG
        startLeakDetection()
        #endif
    }
    
    // MARK: - Resource Registration
    func register<T: AnyObject>(
        _ resource: T,
        type: ResourceType,
        group: String? = nil,
        size: Int64 = 0,
        timeout: TimeInterval? = nil
    ) -> ResourceHandle<T> {
        let identifier = ResourceIdentifier()
        let managedResource = ManagedResource(
            resource: resource,
            type: type,
            size: size,
            timeout: timeout ?? type.defaultTimeout
        )
        
        managedResources[identifier] = managedResource
        
        // Add to group if specified
        if let group = group {
            resourceGroups[group, default: []].insert(identifier)
        }
        
        // Start cleanup timer
        startCleanupTimer(for: identifier)
        
        // Update statistics
        updateStatistics()
        
        logger.debug("Resource registered: \(identifier.id) of type \(String(describing: type))")
        
        return ResourceHandle(
            identifier: identifier,
            manager: self
        )
    }
    
    func unregister(_ identifier: ResourceIdentifier) {
        guard let resource = managedResources.removeValue(forKey: identifier) else { return }
        
        // Remove from groups
        for (group, var identifiers) in resourceGroups {
            identifiers.remove(identifier)
            if identifiers.isEmpty {
                resourceGroups.removeValue(forKey: group)
            } else {
                resourceGroups[group] = identifiers
            }
        }
        
        // Cancel cleanup timer
        cleanupTimers[identifier]?.invalidate()
        cleanupTimers.removeValue(forKey: identifier)
        
        // Cleanup resource
        resource.cleanup()
        
        // Update statistics
        updateStatistics()
        
        logger.debug("Resource unregistered: \(identifier.id)")
    }
    
    // MARK: - Resource Access
    func access(_ identifier: ResourceIdentifier) {
        guard let resource = managedResources[identifier] else { return }
        
        resource.lastAccessed = Date()
        resource.accessCount += 1
        
        // Reset cleanup timer
        resetCleanupTimer(for: identifier)
    }
    
    func getResource<T>(_ identifier: ResourceIdentifier, as type: T.Type) -> T? {
        access(identifier)
        return managedResources[identifier]?.resource as? T
    }
    
    // MARK: - Group Management
    func releaseGroup(_ group: String) {
        guard let identifiers = resourceGroups[group] else { return }
        
        for identifier in identifiers {
            unregister(identifier)
        }
        
        resourceGroups.removeValue(forKey: group)
        
        logger.info("Released resource group: \(group) with \(identifiers.count) resources")
    }
    
    func getGroupMemoryUsage(_ group: String) -> Int64 {
        guard let identifiers = resourceGroups[group] else { return 0 }
        
        return identifiers.compactMap { managedResources[$0]?.size }.reduce(0, +)
    }
    
    // MARK: - Cleanup
    private func startPeriodicCleanup() {
        periodicCleanupTimer = Timer.scheduledTimer(withTimeInterval: cleanupInterval, repeats: true) { _ in
            Task { @MainActor in
                self.performCleanup()
            }
        }
    }
    
    func performCleanup(force: Bool = false) {
        let startTime = Date()
        var cleanedCount = 0
        var freedMemory: Int64 = 0
        
        let cutoffTime = Date().addingTimeInterval(-resourceTimeout)
        
        for (identifier, resource) in managedResources {
            if force || resource.canCleanup(cutoffTime: cutoffTime) {
                freedMemory += resource.size
                unregister(identifier)
                cleanedCount += 1
            }
        }
        
        lastCleanup = Date()
        
        if cleanedCount > 0 {
            logger.info("Cleanup completed: \(cleanedCount) resources freed, \(freedMemory / 1_048_576)MB reclaimed")
        }
        
        // Check memory pressure
        if memoryFootprint > memoryLimit {
            performAggressiveCleanup()
        }
        
        updateStatistics()
    }
    
    private func performAggressiveCleanup() {
        logger.warning("Memory limit exceeded, performing aggressive cleanup")
        
        // Sort resources by last access time
        let sortedResources = managedResources.sorted { (a, b) in a.value.lastAccessed < b.value.lastAccessed }
        
        var freedMemory: Int64 = 0
        let targetMemory = memoryLimit * 3 / 4 // Free 25% of limit
        
        for (identifier, resource) in sortedResources {
            if resource.priority == .low || resource.accessCount < 2 {
                freedMemory += resource.size
                unregister(identifier)
                
                if memoryFootprint - freedMemory <= targetMemory {
                    break
                }
            }
        }
        
        logger.info("Aggressive cleanup freed \(freedMemory / 1_048_576)MB")
    }
    
    // MARK: - Memory Pressure
    private func setupMemoryPressureHandling() {
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])
        
        memoryPressureSource?.setEventHandler { [weak self] in
            self?.handleMemoryPressure()
        }
        
        memoryPressureSource?.resume()
    }
    
    private func handleMemoryPressure() {
        Task { @MainActor in
            logger.warning("Memory pressure detected")
            performCleanup(force: true)
        }
    }
    
    // MARK: - Cleanup Timers
    private func startCleanupTimer(for identifier: ResourceIdentifier) {
        guard let resource = managedResources[identifier] else { return }
        
        cleanupTimers[identifier] = Timer.scheduledTimer(withTimeInterval: resource.timeout, repeats: false) { _ in
            Task { @MainActor in
                self.unregister(identifier)
            }
        }
    }
    
    private func resetCleanupTimer(for identifier: ResourceIdentifier) {
        cleanupTimers[identifier]?.invalidate()
        startCleanupTimer(for: identifier)
    }
    
    // MARK: - Leak Detection
    #if DEBUG
    private func startLeakDetection() {
        Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { _ in
            Task { @MainActor in
                self.detectLeaks()
            }
        }
    }
    
    private func detectLeaks() {
        for (identifier, resource) in managedResources {
            if resource.isLikelyLeak() {
                logger.warning("Potential leak detected: \(identifier.id)")
                leaksDetected += 1
            }
        }
    }
    #endif
    
    // MARK: - Statistics
    private func updateStatistics() {
        totalResources = managedResources.count
        activeResources = managedResources.filter { Date().timeIntervalSince($0.value.lastAccessed) < 60 }.count
        memoryFootprint = managedResources.values.map { $0.size }.reduce(0, +)
    }
    
    func generateReport() -> ResourceReport {
        ResourceReport(
            totalResources: totalResources,
            activeResources: activeResources,
            memoryFootprint: memoryFootprint,
            groupCount: resourceGroups.count,
            leaksDetected: leaksDetected,
            lastCleanup: lastCleanup
        )
    }
    
    struct ResourceReport {
        let totalResources: Int
        let activeResources: Int
        let memoryFootprint: Int64
        let groupCount: Int
        let leaksDetected: Int
        let lastCleanup: Date
        
        var formattedMemory: String {
            let mb = Double(memoryFootprint) / 1_048_576
            return String(format: "%.1f MB", mb)
        }
    }
}

// MARK: - Resource Identifier
struct ResourceIdentifier: Hashable {
    let id = UUID()
}

// MARK: - Managed Resource
private class ManagedResource {
    let resource: AnyObject
    let type: ResourceManager.ResourceType
    let size: Int64
    let timeout: TimeInterval
    let createdAt = Date()
    var lastAccessed = Date()
    var accessCount = 0
    var priority: Priority = .normal
    
    enum Priority {
        case low, normal, high
    }
    
    init(resource: AnyObject, type: ResourceManager.ResourceType, size: Int64, timeout: TimeInterval) {
        self.resource = resource
        self.type = type
        self.size = size
        self.timeout = timeout
    }
    
    func canCleanup(cutoffTime: Date) -> Bool {
        if priority == .high {
            return false
        }
        
        if lastAccessed < cutoffTime {
            return true
        }
        
        if Date().timeIntervalSince(createdAt) > timeout {
            return true
        }
        
        return false
    }
    
    func isLikelyLeak() -> Bool {
        let age = Date().timeIntervalSince(createdAt)
        let timeSinceAccess = Date().timeIntervalSince(lastAccessed)
        
        // Resource created long ago but never accessed
        if age > 300 && accessCount == 0 {
            return true
        }
        
        // Resource not accessed for very long time
        if timeSinceAccess > 600 && accessCount < 5 {
            return true
        }
        
        return false
    }
    
    func cleanup() {
        // Perform type-specific cleanup
        switch type {
        case .image:
            // Clear image data
            break
        case .video, .audio:
            // Release media resources
            break
        case .cache:
            // Clear cache entries
            break
        case .buffer:
            // Release buffer memory
            break
        case .network:
            // Cancel network requests
            break
        case .file:
            // Close file handles
            break
        }
    }
}

// MARK: - Resource Handle
@MainActor
class ResourceHandle<T: AnyObject> {
    private let identifier: ResourceIdentifier
    private weak var manager: ResourceManager?
    private var isValid = true
    
    init(identifier: ResourceIdentifier, manager: ResourceManager) {
        self.identifier = identifier
        self.manager = manager
    }
    
    var resource: T? {
        guard isValid else { return nil }
        return manager?.getResource(identifier, as: T.self)
    }
    
    func access() {
        manager?.access(identifier)
    }
    
    func release() {
        guard isValid else { return }
        manager?.unregister(identifier)
        isValid = false
    }
    
    deinit {
        // Don't call release() in deinit to avoid MainActor isolation issues
        // Resources will be cleaned up when the ResourceManager is deallocated
        // or during periodic cleanup cycles
    }
}

// MARK: - Auto Cleanup Cache
final class AutoCleanupCache<Key: Hashable, Value: AnyObject> {
    private var cache: [Key: CacheEntry<Value>] = [:]
    private let maxSize: Int
    private let ttl: TimeInterval
    private var cleanupTimer: Timer?
    private let lock = NSLock()
    
    struct CacheEntry<T> {
        let value: T
        let timestamp: Date
        var accessCount: Int
    }
    
    init(maxSize: Int = 100, ttl: TimeInterval = 300) {
        self.maxSize = maxSize
        self.ttl = ttl
        startCleanupTimer()
    }
    
    func set(_ value: Value, for key: Key) {
        lock.lock()
        defer { lock.unlock() }
        
        cache[key] = CacheEntry(value: value, timestamp: Date(), accessCount: 0)
        
        // Evict if over size limit
        if cache.count > maxSize {
            evictLRU()
        }
    }
    
    func get(_ key: Key) -> Value? {
        lock.lock()
        defer { lock.unlock() }
        
        guard var entry = cache[key] else { return nil }
        
        // Check TTL
        if Date().timeIntervalSince(entry.timestamp) > ttl {
            cache.removeValue(forKey: key)
            return nil
        }
        
        entry.accessCount += 1
        cache[key] = entry
        
        return entry.value
    }
    
    func remove(_ key: Key) {
        lock.lock()
        defer { lock.unlock() }
        
        cache.removeValue(forKey: key)
    }
    
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        cache.removeAll()
    }
    
    private func evictLRU() {
        // Find least recently used entry
        let lru = cache.min { a, b in
            if a.value.accessCount == b.value.accessCount {
                return a.value.timestamp < b.value.timestamp
            }
            return a.value.accessCount < b.value.accessCount
        }
        
        if let key = lru?.key {
            cache.removeValue(forKey: key)
        }
    }
    
    private func startCleanupTimer() {
        cleanupTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { _ in
            self.removeExpired()
        }
    }
    
    private func removeExpired() {
        lock.lock()
        defer { lock.unlock() }
        
        let now = Date()
        cache = cache.filter { _, entry in
            now.timeIntervalSince(entry.timestamp) <= ttl
        }
    }
    
    deinit {
        cleanupTimer?.invalidate()
    }
}
