// AUTORESOLVE V3.0 - INTELLIGENT REQUEST CACHING
// Advanced caching with TTL, memory management, and invalidation

import Foundation
import Combine
import SwiftUI
import CryptoKit

/// Advanced caching system with memory management
public class RequestCache: ObservableObject {
    public static let shared = RequestCache()
    
    // Cache storage
    private var memoryCache: NSCache<NSString, CacheEntry> = {
        let cache = NSCache<NSString, CacheEntry>()
        cache.countLimit = 100  // Maximum 100 items
        cache.totalCostLimit = 50 * 1024 * 1024  // 50MB
        return cache
    }()
    private var diskCache: URL
    private let queue = DispatchQueue(label: "cache.queue", attributes: .concurrent)
    
    // Configuration
    public struct Configuration {
        let maxMemorySize: Int          // Bytes
        let maxDiskSize: Int            // Bytes
        let defaultTTL: TimeInterval    // Seconds
        let cleanupInterval: TimeInterval
        
        public static let `default` = Configuration(
            maxMemorySize: 50 * 1024 * 1024,    // 50MB
            maxDiskSize: 200 * 1024 * 1024,     // 200MB
            defaultTTL: 300,                    // 5 minutes
            cleanupInterval: 60                 // 1 minute
        )
    }
    
    private let configuration: Configuration
    private var cleanupTimer: Timer?
    
    // Statistics
    @Published public var statistics = CacheStatistics()
    
    public struct CacheStatistics {
        var hits = 0
        var misses = 0
        var evictions = 0
        var currentMemorySize = 0
        var currentDiskSize = 0
        
        var hitRate: Double {
            let total = hits + misses
            return total > 0 ? Double(hits) / Double(total) : 0
        }
    }
    
    private init(configuration: Configuration = .default) {
        self.configuration = configuration
        
        // Setup disk cache
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        diskCache = cacheDir.appendingPathComponent("AutoResolveCache")
        try? FileManager.default.createDirectory(at: diskCache, withIntermediateDirectories: true)
        
        // Configure memory cache
        memoryCache.countLimit = 1000
        memoryCache.totalCostLimit = configuration.maxMemorySize
        memoryCache.evictsObjectsWithDiscardedContent = true
        
        // Start cleanup timer
        startCleanupTimer()
        
        // Load statistics
        loadStatistics()
    }
    
    // MARK: - Public Methods
    
    /// Cache a response
    public func cache<T: Codable>(
        _ value: T,
        for key: String,
        ttl: TimeInterval? = nil,
        category: CacheCategory = .general
    ) {
        let cacheKey = generateKey(for: key, category: category)
        let entry = RequestCacheEntry(
            value: value,
            expiresAt: Date().addingTimeInterval(ttl ?? configuration.defaultTTL),
            category: category
        )
        
        queue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            // Memory cache
            if let data = try? JSONEncoder().encode(entry) {
                let cost = data.count
                self.memoryCache.setObject(entry.toCacheEntry(), forKey: cacheKey as NSString, cost: cost)
                self.statistics.currentMemorySize += cost
            }
            
            // Disk cache for important items
            if category.shouldPersist {
                self.persistToDisk(entry, key: cacheKey)
            }
            
            logDebug("[Cache] Cached \(key) (TTL: \(ttl ?? self.configuration.defaultTTL)s)", category: .storage)
        }
    }
    
    /// Retrieve from cache
    public func retrieve<T: Codable>(
        _ type: T.Type,
        for key: String,
        category: CacheCategory = .general
    ) -> T? {
        let cacheKey = generateKey(for: key, category: category)
        
        // Check memory cache
        if let entry = memoryCache.object(forKey: cacheKey as NSString) {
            if entry.isValid {
                statistics.hits += 1
                logDebug("[Cache] Memory hit: \(key)", category: .storage)
                return entry.getValue(as: type)
            } else {
                // Expired, remove it
                invalidate(key: key, category: category)
            }
        }
        
        // Check disk cache
        if let entry = loadFromDisk(key: cacheKey, type: RequestCacheEntry.self) {
            if entry.isValid {
                // Promote to memory cache
                memoryCache.setObject(entry.toCacheEntry(), forKey: cacheKey as NSString)
                statistics.hits += 1
                logDebug("[Cache] Disk hit: \(key)", category: .storage)
                return entry.getValue(as: type)
            } else {
                // Expired, remove it
                deleteDiskCache(key: cacheKey)
            }
        }
        
        statistics.misses += 1
        logDebug("[Cache] Miss: \(key)", category: .storage)
        return nil
    }
    
    /// Cache-first request execution
    public func cachedRequest<T: Codable>(
        key: String,
        ttl: TimeInterval = 300,
        category: CacheCategory = .api,
        fallback: @escaping () -> AnyPublisher<T, Error>
    ) -> AnyPublisher<T, Error> {
        // Try cache first
        if let cached = retrieve(T.self, for: key, category: category) {
            logInfo("[Cache] Using cached response for \(key)", category: .storage)
            return Just(cached)
                .setFailureType(to: Error.self)
                .eraseToAnyPublisher()
        }
        
        // Execute request and cache result
        return fallback()
            .handleEvents(receiveOutput: { [weak self] value in
                self?.cache(value, for: key, ttl: ttl, category: category)
            })
            .eraseToAnyPublisher()
    }
    
    /// Invalidate cache
    public func invalidate(key: String? = nil, category: CacheCategory? = nil) {
        queue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            if let key = key {
                let cacheKey = self.generateKey(for: key, category: category ?? .general)
                if let entry = self.memoryCache.object(forKey: cacheKey as NSString) {
                    self.statistics.currentMemorySize = max(0, self.statistics.currentMemorySize - entry.data.count)
                }
                self.memoryCache.removeObject(forKey: cacheKey as NSString)
                self.deleteDiskCache(key: cacheKey)
                logDebug("[Cache] Invalidated: \(key)", category: .storage)
            } else {
                // Clear all or by category
                self.clearCache(category: category)
            }
            
            self.statistics.evictions += 1
        }
    }
    
    /// Clear entire cache
    public func clearCache(category: CacheCategory? = nil) {
        queue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            if let category = category {
                // Clear specific category
                // Note: NSCache doesn't support enumeration, so we track keys separately
                logInfo("[Cache] Clearing category: \(category)", category: .storage)
            } else {
                // Clear all
                self.memoryCache.removeAllObjects()
                self.clearDiskCache()
                self.statistics = CacheStatistics()
                logInfo("[Cache] Cleared all caches", category: .storage)
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func generateKey(for key: String, category: CacheCategory) -> String {
        let combined = "\(category.rawValue):\(key)"
        
        // Hash long keys
        if combined.count > 100 {
            guard let data = combined.data(using: .utf8) else {
                return "invalid_key_\(UUID().uuidString)"
            }
            let hash = SHA256.hash(data: data)
            return hash.compactMap { String(format: "%02x", $0) }.joined()
        }
        
        return combined
    }
    
    private func persistToDisk<T: Codable>(_ value: T, key: String) {
        let url = diskCache.appendingPathComponent(key)
        
        do {
            let data = try JSONEncoder().encode(value)
            try data.write(to: url)
            statistics.currentDiskSize += data.count
            
            // Check disk size limit
            if statistics.currentDiskSize > configuration.maxDiskSize {
                performDiskCleanup()
            }
        } catch {
            logError("[Cache] Disk write failed: \(error)", category: .storage)
        }
    }
    
    private func loadFromDisk<T: Decodable>(key: String, type: T.Type) -> T? {
        let url = diskCache.appendingPathComponent(key)
        
        guard let data = try? Data(contentsOf: url) else { return nil }
        
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            logError("[Cache] Disk read failed: \(error)", category: .storage)
            return nil
        }
    }
    
    private func deleteDiskCache(key: String) {
        let url = diskCache.appendingPathComponent(key)
        try? FileManager.default.removeItem(at: url)
    }
    
    private func clearDiskCache() {
        try? FileManager.default.removeItem(at: diskCache)
        try? FileManager.default.createDirectory(at: diskCache, withIntermediateDirectories: true)
        statistics.currentDiskSize = 0
    }
    
    private func performDiskCleanup() {
        // Remove oldest files until under limit
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: diskCache,
            includingPropertiesForKeys: [.creationDateKey, .fileSizeKey]
        ) else { return }
        
        let sorted = files.sorted { url1, url2 in
            let date1 = (try? url1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
            let date2 = (try? url2.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
            return date1 < date2
        }
        
        var totalSize = 0
        for url in sorted {
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            totalSize += size
            
            if totalSize > configuration.maxDiskSize {
                try? FileManager.default.removeItem(at: url)
                statistics.evictions += 1
            }
        }
        
        statistics.currentDiskSize = totalSize
    }
    
    private func startCleanupTimer() {
        cleanupTimer = Timer.scheduledTimer(withTimeInterval: configuration.cleanupInterval, repeats: true) { [weak self] _ in
            self?.performCleanup()
        }
    }
    
    private func performCleanup() {
        queue.async(flags: .barrier) { [weak self] in
            self?.performDiskCleanup()
            logDebug("[Cache] Cleanup completed", category: .storage)
        }
    }
    
    private func loadStatistics() {
        // Calculate disk cache size
        if let files = try? FileManager.default.contentsOfDirectory(
            at: diskCache,
            includingPropertiesForKeys: [.fileSizeKey]
        ) {
            statistics.currentDiskSize = files.reduce(0) { total, url in
                total + ((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
        }
    }
}

// MARK: - Supporting Types

public enum CacheCategory: String, Codable {
    case general = "general"
    case api = "api"
    case media = "media"
    case analysis = "analysis"
    case thumbnail = "thumbnail"
    case project = "project"
    
    var shouldPersist: Bool {
        switch self {
        case .media, .analysis, .project:
            return true
        default:
            return false
        }
    }
}

class RequestCacheEntry: Codable {
    let data: Data
    let expiresAt: Date
    let category: CacheCategory
    let createdAt: Date
    
    init<T: Codable>(value: T, expiresAt: Date, category: CacheCategory) {
        self.data = (try? JSONEncoder().encode(value)) ?? Data()
        self.expiresAt = expiresAt
        self.category = category
        self.createdAt = Date()
    }
    
    init(data: Data, expiresAt: Date, category: CacheCategory, createdAt: Date) {
        self.data = data
        self.expiresAt = expiresAt
        self.category = category
        self.createdAt = createdAt
    }
    
    
    func getValue<T: Codable>(as type: T.Type) -> T? {
        try? JSONDecoder().decode(type, from: data)
    }
}

// MARK: - Cache Monitor View

public struct CacheMonitorView: View {
    @ObservedObject private var cache = RequestCache.shared
    @State private var showDetails = false
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack {
                Image(systemName: "memorychip")
                    .font(.system(size: 12))
                    .foregroundColor(.blue)
                
                Text("Cache")
                    .font(.system(size: 12, weight: .medium))
                
                Text("\(Int(cache.statistics.hitRate * 100))% hit")
                    .font(.system(size: 11))
                    .foregroundColor(cache.statistics.hitRate > 0.7 ? .green : .orange)
                
                Spacer()
                
                Button(action: { showDetails.toggle() }) {
                    Image(systemName: showDetails ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
            }
            
            if showDetails {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Hits/Misses:")
                        Text("\(cache.statistics.hits)/\(cache.statistics.misses)")
                            .font(.system(.caption, design: .monospaced))
                    }
                    .font(.caption)
                    
                    HStack {
                        Text("Memory:")
                        Text(formatBytes(cache.statistics.currentMemorySize))
                            .font(.system(.caption, design: .monospaced))
                    }
                    .font(.caption)
                    
                    HStack {
                        Text("Disk:")
                        Text(formatBytes(cache.statistics.currentDiskSize))
                            .font(.system(.caption, design: .monospaced))
                    }
                    .font(.caption)
                    
                    Button("Clear Cache") {
                        cache.clearCache()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .padding(.top, 4)
                }
                .padding(.leading, 18)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color(white: 0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                )
        )
        .animation(.smooth, value: showDetails)
    }
    
    private func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// Removed external CacheEntry alias; using local CacheEntry defined in Backend/CacheEntryFix.swift
