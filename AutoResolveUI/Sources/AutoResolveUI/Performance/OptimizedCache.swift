import AppKit
// AUTORESOLVE V3.0 - OPTIMIZED CACHE SYSTEM
import Combine
// Multi-tier caching with compression and intelligent prefetching

import Foundation
import SwiftUI
import Compression
import CryptoKit
import os.log
import AVFoundation
import QuickLookThumbnailing

// MARK: - Optimized Cache Manager
@MainActor
final class OptimizedCacheManager {
    static let shared = OptimizedCacheManager()
    
    // Cache instances
    private let thumbnailCache: ThumbnailCache
    private let metadataCache: MetadataCache
    private let frameCache: FrameCache
    
    // Statistics
    @Published var totalHits = 0
    @Published var totalMisses = 0
    @Published var memoryUsage: Int64 = 0
    @Published var diskUsage: Int64 = 0
    @Published var compressionRatio: Double = 1.0
    
    private let logger = Logger.shared
    
    init() {
        thumbnailCache = ThumbnailCache()
        metadataCache = MetadataCache()
        frameCache = FrameCache()
        
        setupCachePolicies()
    }
    
    private func setupCachePolicies() {
        // Configure cache policies based on available memory
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let cacheMemory = min(Int64(totalMemory) / 10, 200_000_000) // Max 200MB or 10% of RAM
        
        thumbnailCache.memoryLimit = cacheMemory / Int64(3)
        metadataCache.memoryLimit = cacheMemory / Int64(6)
        frameCache.memoryLimit = cacheMemory / Int64(2)
        
        logger.info("Cache initialized with \(cacheMemory / 1_048_576)MB memory limit")
    }
    
    // MARK: - Public API
    func thumbnail(for url: URL, size: CGSize) async -> NSImage? {
        let key = ThumbnailKey(url: url, size: size)
        
        if let cached = thumbnailCache.get(key) {
            totalHits += 1
            return cached
        }
        
        totalMisses += 1
        
        // Generate thumbnail
        if let thumbnail = await generateThumbnail(for: url, size: size) {
            thumbnailCache.set(thumbnail, for: key)
            return thumbnail
        }
        
        return nil
    }
    
    func metadata(for url: URL) async -> FileMetadata? {
        if let cached = metadataCache.get(url) {
            totalHits += 1
            return cached
        }
        
        totalMisses += 1
        
        // Load metadata
        if let metadata = await loadMetadata(for: url) {
            metadataCache.set(metadata, for: url)
            return metadata
        }
        
        return nil
    }
    
    func frame(at time: TimeInterval, from url: URL) async -> CGImage? {
        let key = FrameKey(url: url, time: time)
        
        if let cached = frameCache.get(key) {
            totalHits += 1
            return cached
        }
        
        totalMisses += 1
        
        // Extract frame
        if let frame = await extractFrame(at: time, from: url) {
            frameCache.set(frame, for: key)
            return frame
        }
        
        return nil
    }
    
    // MARK: - Cache Management
    func clearAll() {
        thumbnailCache.clear()
        metadataCache.clear()
        frameCache.clear()
        
        totalHits = 0
        totalMisses = 0
        
        logger.info("All caches cleared")
    }
    
    func warmUp(urls: [URL]) {
        Task {
            for url in urls.prefix(20) { // Limit warmup
                _ = await metadata(for: url)
                _ = await thumbnail(for: url, size: CGSize(width: 280, height: 160))
            }
        }
    }
    
    var hitRate: Double {
        let total = totalHits + totalMisses
        return total > 0 ? Double(totalHits) / Double(total) : 0
    }
    
    // MARK: - Generation Methods
    private func generateThumbnail(for url: URL, size: CGSize) async -> NSImage? {
        // Real thumbnail generation using AVAssetImageGenerator
        do {
            let asset = AVAsset(url: url)
            
            // Check if it's a video file
            let videoTracks = try await asset.loadTracks(withMediaType: .video)
            if !videoTracks.isEmpty {
                // Generate video thumbnail
                let generator = AVAssetImageGenerator(asset: asset)
                generator.appliesPreferredTrackTransform = true
                generator.maximumSize = size
                
                let duration = try await asset.load(.duration)
                let thumbnailTime = CMTime(seconds: min(1, CMTimeGetSeconds(duration) / 10), preferredTimescale: 600)
                
                let cgImage = try generator.copyCGImage(at: thumbnailTime, actualTime: nil)
                return NSImage(cgImage: cgImage, size: size)
            }
            
            // For non-video files, try QuickLook thumbnail
            let generator = QLThumbnailGenerator.shared
            let scale = await NSScreen.main?.backingScaleFactor ?? 2.0
            
            let request = QLThumbnailGenerator.Request(
                fileAt: url,
                size: size,
                scale: scale,
                representationTypes: .all
            )
            
            let representation = try await generator.generateBestRepresentation(for: request)
            return representation.nsImage
            
        } catch {
            // Fallback to system icon
            return NSWorkspace.shared.icon(forFile: url.path)
        }
    }
    
    private func loadMetadata(for url: URL) async -> FileMetadata? {
        // Implementation would load actual metadata
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return FileMetadata(
            size: attributes?[.size] as? Int64 ?? 0,
            createdDate: attributes?[.creationDate] as? Date ?? Date(),
            modifiedDate: attributes?[.modificationDate] as? Date ?? Date(),
            type: url.pathExtension
        )
    }
    
    private func extractFrame(at time: TimeInterval, from url: URL) async -> CGImage? {
        // Real frame extraction using AVAssetImageGenerator
        do {
            let asset = AVAsset(url: url)
            
            // Ensure this is a video file
            let videoTracks = try await asset.loadTracks(withMediaType: .video)
            guard !videoTracks.isEmpty else { return nil }
            
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceAfter = .zero
            generator.requestedTimeToleranceBefore = .zero
            
            // Set maximum size for performance
            generator.maximumSize = CGSize(width: 1920, height: 1080)
            
            let requestedTime = CMTime(seconds: time, preferredTimescale: 600)
            
            // Extract the frame
            var actualTime = CMTime.zero
            let cgImage = try generator.copyCGImage(at: requestedTime, actualTime: &actualTime)
            
            return cgImage
            
        } catch {
            logger.error("Failed to extract frame at \(time)s from \(url.lastPathComponent): \(error.localizedDescription)")
            return nil
        }
    }
}

// MARK: - Multi-Tier Cache Base
class MultiTierCache<Key: Hashable & CacheKey, Value: AnyObject> {
    // Memory cache
    private let memoryCache = NSCache<NSString, CacheBox<Value>>()
    var memoryLimit: Int64 = 100_000_000 { // 100MB default
        didSet {
            memoryCache.totalCostLimit = Int(memoryLimit)
        }
    }
    
    // Disk cache
    private let diskCacheURL: URL
    var diskCacheEnabled: Bool { false }
    var diskLimit: Int64 = 1_000_000_000 // 1GB default
    private let diskQueue = DispatchQueue(label: "cache.disk", qos: .background)
    
    // Compression
    var compressionEnabled = true
    private let compressionAlgorithm = COMPRESSION_ZLIB
    
    // Statistics
    private var hits = 0
    private var misses = 0
    private let lock = NSLock()
    
    init() {
        // Setup disk cache directory
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        diskCacheURL = cacheDir.appendingPathComponent("AutoResolve/\(String(describing: Self.self))")
        try? FileManager.default.createDirectory(at: diskCacheURL, withIntermediateDirectories: true)
        
        // Configure memory cache
        memoryCache.totalCostLimit = Int(memoryLimit)
        memoryCache.countLimit = 1000
    }
    
    // MARK: - Cache Operations
    func get(_ key: Key) -> Value? {
        // Check memory cache
        if let cached = memoryCache.object(forKey: key.cacheKey) {
            hits += 1
            cached.lastAccessed = Date()
            return cached.value
        }
        
        // Check disk cache
        if diskCacheEnabled, let value = loadFromDisk(key: key) {
            hits += 1
            // Promote to memory cache
            let box = CacheBox(value: value)
            memoryCache.setObject(box, forKey: key.cacheKey, cost: estimateCost(value))
            return value
        }
        
        misses += 1
        return nil
    }
    
    func set(_ value: Value, for key: Key) {
        let box = CacheBox(value: value)
        let cost = estimateCost(value)
        
        // Store in memory cache
        memoryCache.setObject(box, forKey: key.cacheKey, cost: cost)
        
        // Store on disk if enabled
        if diskCacheEnabled {
            diskQueue.async {
                self.saveToDisk(value: value, key: key)
            }
        }
    }
    
    func remove(_ key: Key) {
        memoryCache.removeObject(forKey: key.cacheKey)
        
        if diskCacheEnabled {
            diskQueue.async {
                self.removeFromDisk(key: key)
            }
        }
    }
    
    func clear() {
        memoryCache.removeAllObjects()
        
        if diskCacheEnabled {
            diskQueue.async {
                try? FileManager.default.removeItem(at: self.diskCacheURL)
                try? FileManager.default.createDirectory(at: self.diskCacheURL, withIntermediateDirectories: true)
            }
        }
        
        hits = 0
        misses = 0
    }
    
    // MARK: - Disk Operations
    private func diskPath(for key: Key) -> URL {
        let hash = SHA256.hash(data: key.stringValue.data(using: .utf8)!)
        let hashString = hash.compactMap { String(format: "%02x", $0) }.joined()
        return diskCacheURL.appendingPathComponent(hashString)
    }
    
    private func saveToDisk(value: Value, key: Key) {
        guard let data = serialize(value) else { return }
        
        let finalData: Data
        if compressionEnabled, let compressed = compress(data) {
            finalData = compressed
        } else {
            finalData = data
        }
        
        let path = diskPath(for: key)
        try? finalData.write(to: path)
        
        // Check disk usage
        cleanDiskCacheIfNeeded()
    }
    
    private func loadFromDisk(key: Key) -> Value? {
        let path = diskPath(for: key)
        guard let data = try? Data(contentsOf: path) else { return nil }
        
        let finalData: Data
        if compressionEnabled, let decompressed = decompress(data) {
            finalData = decompressed
        } else {
            finalData = data
        }
        
        return deserialize(finalData)
    }
    
    private func removeFromDisk(key: Key) {
        let path = diskPath(for: key)
        try? FileManager.default.removeItem(at: path)
    }
    
    private func cleanDiskCacheIfNeeded() {
        let fileManager = FileManager.default
        
        do {
            let files = try fileManager.contentsOfDirectory(at: diskCacheURL, includingPropertiesForKeys: [.fileSizeKey, .contentAccessDateKey])
            
            var totalSize: Int64 = 0
            var fileInfos: [(url: URL, size: Int64, accessed: Date)] = []
            
            for file in files {
                let attributes = try file.resourceValues(forKeys: [.fileSizeKey, .contentAccessDateKey])
                let size = Int64(attributes.fileSize ?? 0)
                let accessed = attributes.contentAccessDate ?? Date.distantPast
                
                totalSize += size
                fileInfos.append((file, size, accessed))
            }
            
            if totalSize > diskLimit {
                // Sort by last accessed (LRU)
                fileInfos.sort { $0.accessed < $1.accessed }
                
                // Remove oldest files until under limit
                var removed: Int64 = 0
                for info in fileInfos {
                    try fileManager.removeItem(at: info.url)
                    removed += info.size
                    
                    if totalSize - removed <= diskLimit * 3 / 4 {
                        break
                    }
                }
            }
        } catch {
            // Ignore errors
        }
    }
    
    // MARK: - Compression
    private func compress(_ data: Data) -> Data? {
        return data.withUnsafeBytes { bytes in
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
            defer { buffer.deallocate() }
            
            let compressedSize = compression_encode_buffer(
                buffer, data.count,
                bytes.bindMemory(to: UInt8.self).baseAddress!, data.count,
                nil, compressionAlgorithm
            )
            
            guard compressedSize > 0 else { return nil }
            return Data(bytes: buffer, count: compressedSize)
        }
    }
    
    private func decompress(_ data: Data) -> Data? {
        return data.withUnsafeBytes { bytes in
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count * 4)
            defer { buffer.deallocate() }
            
            let decompressedSize = compression_decode_buffer(
                buffer, data.count * 4,
                bytes.bindMemory(to: UInt8.self).baseAddress!, data.count,
                nil, compressionAlgorithm
            )
            
            guard decompressedSize > 0 else { return nil }
            return Data(bytes: buffer, count: decompressedSize)
        }
    }
    
    // MARK: - Serialization (Override in subclasses)
    func serialize(_ value: Value) -> Data? {
        return nil
    }
    
    func deserialize(_ data: Data) -> Value? {
        return nil
    }
    
    func estimateCost(_ value: Value) -> Int {
        return 1000 // Default cost
    }
    
    // MARK: - Statistics
    var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0
    }
}

// MARK: - Cache Box
private class CacheBox<T> {
    let value: T
    var lastAccessed: Date
    
    init(value: T) {
        self.value = value
        self.lastAccessed = Date()
    }
}

// MARK: - Cache Key Protocol
protocol CacheKey {
    var stringValue: String { get }
}

extension CacheKey where Self: Hashable {
    var cacheKey: NSString {
        NSString(string: stringValue)
    }
}

extension URL: CacheKey {
    var stringValue: String {
        absoluteString
    }
}

// MARK: - Thumbnail Cache
final class ThumbnailCache: MultiTierCache<ThumbnailKey, NSImage> {
    override var diskCacheEnabled: Bool { true }
    
    override init() {
        super.init()
    }
    
    override func serialize(_ value: NSImage) -> Data? {
        guard let tiffData = value.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData) else { return nil }
        
        return bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.8])
    }
    
    override func deserialize(_ data: Data) -> NSImage? {
        return NSImage(data: data)
    }
}

struct ThumbnailKey: Hashable, CacheKey {
    let url: URL
    let size: CGSize
    
    var stringValue: String {
        "\(url.path)_\(Int(size.width))x\(Int(size.height))"
    }
}

// MARK: - Metadata Cache
final class MetadataCache: MultiTierCache<URL, FileMetadata> {
    override var diskCacheEnabled: Bool { true }
    
    override init() {
        super.init()
    }
    
    override func serialize(_ value: FileMetadata) -> Data? {
        try? JSONEncoder().encode(value)
    }
    
    override func deserialize(_ data: Data) -> FileMetadata? {
        try? JSONDecoder().decode(FileMetadata.self, from: data)
    }
}

class FileMetadata: NSObject, Codable {
    let size: Int64
    let createdDate: Date
    let modifiedDate: Date
    let type: String
    
    init(size: Int64, createdDate: Date, modifiedDate: Date, type: String) {
        self.size = size
        self.createdDate = createdDate
        self.modifiedDate = modifiedDate
        self.type = type
        super.init()
    }
}

// MARK: - Frame Cache
final class FrameCache: MultiTierCache<FrameKey, CGImage> {
    override var diskCacheEnabled: Bool { false } // Frames are memory-only
    
    override init() {
        super.init()
    }
    
    // MARK: - Additional Methods for MotionGraphics compatibility
    
    func clearCache() {
        clear()
    }
    
    func getFrame(for key: FrameKey) -> CGImage? {
        return get(key)
    }
    
    func cacheFrame(_ frame: CGImage, for key: FrameKey) {
        set(frame, for: key)
    }
    
    var memoryUsage: Int64 {
        return memoryLimit
    }
}

struct FrameKey: Hashable, CacheKey {
    let url: URL
    let time: TimeInterval
    
    var stringValue: String {
        "\(url.path)_\(time)"
    }
}

// MARK: - Prefetcher
final class CachePrefetcher {
    private let cacheManager = OptimizedCacheManager.shared
    private var prefetchQueue = DispatchQueue(label: "cache.prefetch", qos: .background)
    private var prefetchTasks: [URL: Task<Void, Never>] = [:]
    
    func prefetchThumbnails(for urls: [URL], size: CGSize) {
        for url in urls {
            guard prefetchTasks[url] == nil else { continue }
            
            let task = Task { @MainActor in
                _ = await cacheManager.thumbnail(for: url, size: size)
            }
            
            prefetchTasks[url] = task
        }
    }
    
    func prefetchMetadata(for urls: [URL]) {
        for url in urls {
            Task {
                _ = await cacheManager.metadata(for: url)
            }
        }
    }
    
    func cancelAll() {
        for task in prefetchTasks.values {
            task.cancel()
        }
        prefetchTasks.removeAll()
    }
}
