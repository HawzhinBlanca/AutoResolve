import AppKit
// AUTORESOLVE V3.0 - MEDIA ITEM MODEL
import Combine
// Professional media item with thumbnail generation and metadata

import Foundation
import SwiftUI
import AVFoundation
import QuickLookThumbnailing
import UniformTypeIdentifiers

// MARK: - Media Pool Item Model
@MainActor
class MediaPoolItem: ObservableObject, Identifiable, Hashable, Transferable {
    public let id = UUID()
    let url: URL
    @Published var name: String
    @Published var thumbnail: NSImage?
    @Published var duration: Double = 0
    @Published var fileSize: Int64 = 0
    @Published var dateCreated: Date
    @Published var dateModified: Date
    @Published var hasVideo: Bool = false
    @Published var hasAudio: Bool = false
    @Published var videoCodec: String?
    @Published var audioCodec: String?
    @Published var resolution: CGSize?
    @Published var frameRate: Double?
    @Published var metadata: [String: Any] = [:]
    
    // Thumbnail generation
    private var thumbnailTask: Task<Void, Never>?
    private let thumbnailGenerator = QLThumbnailGenerator.shared
    
    // Transferable protocol
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { item in
            SentTransferredFile(item.url)
        } importing: { received in
            let copy = URL.documentsDirectory.appendingPathComponent(received.file.lastPathComponent)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return await MediaPoolItem(url: copy)
        }
    }
    
    init(url: URL) {
        self.url = url
        self.name = url.deletingPathExtension().lastPathComponent
        
        // Get file attributes
        if let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) {
            self.fileSize = attributes[.size] as? Int64 ?? 0
            self.dateCreated = attributes[.creationDate] as? Date ?? Date()
            self.dateModified = attributes[.modificationDate] as? Date ?? Date()
        } else {
            self.dateCreated = Date()
            self.dateModified = Date()
        }
        
        // Load media properties
        Task {
            await loadMediaProperties()
            await generateThumbnail()
        }
    }
    
    // MARK: - Hashable
    nonisolated static func == (lhs: MediaPoolItem, rhs: MediaPoolItem) -> Bool {
        lhs.id == rhs.id
    }
    
    nonisolated func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    // MARK: - Media Properties
    @MainActor
    func loadMediaProperties() async {
        let asset = AVAsset(url: url)
        
        do {
            // Load duration
            let duration = try await asset.load(.duration)
            self.duration = CMTimeGetSeconds(duration)
            
            // Check for video tracks
            let videoTracks = try await asset.loadTracks(withMediaType: .video)
            self.hasVideo = !videoTracks.isEmpty
            
            if let videoTrack = videoTracks.first {
                // Load video properties
                let size = try await videoTrack.load(.naturalSize)
                self.resolution = size
                
                let frameRate = try await videoTrack.load(.nominalFrameRate)
                self.frameRate = Double(frameRate)
                
                // Get codec info
                let formatDescriptions = try await videoTrack.load(.formatDescriptions)
                if let desc = formatDescriptions.first {
                    let fourCC = CMFormatDescriptionGetMediaSubType(desc)
                    self.videoCodec = fourCCString(fourCC)
                }
            }
            
            // Check for audio tracks
            let audioTracks = try await asset.loadTracks(withMediaType: .audio)
            self.hasAudio = !audioTracks.isEmpty
            
            if let audioTrack = audioTracks.first {
                // Get audio codec info
                let formatDescriptions = try await audioTrack.load(.formatDescriptions)
                if let desc = formatDescriptions.first {
                    let fourCC = CMFormatDescriptionGetMediaSubType(desc)
                    self.audioCodec = fourCCString(fourCC)
                }
            }
            
            // Load metadata
            let metadata = try await asset.load(.metadata)
            for item in metadata {
                if let key = item.commonKey?.rawValue,
                   let value = try await item.load(.value) {
                    self.metadata[key] = value
                }
            }
        } catch {
            print("Failed to load media properties: \(error)")
        }
    }
    
    // MARK: - Thumbnail Generation
    @MainActor
    func generateThumbnail() async {
        // Cancel previous task if any
        thumbnailTask?.cancel()
        
        thumbnailTask = Task { @MainActor in
            let size = CGSize(width: 280, height: 160)
            let scale = NSScreen.main?.backingScaleFactor ?? 2.0
            
            let request = QLThumbnailGenerator.Request(
                fileAt: url,
                size: size,
                scale: scale,
                representationTypes: .all
            )
            
            do {
                let representation = try await thumbnailGenerator.generateBestRepresentation(for: request)
                if !Task.isCancelled {
                    self.thumbnail = representation.nsImage
                }
            } catch {
                // Fallback to AVAsset thumbnail for video
                if hasVideo {
                    await generateVideoThumbnail(size: size)
                } else {
                    // Use generic icon
                    self.thumbnail = NSWorkspace.shared.icon(forFile: url.path)
                }
            }
        }
    }
    
    @MainActor
    private func generateVideoThumbnail(size: CGSize) async {
        let asset = AVAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = size
        
        let time = CMTime(seconds: min(1, duration / 10), preferredTimescale: 600)
        
        do {
            let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
            self.thumbnail = NSImage(cgImage: cgImage, size: size)
        } catch {
            print("Failed to generate video thumbnail: \(error)")
            self.thumbnail = NSWorkspace.shared.icon(forFile: url.path)
        }
    }
    
    // MARK: - Formatted Properties
    var formattedDuration: String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.unitsStyle = .abbreviated
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: duration) ?? "0s"
    }
    
    var formattedSize: String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: fileSize)
    }
    
    var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: dateModified)
    }
    
    var formattedResolution: String? {
        guard let resolution = resolution else { return nil }
        return "\(Int(resolution.width))×\(Int(resolution.height))"
    }
    
    var formattedFrameRate: String? {
        guard let frameRate = frameRate else { return nil }
        return String(format: "%.2f fps", frameRate)
    }
    
    // MARK: - Utilities
    private func fourCCString(_ fourCC: FourCharCode) -> String {
        let chars = [
            Character(UnicodeScalar((fourCC >> 24) & 0xFF)!),
            Character(UnicodeScalar((fourCC >> 16) & 0xFF)!),
            Character(UnicodeScalar((fourCC >> 8) & 0xFF)!),
            Character(UnicodeScalar(fourCC & 0xFF)!)
        ]
        return String(chars)
    }
}

// MARK: - Thumbnail Cache Manager
actor ThumbnailCacheManager {
    static let shared = ThumbnailCacheManager()
    
    private var cache = NSCache<NSURL, NSImage>()
    private var generateTasks: [URL: Task<NSImage?, Never>] = [:]
    
    init() {
        cache.countLimit = 100
        cache.totalCostLimit = 100 * 1024 * 1024 // 100MB
    }
    
    func clearTask(for url: URL) {
        generateTasks[url] = nil
    }
    
    func thumbnail(for url: URL, size: CGSize) async -> NSImage? {
        // Check cache first
        if let cached = cache.object(forKey: url as NSURL) {
            return cached
        }
        
        // Check if generation is in progress
        if let task = generateTasks[url] {
            return await task.value
        }
        
        // Start new generation
        let task = Task { () -> NSImage? in
            let thumbnail = await generateThumbnail(for: url, size: size)
            if let thumbnail = thumbnail {
                cache.setObject(thumbnail, forKey: url as NSURL, cost: Int(size.width * size.height * 4))
            }
            await self.clearTask(for: url)
            return thumbnail
        }
        
        generateTasks[url] = task
        return await task.value
    }
    
    private func generateThumbnail(for url: URL, size: CGSize) async -> NSImage? {
        let generator = QLThumbnailGenerator.shared
        let scale = await NSScreen.main?.backingScaleFactor ?? 2.0
        
        let request = QLThumbnailGenerator.Request(
            fileAt: url,
            size: size,
            scale: scale,
            representationTypes: .all
        )
        
        do {
            let representation = try await generator.generateBestRepresentation(for: request)
            return representation.nsImage
        } catch {
            // Fallback to system icon
            return await NSWorkspace.shared.icon(forFile: url.path)
        }
    }
    
    func clearCache() {
        cache.removeAllObjects()
        for task in generateTasks.values {
            task.cancel()
        }
        generateTasks.removeAll()
    }
}

// MARK: - Media Import Manager
@MainActor
class MediaImportManager: ObservableObject {
    @Published var isImporting = false
    @Published var importProgress: Double = 0
    @Published var currentFile = ""
    @Published var importErrors: [MediaImportError] = []
    
    struct MediaImportError: Identifiable {
        public let id = UUID()
        let url: URL
        let error: Error
    }
    
    func importFiles(_ urls: [URL]) async {
        isImporting = true
        importProgress = 0
        importErrors = []
        
        for (index, url) in urls.enumerated() {
            currentFile = url.lastPathComponent
            importProgress = Double(index) / Double(urls.count)
            
            do {
                _ = try await importFile(url)
            } catch {
                importErrors.append(MediaImportError(url: url, error: error))
            }
        }
        
        importProgress = 1.0
        isImporting = false
    }
    
    private func importFile(_ url: URL) async throws -> MediaPoolItem {
        guard url.startAccessingSecurityScopedResource() else {
            throw MediaImportErrorType.securityScopedResourceFailed
        }
        defer { url.stopAccessingSecurityScopedResource() }
        
        // Validate file
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MediaImportErrorType.fileNotFound
        }
        
        // Check if it's a supported media type
        let asset = AVAsset(url: url)
        let isPlayable = try? await asset.load(.isPlayable)
        let isReadable = try? await asset.load(.isReadable)
        guard isPlayable == true || isReadable == true else {
            throw MediaImportErrorType.unsupportedMediaType
        }
        
        // Create media item
        let mediaItem = MediaPoolItem(url: url)
        await mediaItem.loadMediaProperties()
        await mediaItem.generateThumbnail()
        
        return mediaItem
    }
    
    enum MediaImportErrorType: LocalizedError {
        case securityScopedResourceFailed
        case fileNotFound
        case unsupportedMediaType
        
        var errorDescription: String? {
            switch self {
            case .securityScopedResourceFailed:
                return "Failed to access file"
            case .fileNotFound:
                return "File not found"
            case .unsupportedMediaType:
                return "Unsupported media type"
            }
        }
    }
}

// Fix for missing type property
extension MediaPoolItem {
    var type: String {
        let ext = url.pathExtension.lowercased()
        switch ext {
        case "mp4", "mov", "avi", "mkv", "m4v":
            return "video"
        case "mp3", "wav", "aiff", "m4a":
            return "audio"
        case "jpg", "jpeg", "png", "gif", "bmp":
            return "image"
        default:
            return "other"
        }
    }
}
