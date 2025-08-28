// AUTORESOLVE V3.0 - IMPORT DIALOG MANAGER
import Combine
// Handles batch import, validation, and progress tracking

import SwiftUI
import AppKit
import AVFoundation
import CryptoKit
import UniformTypeIdentifiers

// MARK: - Import Dialog Manager
@MainActor
class ImportDialogManager: ObservableObject {
    @Published var selectedFiles: [URL] = []
    @Published var isImporting = false
    @Published var progress: Double = 0
    @Published var currentOperation = ""
    @Published var importErrors: [ImportError] = []
    @Published var settings = ImportSettings()
    @Published var duplicates: [DuplicateFile] = []
    @Published var validationResults: [URL: ValidationResult] = [:]
    
    private var thumbnailCache: [URL: NSImage] = [:]
    private var fileHashes: [URL: String] = [:]
    private let supportedExtensions = ["mov", "mp4", "m4v", "avi", "mkv", "mxf", "r3d", "braw", 
                                       "wav", "aiff", "mp3", "m4a", "jpg", "png", "tiff", "psd"]
    
    enum SortOrder: String, CaseIterable {
        case name = "Name"
        case date = "Date"
        case size = "Size"
        case type = "Type"
        case duration = "Duration"
    }
    
    struct ImportError: Identifiable {
        public let id = UUID()
        let file: URL
        let error: Error
        let timestamp: Date
    }
    
    struct DuplicateFile: Identifiable {
        public let id = UUID()
        let original: URL
        let duplicate: URL
        let hash: String
    }
    
    struct MediaMetadata {
        let duration: TimeInterval
        let frameRate: Double?
        let resolution: CGSize?
        
        init(duration: TimeInterval = 0, frameRate: Double? = nil, resolution: CGSize? = nil) {
            self.duration = duration
            self.frameRate = frameRate
            self.resolution = resolution
        }
    }
    
    struct ValidationResult {
        let isValid: Bool
        let warnings: [String]
        let metadata: MediaMetadata
    }
    
    struct TechnicalMediaMetadata {
        let duration: TimeInterval?
        let frameRate: Double?
        let resolution: CGSize?
        let codec: String?
        let audioChannels: Int?
        let hasAlpha: Bool
        
        var asMediaMetadata: MediaMetadata {
            return MediaMetadata(
                duration: duration ?? 0,
                frameRate: frameRate,
                resolution: resolution
            )
        }
    }
    
    // MARK: - File Management
    func addFile(_ url: URL) {
        guard !selectedFiles.contains(url) else { return }
        
        Task {
            // Validate file
            let result = await validateFile(url)
            validationResults[url] = result
            
            if result.isValid {
                selectedFiles.append(url)
                
                // Check for duplicates
                if settings.detectDuplicates {
                    await checkForDuplicate(url)
                }
                
                // Generate thumbnail
                await generateThumbnail(for: url)
            } else {
                importErrors.append(ImportError(
                    file: url,
                    error: ImportErrorType.validationFailed,
                    timestamp: Date()
                ))
            }
        }
    }
    
    func addFiles(_ urls: [URL]) {
        Task {
            await withTaskGroup(of: Void.self) { group in
                for url in urls {
                    group.addTask { [weak self] in
                        await self?.addFile(url)
                    }
                }
            }
        }
    }
    
    func removeFile(_ url: URL) {
        selectedFiles.removeAll { $0 == url }
        validationResults.removeValue(forKey: url)
        thumbnailCache.removeValue(forKey: url)
    }
    
    func selectAll() {
        // Select all valid files in current directory
    }
    
    func deselectAll() {
        selectedFiles.removeAll()
    }
    
    func updateSelection(_ files: Set<URL>) {
        selectedFiles = Array(files)
    }
    
    // MARK: - Sorting
    func sort(by order: SortOrder) {
        switch order {
        case .name:
            selectedFiles.sort { $0.lastPathComponent < $1.lastPathComponent }
        case .date:
            selectedFiles.sort { 
                (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? Date()) ?? Date() <
                (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? Date()) ?? Date()
            }
        case .size:
            selectedFiles.sort {
                (try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) ?? 0 <
                (try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) ?? 0
            }
        case .type:
            selectedFiles.sort { $0.pathExtension < $1.pathExtension }
        case .duration:
            selectedFiles.sort {
                (validationResults[$0]?.metadata.duration ?? 0) <
                (validationResults[$1]?.metadata.duration ?? 0)
            }
        }
    }
    
    // MARK: - Directory Loading
    func loadDirectory(at url: URL) {
        Task {
            do {
                let contents = try FileManager.default.contentsOfDirectory(
                    at: url,
                    includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                    options: [.skipsHiddenFiles]
                )
                
                let mediaFiles = contents.filter { url in
                    supportedExtensions.contains(url.pathExtension.lowercased())
                }
                
                // Clear and reload
                selectedFiles.removeAll()
                addFiles(mediaFiles)
            } catch {
                print("Failed to load directory: \(error)")
            }
        }
    }
    
    // MARK: - Validation
    private func validateFile(_ url: URL) async -> ValidationResult {
        var warnings: [String] = []
        var metadata = TechnicalMediaMetadata(
            duration: nil,
            frameRate: nil,
            resolution: nil,
            codec: nil,
            audioChannels: nil,
            hasAlpha: false
        )
        
        // Check file is safe and readable
        guard url.isFileURL,
              url.path.contains("..") == false,
              FileManager.default.isReadableFile(atPath: url.path) else {
            return ValidationResult(isValid: false, warnings: ["File is not readable"], metadata: metadata.asMediaMetadata)
        }
        
        // Check file size
        if let fileSize = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            if fileSize > 10_737_418_240 { // 10GB
                warnings.append("File size exceeds 10GB")
            }
        }
        
        // Validate media file
        if isMediaFile(url) {
            let asset = AVAsset(url: url)
            
            do {
                // Parallel load of basic properties
                async let playableAsync = asset.load(.isPlayable)
                async let durationAsync = asset.load(.duration)
                let (isPlayable, duration) = try await (playableAsync, durationAsync)
                if !isPlayable {
                    warnings.append("File may not be playable")
                }
                
                metadata = TechnicalMediaMetadata(
                    duration: CMTimeGetSeconds(duration),
                    frameRate: await getFrameRate(from: asset),
                    resolution: await getResolution(from: asset),
                    codec: await getCodec(from: asset),
                    audioChannels: await getAudioChannels(from: asset),
                    hasAlpha: await hasAlphaChannel(asset)
                )
                
                // Check for common issues
                if let fps = metadata.frameRate {
                    if fps < 23.976 || fps > 60 {
                        warnings.append("Unusual frame rate: \(String(format: "%.2f", fps)) fps")
                    }
                }
                
                if let resolution = metadata.resolution {
                    if resolution.width > 8192 || resolution.height > 4320 {
                        warnings.append("Resolution exceeds 8K")
                    }
                }
                
            } catch {
                warnings.append("Failed to read media properties: \(error.localizedDescription)")
            }
        }
        
        return ValidationResult(
            isValid: warnings.filter { $0.contains("Failed") || $0.contains("not") }.isEmpty,
            warnings: warnings,
            metadata: metadata.asMediaMetadata
        )
    }
    
    private func isMediaFile(_ url: URL) -> Bool {
        let mediaExtensions = ["mov", "mp4", "m4v", "avi", "mkv", "mxf", "r3d", "braw", "wav", "aiff", "mp3", "m4a"]
        return mediaExtensions.contains(url.pathExtension.lowercased())
    }
    
    private func getFrameRate(from asset: AVAsset) async -> Double? {
        guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return nil }
        return try? await Double(track.load(.nominalFrameRate))
    }
    
    private func getResolution(from asset: AVAsset) async -> CGSize? {
        guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return nil }
        return try? await track.load(.naturalSize)
    }
    
    private func getCodec(from asset: AVAsset) async -> String? {
        guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return nil }
        guard let descriptions = try? await track.load(.formatDescriptions) else { return nil }
        
        if let desc = descriptions.first {
            let fourCC = CMFormatDescriptionGetMediaSubType(desc)
            return fourCCString(fourCC)
        }
        return nil
    }
    
    private func getAudioChannels(from asset: AVAsset) async -> Int? {
        guard let track = try? await asset.loadTracks(withMediaType: .audio).first else { return nil }
        guard let descriptions = try? await track.load(.formatDescriptions) else { return nil }
        
        if let desc = descriptions.first {
            if let basicDescription = CMAudioFormatDescriptionGetStreamBasicDescription(desc) {
                return Int(basicDescription.pointee.mChannelsPerFrame)
            }
        }
        return nil
    }
    
    private func hasAlphaChannel(_ asset: AVAsset) async -> Bool {
        // Check if video has alpha channel
        guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return false }
        guard let descriptions = try? await track.load(.formatDescriptions) else { return false }
        
        if let desc = descriptions.first {
            let mediaSubType = CMFormatDescriptionGetMediaSubType(desc)
            // ProRes 4444 and other formats with alpha
            return mediaSubType == kCVPixelFormatType_4444AYpCbCr16 ||
                   mediaSubType == kCVPixelFormatType_4444AYpCbCr8
        }
        return false
    }
    
    private func fourCCString(_ fourCC: FourCharCode) -> String {
        let chars = [
            Character(UnicodeScalar((fourCC >> 24) & 0xFF)!),
            Character(UnicodeScalar((fourCC >> 16) & 0xFF)!),
            Character(UnicodeScalar((fourCC >> 8) & 0xFF)!),
            Character(UnicodeScalar(fourCC & 0xFF)!)
        ]
        return String(chars)
    }
    
    // MARK: - Duplicate Detection
    private func checkForDuplicate(_ url: URL) async {
        guard settings.detectDuplicates else { return }
        
        let hash = await calculateFileHash(url)
        fileHashes[url] = hash
        
        // Check against existing files
        for (existingURL, existingHash) in fileHashes where existingURL != url {
            if existingHash == hash {
                duplicates.append(DuplicateFile(
                    original: existingURL,
                    duplicate: url,
                    hash: hash
                ))
                break
            }
        }
    }
    
    private func calculateFileHash(_ url: URL) async -> String {
        // Stream-based hashing for large files
        return await Task.detached(priority: .background) {
            do {
                let bufferSize = 1024 * 1024  // 1MB chunks
                let handle = try FileHandle(forReadingFrom: url)
                defer { try? handle.close() }
                
                var hasher = SHA256()
                var totalBytes: Int64 = 0
                let fileSize = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
                
                while autoreleasepool(invoking: {
                    guard let data = try? handle.read(upToCount: bufferSize), !data.isEmpty else {
                        return false
                    }
                    hasher.update(data: data)
                    totalBytes += Int64(data.count)
                    
                    // Report progress for large files
                    if fileSize > 10_000_000 { // 10MB+
                        let progress = Double(totalBytes) / Double(fileSize)
                        Task { @MainActor in
                            self.progress = progress * 0.5 // Hashing is 50% of import
                        }
                    }
                    return true
                }) {}
                
                let digest = hasher.finalize()
                return digest.compactMap { String(format: "%02x", $0) }.joined()
            } catch {
                return UUID().uuidString // Fallback to unique ID if hashing fails
            }
        }.value
    }
    
    // MARK: - Thumbnail Generation
    private func generateThumbnail(for url: URL) async {
        if thumbnailCache[url] != nil { return }
        
        if isMediaFile(url) {
            let asset = AVAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 280, height: 160)
            
            let time = CMTime(seconds: 1, preferredTimescale: 600)
            
            do {
                let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
                thumbnailCache[url] = NSImage(cgImage: cgImage, size: CGSize(width: 280, height: 160))
            } catch {
                // Use file icon as fallback
                thumbnailCache[url] = NSWorkspace.shared.icon(forFile: url.path)
            }
        } else {
            // Use QuickLook for images
            thumbnailCache[url] = NSImage(contentsOf: url) ?? NSWorkspace.shared.icon(forFile: url.path)
        }
    }
    
    func getPreview(for url: URL) -> NSImage? {
        return thumbnailCache[url]
    }
    
    // MARK: - Import Execution
    func performImport(to mediaPool: MediaPoolViewModel) async {
        isImporting = true
        progress = 0
        importErrors.removeAll()
        
        let totalFiles = selectedFiles.count
        
        for (index, file) in selectedFiles.enumerated() {
            currentOperation = "Importing \(file.lastPathComponent)"
            progress = Double(index) / Double(totalFiles)
            
            do {
                // Copy to project folder if needed
                var importURL = file
                if settings.copyToProject {
                    importURL = try await copyToProjectFolder(file)
                }
                
                // Create optimized media if needed
                if settings.createOptimized {
                    currentOperation = "Creating optimized media for \(file.lastPathComponent)"
                    try await createOptimizedMedia(for: importURL)
                }
                
                // Create proxies if needed
                if settings.createProxies {
                    currentOperation = "Creating proxy for \(file.lastPathComponent)"
                    try await createProxy(for: importURL)
                }
                
                // Add to media pool
                let mediaItem = MediaPoolItem(url: importURL)
                await mediaItem.loadMediaProperties()
                mediaPool.addItem(mediaItem)
                
            } catch {
                importErrors.append(ImportError(
                    file: file,
                    error: error,
                    timestamp: Date()
                ))
            }
        }
        
        progress = 1.0
        currentOperation = "Import complete"
        isImporting = false
    }
    
    private func copyToProjectFolder(_ url: URL) async throws -> URL {
        // Implementation for copying to project folder
        return url
    }
    
    private func createOptimizedMedia(for url: URL) async throws {
        // Implementation for creating optimized media
    }
    
    private func createProxy(for url: URL) async throws {
        // Implementation for creating proxy media
    }
    
    // MARK: - Computed Properties
    var totalSize: Int64 {
        selectedFiles.compactMap { url in
            try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize
        }.reduce(Int64(0)) { $0 + Int64($1 ?? 0) }
    }
    
    var totalSizeFormatted: String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: totalSize)
    }
}

// MARK: - Import Settings
struct ImportSettings: Codable, Sendable {
    var copyToProject = false
    var createOptimized = false
    var createProxies = false
    var proxyResolution = ProxyResolution.half
    var proxyCodec = ProxyCodec.proRes422Proxy
    
    var autoDetectFrameRate = true
    var removePulldown = false
    var deinterlace = false
    var alphaHandling = AlphaHandling.none
    
    var autoSyncAudio = false
    var normalizeAudio = false
    var sampleRate = 0 // 0 = project rate
    
    var importMetadata = true
    var importMarkers = true
    var importLUTs = false
    
    var detectDuplicates = true
    var skipExisting = true
}

enum ProxyResolution: String, CaseIterable, Codable {
    case quarter = "Quarter"
    case half = "Half"
    case full = "Full"
}

enum ProxyCodec: String, CaseIterable, Codable {
    case proRes422Proxy = "ProRes 422 Proxy"
    case proRes422LT = "ProRes 422 LT"
    case h264 = "H.264"
    case h265 = "H.265"
}

enum AlphaHandling: String, CaseIterable, Codable {
    case none = "None"
    case straight = "Straight"
    case premultiplied = "Premultiplied"
}

enum ImportErrorType: LocalizedError {
    case validationFailed
    case copyFailed
    case proxyGenerationFailed
    case optimizedMediaFailed
    case duplicateFile
    
    var errorDescription: String? {
        switch self {
        case .validationFailed:
            return "File validation failed"
        case .copyFailed:
            return "Failed to copy file to project"
        case .proxyGenerationFailed:
            return "Failed to generate proxy"
        case .optimizedMediaFailed:
            return "Failed to create optimized media"
        case .duplicateFile:
            return "Duplicate file detected"
        }
    }
}
