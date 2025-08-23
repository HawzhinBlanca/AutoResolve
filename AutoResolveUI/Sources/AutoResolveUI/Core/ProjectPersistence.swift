// AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
// Robust Project Management and Persistence Layer

import Foundation
import SwiftUI
import UniformTypeIdentifiers
import CryptoKit

// MARK: - Enhanced Project Manager
class EnhancedProjectManager: ObservableObject {
    static let shared = EnhancedProjectManager()
    
    @Published var isLoading: Bool = false
    @Published var loadingProgress: Double = 0.0
    @Published var lastError: ProjectError?
    
    private let fileManager = FileManager.default
    private let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    private let projectsDirectory: URL
    private let backupsDirectory: URL
    private let templatesDirectory: URL
    private let autosaveDirectory: URL
    
    // File format constants
    private let projectExtension = "autoresolve"
    private let backupExtension = "autoresolve-backup"
    private let templateExtension = "autoresolve-template"
    
    private init() {
        projectsDirectory = documentsURL.appendingPathComponent("AutoResolve Projects")
        backupsDirectory = documentsURL.appendingPathComponent("AutoResolve Backups")
        templatesDirectory = documentsURL.appendingPathComponent("AutoResolve Templates")
        autosaveDirectory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("AutoResolve_Autosave")
        
        createDirectories()
        setupAutosaveCleanup()
    }
    
    private func createDirectories() {
        let directories = [projectsDirectory, backupsDirectory, templatesDirectory, autosaveDirectory]
        
        for directory in directories {
            try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }
    
    private func setupAutosaveCleanup() {
        // Clean up old autosave files on startup
        cleanupAutosaveFiles(olderThan: 24 * 60 * 60) // 24 hours
    }
    
    // MARK: - Project Loading
    func loadProject(from url: URL) async throws -> VideoProject {
        await MainActor.run { isLoading = true }
        defer { Task { @MainActor in isLoading = false } }
        
        do {
            let data = try Data(contentsOf: url)
            await MainActor.run { loadingProgress = 0.3 }
            
            let project = try JSONDecoder().decode(VideoProject.self, from: data)
            await MainActor.run { loadingProgress = 0.7 }
            
            // Validate project integrity
            try await validateProjectIntegrity(project, at: url)
            await MainActor.run { loadingProgress = 1.0 }
            
            return project
        } catch {
            await MainActor.run { lastError = ProjectError.loadFailed(error) }
            throw error
        }
    }
    
    private func validateProjectIntegrity(_ project: VideoProject, at url: URL) async throws {
        // Check for missing media files
        let missingFiles = await findMissingMediaFiles(in: project)
        if !missingFiles.isEmpty {
            throw ProjectError.missingMediaFiles(missingFiles)
        }
        
        // Validate timeline structure
        try validateTimelineStructure(project.timeline)
        
        // Check project settings compatibility
        try validateProjectSettings(project.settings)
    }
    
    private func findMissingMediaFiles(in project: VideoProject) async -> [URL] {
        var missingFiles: [URL] = []
        
        // Check media pool items
        for item in project.mediaPool.mediaItems {
            if !fileManager.fileExists(atPath: item.url.path) {
                missingFiles.append(item.url)
            }
        }
        
        // Check video clips
        for track in project.timeline.videoTracks {
            for clip in track.clips {
                if let sourceURL = clip.sourceURL,
                   !fileManager.fileExists(atPath: sourceURL.path) {
                    missingFiles.append(sourceURL)
                }
            }
        }
        
        // Check audio clips
        for track in project.timeline.audioTracks {
            for clip in track.clips {
                if let sourceURL = clip.sourceURL,
                   !fileManager.fileExists(atPath: sourceURL.path) {
                    missingFiles.append(sourceURL)
                }
            }
        }
        
        return missingFiles
    }
    
    private func validateTimelineStructure(_ timeline: Timeline) throws {
        // Validate track consistency
        for track in timeline.videoTracks {
            for clip in track.clips {
                if clip.duration <= 0 {
                    throw ProjectError.invalidTimelineStructure("Video clip has invalid duration")
                }
                if clip.timelineStartTime < 0 {
                    throw ProjectError.invalidTimelineStructure("Video clip has negative start time")
                }
            }
        }
        
        for track in timeline.audioTracks {
            for clip in track.clips {
                if clip.duration <= 0 {
                    throw ProjectError.invalidTimelineStructure("Audio clip has invalid duration")
                }
                if clip.timelineStartTime < 0 {
                    throw ProjectError.invalidTimelineStructure("Audio clip has negative start time")
                }
            }
        }
    }
    
    private func validateProjectSettings(_ settings: ProjectSettings) throws {
        // Validate resolution
        let supportedResolutions: [Resolution] = [.hd720, .hd1080, .uhd4K, .dci4K]
        if !supportedResolutions.contains(settings.resolution) {
            throw ProjectError.unsupportedSettings("Unsupported resolution: \(settings.resolution.rawValue)")
        }
        
        // Validate frame rate
        let supportedFrameRates: [FrameRate] = [.fps23_976, .fps24, .fps25, .fps29_97, .fps30, .fps60]
        if !supportedFrameRates.contains(settings.frameRate) {
            throw ProjectError.unsupportedSettings("Unsupported frame rate: \(settings.frameRate.rawValue)")
        }
    }
    
    // MARK: - Project Saving
    func saveProject(_ project: VideoProject, to url: URL? = nil) async throws -> URL {
        await MainActor.run { isLoading = true }
        defer { Task { @MainActor in isLoading = false } }
        
        let saveURL = url ?? projectsDirectory.appendingPathComponent("\(project.name).\(projectExtension)")
        
        do {
            // Create backup if file exists
            if fileManager.fileExists(atPath: saveURL.path) {
                try await createBackup(from: saveURL)
            }
            
            await MainActor.run { loadingProgress = 0.2 }
            
            // Prepare project for saving
            let projectToSave = project
            projectToSave.updateModifiedDate()
            
            await MainActor.run { loadingProgress = 0.4 }
            
            // Encode project
            let data = try JSONEncoder().encode(projectToSave)
            await MainActor.run { loadingProgress = 0.7 }
            
            // Write to temporary file first
            let tempURL = saveURL.appendingPathExtension("tmp")
            try data.write(to: tempURL)
            
            await MainActor.run { loadingProgress = 0.9 }
            
            // Atomic move to final location
            _ = try fileManager.replaceItem(at: saveURL, withItemAt: tempURL, backupItemName: nil, options: [], resultingItemURL: nil)
            
            await MainActor.run { loadingProgress = 1.0 }
            
            return saveURL
        } catch {
            await MainActor.run { lastError = ProjectError.saveFailed(error) }
            throw error
        }
    }
    
    private func createBackup(from url: URL) async throws {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let backupURL = backupsDirectory
            .appendingPathComponent(url.deletingPathExtension().lastPathComponent)
            .appendingPathExtension("\(timestamp).\(backupExtension)")
        
        try fileManager.copyItem(at: url, to: backupURL)
        
        // Limit number of backups per project
        try await limitBackups(for: url.deletingPathExtension().lastPathComponent, to: 10)
    }
    
    private func limitBackups(for projectName: String, to maxCount: Int) async throws {
        let backupURLs = try fileManager.contentsOfDirectory(at: backupsDirectory, includingPropertiesForKeys: [.creationDateKey])
            .filter { $0.lastPathComponent.hasPrefix(projectName) }
            .sorted { url1, url2 in
                let date1 = try? url1.resourceValues(forKeys: [.creationDateKey]).creationDate
                let date2 = try? url2.resourceValues(forKeys: [.creationDateKey]).creationDate
                return (date1 ?? Date.distantPast) > (date2 ?? Date.distantPast)
            }
        
        if backupURLs.count > maxCount {
            for url in backupURLs.dropFirst(maxCount) {
                try fileManager.removeItem(at: url)
            }
        }
    }
    
    // MARK: - Autosave
    func autosaveProject(_ project: VideoProject) async throws {
        let autosaveURL = autosaveDirectory.appendingPathComponent("\(project.id.uuidString).\(projectExtension)")
        
        let data = try JSONEncoder().encode(project)
        try data.write(to: autosaveURL)
        
        // Set file attributes for cleanup tracking
        let attributes = [FileAttributeKey.modificationDate: Date()]
        try fileManager.setAttributes(attributes, ofItemAtPath: autosaveURL.path)
    }
    
    func getAutosavedProjects() async -> [VideoProject] {
        guard let files = try? fileManager.contentsOfDirectory(at: autosaveDirectory, includingPropertiesForKeys: nil) else {
            return []
        }
        
        let projects = await withTaskGroup(of: VideoProject?.self) { group in
            for file in files.filter({ $0.pathExtension == projectExtension }) {
                group.addTask {
                    try? await self.loadProject(from: file)
                }
            }
            
            var results: [VideoProject] = []
            for await project in group {
                if let project = project {
                    results.append(project)
                }
            }
            return results
        }
        
        return projects
    }
    
    private func cleanupAutosaveFiles(olderThan seconds: TimeInterval) {
        guard let files = try? fileManager.contentsOfDirectory(at: autosaveDirectory, includingPropertiesForKeys: [.creationDateKey]) else {
            return
        }
        
        let cutoffDate = Date().addingTimeInterval(-seconds)
        
        for file in files {
            if let creationDate = try? file.resourceValues(forKeys: [.creationDateKey]).creationDate,
               creationDate < cutoffDate {
                try? fileManager.removeItem(at: file)
            }
        }
    }
    
    // MARK: - Templates
    func saveAsTemplate(_ project: VideoProject, name: String) async throws -> URL {
        let templateURL = templatesDirectory.appendingPathComponent("\(name).\(templateExtension)")
        
        // Create template version (remove media references)
        let template = project
        template.name = name
        template.mediaPool.mediaItems.removeAll()
        
        // Clear media references from clips
        for i in 0..<template.timeline.videoTracks.count {
            for j in 0..<template.timeline.videoTracks[i].clips.count {
                template.timeline.videoTracks[i].clips[j].sourceURL = nil
                template.timeline.videoTracks[i].clips[j].thumbnailCache.removeAll()
            }
        }
        
        for i in 0..<template.timeline.audioTracks.count {
            for j in 0..<template.timeline.audioTracks[i].clips.count {
                template.timeline.audioTracks[i].clips[j].sourceURL = nil
                template.timeline.audioTracks[i].clips[j].waveformCache.removeAll()
            }
        }
        
        let data = try JSONEncoder().encode(template)
        try data.write(to: templateURL)
        
        return templateURL
    }
    
    func loadTemplate(from url: URL) async throws -> VideoProject {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(VideoProject.self, from: data)
    }
    
    func getAvailableTemplates() async -> [TemplateInfo] {
        guard let files = try? fileManager.contentsOfDirectory(at: templatesDirectory, includingPropertiesForKeys: [.creationDateKey, .fileSizeKey]) else {
            return []
        }
        
        let templates = await withTaskGroup(of: TemplateInfo?.self) { group in
            for file in files.filter({ $0.pathExtension == templateExtension }) {
                group.addTask {
                    await self.createTemplateInfo(from: file)
                }
            }
            
            var results: [TemplateInfo] = []
            for await template in group {
                if let template = template {
                    results.append(template)
                }
            }
            return results
        }
        
        return templates.sorted { $0.name < $1.name }
    }
    
    private func createTemplateInfo(from url: URL) async -> TemplateInfo? {
        do {
            let project = try await loadTemplate(from: url)
            let resourceValues = try url.resourceValues(forKeys: [.creationDateKey, .fileSizeKey])
            
            return TemplateInfo(
                name: project.name,
                url: url,
                createdAt: resourceValues.creationDate ?? Date(),
                fileSize: Int64(resourceValues.fileSize ?? 0),
                videoTracks: project.timeline.videoTracks.count,
                audioTracks: project.timeline.audioTracks.count,
                resolution: project.settings.resolution,
                frameRate: project.settings.frameRate
            )
        } catch {
            return nil
        }
    }
    
    // MARK: - Import/Export
    func importProject(from url: URL, format: ImportFormat) async throws -> VideoProject {
        switch format {
        case .finalCutPro:
            return try await importFromFinalCutPro(url)
        case .premierePro:
            return try await importFromPremierePro(url)
        case .daVinciResolve:
            return try await importFromDaVinciResolve(url)
        case .autoResolve:
            return try await loadProject(from: url)
        }
    }
    
    private func importFromFinalCutPro(_ url: URL) async throws -> VideoProject {
        // Implementation for FCP XML import
        throw ProjectError.importNotSupported("Final Cut Pro import not yet implemented")
    }
    
    private func importFromPremierePro(_ url: URL) async throws -> VideoProject {
        // Implementation for Premiere Pro project import
        throw ProjectError.importNotSupported("Premiere Pro import not yet implemented")
    }
    
    private func importFromDaVinciResolve(_ url: URL) async throws -> VideoProject {
        // Implementation for DaVinci Resolve project import
        throw ProjectError.importNotSupported("DaVinci Resolve import not yet implemented")
    }
    
    func exportProject(_ project: VideoProject, to url: URL, format: ExportFormat) async throws {
        switch format {
        case .fcpxml:
            try await exportToFinalCutProXML(project, to: url)
        case .drp:
            _ = try await saveProject(project, to: url)
        case .aaf:
            try await exportToAvidMediaComposer(project, to: url)
        case .edl:
            throw ProjectError.exportNotSupported("EDL export not yet implemented")
        case .otio:
            throw ProjectError.exportNotSupported("OpenTimelineIO export not yet implemented")
        case .h264_mp4, .h265_mp4, .prores_mov, .dnxhd_mov:
            throw ProjectError.exportNotSupported("Video format export not supported in this context")
        case .gif, .image_sequence:
            throw ProjectError.exportNotSupported("Image format export not supported in this context")
        }
    }
    
    private func exportToFinalCutProXML(_ project: VideoProject, to url: URL) async throws {
        // Implementation for FCP XML export
        throw ProjectError.exportNotSupported("Final Cut Pro XML export not yet implemented")
    }
    
    private func exportToAvidMediaComposer(_ project: VideoProject, to url: URL) async throws {
        // Implementation for Avid Media Composer export
        throw ProjectError.exportNotSupported("Avid Media Composer export not yet implemented")
    }
    
    // MARK: - Project Organization
    func getRecentProjects(limit: Int = 10) -> [ProjectInfo] {
        let recentURLs = UserDefaults.standard.stringArray(forKey: "RecentProjects")?
            .compactMap { URL(string: $0) } ?? []
        
        return recentURLs.prefix(limit).compactMap { url in
            createProjectInfo(from: url)
        }
    }
    
    func addToRecentProjects(_ url: URL) {
        var recent = UserDefaults.standard.stringArray(forKey: "RecentProjects") ?? []
        recent.removeAll { $0 == url.absoluteString }
        recent.insert(url.absoluteString, at: 0)
        recent = Array(recent.prefix(20))
        
        UserDefaults.standard.set(recent, forKey: "RecentProjects")
    }
    
    private func createProjectInfo(from url: URL) -> ProjectInfo? {
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        
        do {
            let resourceValues = try url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            
            return ProjectInfo(
                name: url.deletingPathExtension().lastPathComponent,
                url: url,
                modifiedAt: resourceValues.contentModificationDate ?? Date(),
                fileSize: Int64(resourceValues.fileSize ?? 0)
            )
        } catch {
            return nil
        }
    }
}

// MARK: - Supporting Types
struct TemplateInfo: Identifiable {
    let id = UUID()
    let name: String
    let url: URL
    let createdAt: Date
    let fileSize: Int64
    let videoTracks: Int
    let audioTracks: Int
    let resolution: Resolution
    let frameRate: FrameRate
}

struct ProjectInfo: Identifiable {
    let id = UUID()
    let name: String
    let url: URL
    let modifiedAt: Date
    let fileSize: Int64
}

enum ImportFormat: String, CaseIterable {
    case autoResolve = "AutoResolve Project"
    case finalCutPro = "Final Cut Pro XML"
    case premierePro = "Premiere Pro Project"
    case daVinciResolve = "DaVinci Resolve Project"
    
    var fileExtensions: [String] {
        switch self {
        case .autoResolve: return ["autoresolve"]
        case .finalCutPro: return ["fcpxml"]
        case .premierePro: return ["prproj"]
        case .daVinciResolve: return ["drp"]
        }
    }
}

// Duplicate ExportFormat removed - using the one defined at line 460

enum ProjectError: LocalizedError {
    case loadFailed(Error)
    case saveFailed(Error)
    case missingMediaFiles([URL])
    case invalidTimelineStructure(String)
    case unsupportedSettings(String)
    case importNotSupported(String)
    case exportNotSupported(String)
    case corruptedProject(String)
    
    var errorDescription: String? {
        switch self {
        case .loadFailed(let error):
            return "Failed to load project: \(error.localizedDescription)"
        case .saveFailed(let error):
            return "Failed to save project: \(error.localizedDescription)"
        case .missingMediaFiles(let urls):
            return "Missing media files: \(urls.map { $0.lastPathComponent }.joined(separator: ", "))"
        case .invalidTimelineStructure(let message):
            return "Invalid timeline structure: \(message)"
        case .unsupportedSettings(let message):
            return "Unsupported project settings: \(message)"
        case .importNotSupported(let message):
            return "Import not supported: \(message)"
        case .exportNotSupported(let message):
            return "Export not supported: \(message)"
        case .corruptedProject(let message):
            return "Corrupted project: \(message)"
        }
    }
}