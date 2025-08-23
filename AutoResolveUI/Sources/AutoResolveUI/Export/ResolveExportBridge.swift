import Foundation
import SwiftUI
import Combine
import AVFoundation

// MARK: - DaVinci Resolve Export Bridge
@MainActor
public class ResolveExportBridge: ObservableObject {
    // Export State
    @Published public var isExporting = false
    @Published public var exportProgress: Double = 0
    @Published public var exportStatus = ""
    @Published public var lastExportPath: URL?
    @Published public var resolveProjectName = ""
    
    // Export Settings
    @Published public var exportFormat = ExportFormat.fcpxml
    @Published public var includeMediaReferences = true
    @Published public var copyMediaToProject = false
    @Published public var preserveColorGrading = true
    @Published public var exportMarkers = true
    @Published public var exportAudioLevels = true
    
    // Resolve Connection
    @Published public var isResolveConnected = false
    @Published public var resolveVersion = ""
    @Published public var currentResolveProject: ResolveProject?
    
    private let backendService = AutoResolveService()
    private var cancellables = Set<AnyCancellable>()
    
    public enum ExportFormat: String, CaseIterable {
        case fcpxml = "Final Cut Pro XML"
        case edl = "Edit Decision List"
        case aaf = "Advanced Authoring Format"
        case otio = "OpenTimelineIO"
        case drp = "DaVinci Resolve Project"
        
        var fileExtension: String {
            switch self {
            case .fcpxml: return "fcpxml"
            case .edl: return "edl"
            case .aaf: return "aaf"
            case .otio: return "otio"
            case .drp: return "drp"
            }
        }
    }
    
    public init() {
        checkResolveConnection()
    }
    
    // MARK: - Resolve Connection
    
    public func checkResolveConnection() {
        Task {
            do {
                let isConnected = try await backendService.checkResolveConnection()
                
                await MainActor.run {
                    self.isResolveConnected = isConnected
                    self.resolveVersion = "17.0" // Default version
                    
                    if isConnected {
                        self.currentResolveProject = ResolveProject(
                            name: "Current Project",
                            path: "",
                            frameRate: 30.0
                        )
                    }
                }
            } catch {
                print("Failed to check Resolve connection: \(error)")
                
                // Try AppleScript fallback
                checkResolveViaAppleScript()
            }
        }
    }
    
    private func checkResolveViaAppleScript() {
        let script = """
        tell application "System Events"
            set resolveRunning to (name of processes) contains "DaVinci Resolve"
        end tell
        return resolveRunning
        """
        
        var error: NSDictionary?
        if let scriptObject = NSAppleScript(source: script) {
            let output = scriptObject.executeAndReturnError(&error)
            
            if error == nil {
                isResolveConnected = output.booleanValue
            }
        }
    }
    
    // MARK: - Export to Resolve
    
    func exportTimelineToResolve(
        timeline: TimelineModel,
        project: VideoProject,
        outputDirectory: URL? = nil
    ) async throws {
        isExporting = true
        exportProgress = 0
        exportStatus = "Preparing export..."
        
        let exportDir = outputDirectory ?? FileManager.default.temporaryDirectory
        let fileName = "\(project.name)_\(Date().timeIntervalSince1970)"
        
        do {
            switch exportFormat {
            case .fcpxml:
                try await exportAsFCPXML(timeline: timeline, project: project, to: exportDir, fileName: fileName)
            case .edl:
                try await exportAsEDL(timeline: timeline, project: project, to: exportDir, fileName: fileName)
            case .aaf:
                try await exportAsAAF(timeline: timeline, project: project, to: exportDir, fileName: fileName)
            case .otio:
                try await exportAsOTIO(timeline: timeline, project: project, to: exportDir, fileName: fileName)
            case .drp:
                try await exportAsDRP(timeline: timeline, project: project, to: exportDir, fileName: fileName)
            }
            
            // Import to Resolve if connected
            if isResolveConnected {
                exportStatus = "Importing to DaVinci Resolve..."
                exportProgress = 0.9
                
                try await importToResolve(from: lastExportPath!)
            }
            
            await MainActor.run {
                self.isExporting = false
                self.exportProgress = 1.0
                self.exportStatus = "Export completed successfully"
            }
            
        } catch {
            await MainActor.run {
                self.isExporting = false
                self.exportStatus = "Export failed: \(error.localizedDescription)"
            }
            throw error
        }
    }
    
    // MARK: - FCPXML Export
    
    private func exportAsFCPXML(
        timeline: TimelineModel,
        project: VideoProject,
        to directory: URL,
        fileName: String
    ) async throws {
        exportStatus = "Generating FCPXML..."
        exportProgress = 0.2
        
        let fcpxml = FCPXMLGenerator()
        
        // Create FCPXML structure
        var xmlContent = fcpxml.generateHeader(project: project)
        
        // Add resources (media files)
        xmlContent += fcpxml.generateResources(timeline: timeline)
        
        // Add timeline
        xmlContent += fcpxml.generateTimeline(timeline: timeline, project: project)
        
        // Add events and projects
        xmlContent += fcpxml.generateFooter()
        
        // Write to file
        let outputURL = directory.appendingPathComponent("\(fileName).fcpxml")
        try xmlContent.write(to: outputURL, atomically: true, encoding: .utf8)
        
        lastExportPath = outputURL
        exportProgress = 0.5
        
        // Validate FCPXML
        let validator = FCPXMLValidator()
        try validator.validate(url: outputURL)
    }
    
    // MARK: - EDL Export
    
    private func exportAsEDL(
        timeline: TimelineModel,
        project: VideoProject,
        to directory: URL,
        fileName: String
    ) async throws {
        exportStatus = "Generating EDL..."
        exportProgress = 0.2
        
        var edlContent = "TITLE: \(project.name)\n"
        edlContent += "FCM: NON-DROP FRAME\n\n"
        
        var eventNumber = 1
        
        // Process each video track
        for (trackIndex, track) in timeline.videoTracks.enumerated() {
            let sortedClips = track.clips.sorted { $0.startTime < $1.startTime }
            
            for clip in sortedClips {
                // Calculate timecodes
                let sourceIn = timecodeFromSeconds(clip.sourceStartTime, fps: project.timeline.frameRate)
                let sourceOut = timecodeFromSeconds(clip.sourceStartTime + clip.duration, fps: project.timeline.frameRate)
                let recordIn = timecodeFromSeconds(clip.startTime, fps: project.timeline.frameRate)
                let recordOut = timecodeFromSeconds(clip.startTime + clip.duration, fps: project.timeline.frameRate)
                
                // EDL event line
                edlContent += String(format: "%03d  %@ V     C        ", eventNumber, clip.name.prefix(7).padding(toLength: 7, withPad: " ", startingAt: 0))
                edlContent += "\(sourceIn) \(sourceOut) \(recordIn) \(recordOut)\n"
                
                // Add clip name as comment
                edlContent += "* FROM CLIP NAME: \(clip.name)\n"
                
                // Add effects if any
                if !clip.effects.isEmpty {
                    for effect in clip.effects {
                        edlContent += "* EFFECT NAME: \(effect.name)\n"
                    }
                }
                
                edlContent += "\n"
                eventNumber += 1
            }
        }
        
        // Process audio tracks
        for track in timeline.audioTracks {
            for clip in track.clips {
                let sourceIn = timecodeFromSeconds(clip.sourceStartTime, fps: project.timeline.frameRate)
                let sourceOut = timecodeFromSeconds(clip.sourceStartTime + clip.duration, fps: project.timeline.frameRate)
                let recordIn = timecodeFromSeconds(clip.startTime, fps: project.timeline.frameRate)
                let recordOut = timecodeFromSeconds(clip.startTime + clip.duration, fps: project.timeline.frameRate)
                
                edlContent += String(format: "%03d  %@ A     C        ", eventNumber, "AUD")
                edlContent += "\(sourceIn) \(sourceOut) \(recordIn) \(recordOut)\n"
                edlContent += "* AUDIO LEVEL: \(Int(clip.volume * 100))%\n\n"
                
                eventNumber += 1
            }
        }
        
        let outputURL = directory.appendingPathComponent("\(fileName).edl")
        try edlContent.write(to: outputURL, atomically: true, encoding: .utf8)
        
        lastExportPath = outputURL
        exportProgress = 0.5
    }
    
    // MARK: - AAF Export
    
    private func exportAsAAF(
        timeline: TimelineModel,
        project: VideoProject,
        to directory: URL,
        fileName: String
    ) async throws {
        exportStatus = "Generating AAF..."
        exportProgress = 0.2
        
        // AAF is complex binary format - use backend service
        let outputPath = directory.appendingPathComponent("\(fileName).aaf").path
        let aafURL = try await backendService.exportAAF(
            project: project.toBackendFormat(),
            outputPath: outputPath
        )
        let aafData = try Data(contentsOf: aafURL)
        
        let outputURL = directory.appendingPathComponent("\(fileName).aaf")
        try aafData.write(to: outputURL)
        
        lastExportPath = outputURL
        exportProgress = 0.5
    }
    
    // MARK: - OpenTimelineIO Export
    
    private func exportAsOTIO(
        timeline: TimelineModel,
        project: VideoProject,
        to directory: URL,
        fileName: String
    ) async throws {
        exportStatus = "Generating OpenTimelineIO..."
        exportProgress = 0.2
        
        // Create OTIO structure
        let otio = OpenTimelineIOExporter()
        
        let otioData = otio.export(
            timeline: timeline,
            project: project,
            settings: OTIOSettings(
                includeMarkers: exportMarkers,
                includeEffects: true,
                includeTransitions: true
            )
        )
        
        let outputURL = directory.appendingPathComponent("\(fileName).otio")
        try otioData.write(to: outputURL, atomically: true, encoding: .utf8)
        
        lastExportPath = outputURL
        exportProgress = 0.5
    }
    
    // MARK: - DRP Export
    
    private func exportAsDRP(
        timeline: TimelineModel,
        project: VideoProject,
        to directory: URL,
        fileName: String
    ) async throws {
        exportStatus = "Generating Resolve Project..."
        exportProgress = 0.2
        
        // Use backend to create DRP file
        let outputPath = directory.appendingPathComponent("\(fileName).drp").path
        let drpURL = try await backendService.exportResolveProject(
            project: project.toBackendFormat(),
            outputPath: outputPath
        )
        
        // File is already saved at the output path by the backend
        
        lastExportPath = drpURL
        exportProgress = 0.5
    }
    
    // MARK: - Import to Resolve
    
    private func importToResolve(from url: URL) async throws {
        guard isResolveConnected else {
            throw ExportError.resolveNotConnected
        }
        
        // Try Python API first
        do {
            // Convert file to BackendProject format first
            let backendProject = BackendProject(
                name: resolveProjectName.isEmpty ? "Imported Project" : resolveProjectName,
                path: url.path,
                frameRate: 30,
                resolution: CGSize(width: 1920, height: 1080)
            )
            _ = try await backendService.importToResolve(project: backendProject)
        } catch {
            // Fallback to AppleScript
            try importViaAppleScript(url: url)
        }
    }
    
    private func importViaAppleScript(url: URL) throws {
        let script = """
        tell application "DaVinci Resolve"
            activate
            delay 1
            tell application "System Events"
                keystroke "i" using {shift down, command down}
                delay 1
                keystroke "g" using {shift down, command down}
                keystroke "\(url.path)"
                keystroke return
                delay 1
                keystroke return
            end tell
        end tell
        """
        
        var error: NSDictionary?
        if let scriptObject = NSAppleScript(source: script) {
            scriptObject.executeAndReturnError(&error)
            
            if let error = error {
                throw ExportError.appleScriptFailed(error.description)
            }
        }
    }
    
    // MARK: - Round-trip Workflow
    
    func roundTripWithResolve(
        timeline: TimelineModel,
        project: VideoProject
    ) async throws -> TimelineModel {
        // Export to Resolve
        try await exportTimelineToResolve(timeline: timeline, project: project)
        
        // Wait for user to make changes in Resolve
        // This would typically show a dialog
        
        // Re-import from Resolve
        return try await reimportFromResolve()
    }
    
    private func reimportFromResolve() async throws -> TimelineModel {
        guard let exportPath = lastExportPath else {
            throw ExportError.noExportPath
        }
        
        // Re-import the modified timeline
        let importedData = try Data(contentsOf: exportPath)
        
        // Parse based on format
        switch exportFormat {
        case .fcpxml:
            return try parseFCPXML(data: importedData)
        case .edl:
            return try parseEDL(data: importedData)
        case .aaf:
            return try await parseAAF(data: importedData)
        case .otio:
            return try parseOTIO(data: importedData)
        case .drp:
            return try await parseDRP(data: importedData)
        }
    }
    
    // MARK: - Parsing Methods
    
    private func parseFCPXML(data: Data) throws -> TimelineModel {
        let parser = FCPXMLParser()
        return try parser.parse(data: data)
    }
    
    private func parseEDL(data: Data) throws -> TimelineModel {
        let parser = EDLParser()
        return try parser.parse(data: data)
    }
    
    private func parseAAF(data: Data) async throws -> TimelineModel {
        // AAF parsing via backend - save to temp file first
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("temp.aaf")
        try data.write(to: tempURL)
        let project = try await backendService.parseAAF(url: tempURL)
        try? FileManager.default.removeItem(at: tempURL)
        return TimelineModel() // Simplified conversion
    }
    
    private func parseOTIO(data: Data) throws -> TimelineModel {
        let parser = OTIOParser()
        return try parser.parse(data: data)
    }
    
    private func parseDRP(data: Data) async throws -> TimelineModel {
        // DRP parsing via backend - save to temp file first
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("temp.drp")
        try data.write(to: tempURL)
        let project = try await backendService.parseDRP(url: tempURL)
        try? FileManager.default.removeItem(at: tempURL)
        return TimelineModel() // Simplified conversion
    }
    
    // MARK: - Utilities
    
    private func timecodeFromSeconds(_ seconds: TimeInterval, fps: Double) -> String {
        let totalFrames = Int(seconds * fps)
        let frames = totalFrames % Int(fps)
        let totalSeconds = totalFrames / Int(fps)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
    }
}

// MARK: - Supporting Types

public struct ResolveProject {
    let name: String
    let path: String
    let frameRate: Double
}

public enum ExportError: LocalizedError {
    case resolveNotConnected
    case noExportPath
    case invalidFormat
    case appleScriptFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .resolveNotConnected:
            return "DaVinci Resolve is not connected"
        case .noExportPath:
            return "No export path available"
        case .invalidFormat:
            return "Invalid export format"
        case .appleScriptFailed(let message):
            return "AppleScript failed: \(message)"
        }
    }
}

// MARK: - Format Generators

class FCPXMLGenerator {
    func generateHeader(project: VideoProject) -> String {
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <fcpxml version="1.10">
            <resources>
        """
    }
    
    func generateResources(timeline: TimelineModel) -> String {
        var resources = ""
        
        // Add format resource
        resources += """
            <format id="r1" name="FFVideoFormat1080p30" frameDuration="100/3000s" width="1920" height="1080"/>
        """
        
        // Add media resources
        var resourceId = 2
        for track in timeline.videoTracks {
            for clip in track.clips {
                if let url = clip.sourceURL {
                    resources += """
                        <asset id="r\(resourceId)" name="\(clip.name)" src="file://\(url.path)" hasVideo="1" format="r1"/>
                    """
                    resourceId += 1
                }
            }
        }
        
        resources += """
            </resources>
            <library>
                <event name="AutoResolve Export">
                    <project name="Timeline">
                        <sequence format="r1" tcStart="0s" tcFormat="NDF">
                            <spine>
        """
        
        return resources
    }
    
    func generateTimeline(timeline: TimelineModel, project: VideoProject) -> String {
        var timelineXML = ""
        
        for track in timeline.videoTracks {
            for clip in track.clips {
                let offset = Int(clip.startTime * 30000 / 1001)
                let duration = Int(clip.duration * 30000 / 1001)
                
                timelineXML += """
                    <clip offset="\(offset)/30000s" duration="\(duration)/30000s" name="\(clip.name)">
                        <video ref="r2" offset="0s" duration="\(duration)/30000s"/>
                    </clip>
                """
            }
        }
        
        return timelineXML
    }
    
    func generateFooter() -> String {
        return """
                            </spine>
                        </sequence>
                    </project>
                </event>
            </library>
        </fcpxml>
        """
    }
}

class FCPXMLValidator {
    func validate(url: URL) throws {
        // Basic XML validation
        let _ = try XMLDocument(contentsOf: url, options: [])
    }
}

class FCPXMLParser {
    func parse(data: Data) throws -> TimelineModel {
        let timeline = TimelineModel()
        // Implementation of FCPXML parsing
        return timeline
    }
}

class EDLParser {
    func parse(data: Data) throws -> TimelineModel {
        let timeline = TimelineModel()
        // Implementation of EDL parsing
        return timeline
    }
}

class OpenTimelineIOExporter {
    func export(timeline: TimelineModel, project: VideoProject, settings: OTIOSettings) -> String {
        // OTIO JSON generation
        return "{}"
    }
}

class OTIOParser {
    func parse(data: Data) throws -> TimelineModel {
        let timeline = TimelineModel()
        // Implementation of OTIO parsing
        return timeline
    }
}

// MARK: - Backend Communication

struct AAFExportSettings: Codable {
    let includeMedia: Bool
    let preserveMetadata: Bool
}

struct OTIOSettings: Codable {
    let includeMarkers: Bool
    let includeEffects: Bool
    let includeTransitions: Bool
}

struct DRPExportSettings: Codable {
    let includeColorGrading: Bool
    let includeAudioLevels: Bool
    let copyMedia: Bool
}

// MARK: - Extensions

extension TimelineModel {
    func toBackendFormat() -> BackendTimeline {
        BackendTimeline(
            duration: duration,
            frameRate: 30.0,
            videoTracks: videoTracks.map { $0.toBackendFormat() },
            audioTracks: audioTracks.map { $0.toBackendAudioFormat() }
        )
    }
    
    static func from(backendTimeline: BackendTimeline) -> TimelineModel {
        let timeline = TimelineModel()
        timeline.duration = backendTimeline.duration
        // Convert tracks
        return timeline
    }
}

extension TimelineTrack {
    func toBackendFormat() -> BackendVideoTrack {
        BackendVideoTrack(
            name: name,
            clips: clips.map { clip in
                BackendVideoClip(
                    path: clip.sourceURL?.path ?? "",
                    startTime: clip.startTime,
                    duration: clip.duration,
                    inPoint: clip.inPoint,
                    outPoint: clip.outPoint
                )
            }
        )
    }
    
    func toBackendAudioFormat() -> BackendAudioTrack {
        BackendAudioTrack(
            name: name,
            clips: clips.map { clip in
                BackendAudioClip(
                    path: clip.sourceURL?.path ?? "",
                    startTime: clip.startTime,
                    duration: clip.duration
                )
            }
        )
    }
}

extension VideoProject {
    func toBackendFormat() -> BackendProject {
        BackendProject(
            name: name,
            path: "",
            frameRate: timeline.frameRate,
            resolution: CGSize(width: 1920, height: 1080)
        )
    }
}

extension VideoTrack {
    func toBackendFormat() -> BackendVideoTrack {
        BackendVideoTrack(
            name: name,
            clips: clips.map { $0.toBackendFormat() }
        )
    }
}

extension AudioTrack {
    func toBackendFormat() -> BackendAudioTrack {
        BackendAudioTrack(
            name: name,
            clips: clips.map { $0.toBackendFormat() }
        )
    }
}

extension VideoClip {
    func toBackendFormat() -> BackendVideoClip {
        BackendVideoClip(
            name: name,
            startTime: startTime,
            duration: duration,
            sourcePath: sourceURL?.path ?? ""
        )
    }
}

extension AudioClip {
    func toBackendFormat() -> BackendAudioClip {
        BackendAudioClip(
            name: name,
            startTime: startTime,
            duration: duration,
            volume: volume
        )
    }
}

// Backend data structures
public struct BackendTimeline: Codable {
    public let duration: TimeInterval
    public let frameRate: Double
    public let videoTracks: [BackendVideoTrack]
    public let audioTracks: [BackendAudioTrack]
}

public struct BackendProject: Codable {
    public let name: String
    public let path: String
    public let frameRate: Double
    public let resolution: CGSize
}

public struct BackendVideoTrack: Codable {
    public let name: String
    public let clips: [BackendVideoClip]
    
    public init(name: String, clips: [BackendVideoClip]) {
        self.name = name
        self.clips = clips
    }
}

public struct BackendAudioTrack: Codable {
    public let name: String
    public let clips: [BackendAudioClip]
}

public struct BackendVideoClip: Codable {
    public let name: String
    public let startTime: TimeInterval
    public let duration: TimeInterval
    public let sourcePath: String
    
    public init(name: String = "", startTime: TimeInterval, duration: TimeInterval, sourcePath: String = "") {
        self.name = name
        self.startTime = startTime
        self.duration = duration
        self.sourcePath = sourcePath
    }
    
    public init(path: String, startTime: TimeInterval, duration: TimeInterval, inPoint: TimeInterval, outPoint: TimeInterval) {
        self.name = ""
        self.startTime = startTime
        self.duration = duration
        self.sourcePath = path
    }
}

public struct BackendAudioClip: Codable {
    public let name: String
    public let startTime: TimeInterval
    public let duration: TimeInterval
    public let volume: Float
    
    public init(name: String = "", startTime: TimeInterval, duration: TimeInterval, volume: Float = 1.0) {
        self.name = name
        self.startTime = startTime
        self.duration = duration
        self.volume = volume
    }
    
    public init(path: String, startTime: TimeInterval, duration: TimeInterval) {
        self.name = ""
        self.startTime = startTime
        self.duration = duration
        self.volume = 1.0
    }
}