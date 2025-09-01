import Foundation

// MARK: - Project Management

public struct Project: Codable {
    public let id: UUID
    public var name: String
    public var timeline: Timeline
    public var mediaPool: [MediaItem]
    public var createdAt: Date
    public var modifiedAt: Date
    public var metadata: ProjectMetadata
    
    public init(name: String) {
        self.id = UUID()
        self.name = name
        self.timeline = Timeline()
        self.mediaPool = []
        self.createdAt = Date()
        self.modifiedAt = Date()
        self.metadata = ProjectMetadata()
    }
}

public struct MediaItem: Identifiable, Codable {
    public let id: UUID
    public var url: URL
    public var name: String
    public var duration: Tick?
    public var frameRate: Double?
    public var hasVideo: Bool
    public var hasAudio: Bool
    public var thumbnailData: Data?
    
    public init(url: URL) {
        self.id = UUID()
        self.url = url
        self.name = url.lastPathComponent
        self.duration = nil
        self.frameRate = nil
        self.hasVideo = true
        self.hasAudio = true
        self.thumbnailData = nil
    }
}

public struct ProjectMetadata: Codable {
    public var frameRate: Double
    public var resolution: Resolution
    public var aspectRatio: AspectRatio
    public var colorSpace: String
    
    public init(
        frameRate: Double = 30.0,
        resolution: Resolution = .hd1080,
        aspectRatio: AspectRatio = .sixteen9,
        colorSpace: String = "Rec. 709"
    ) {
        self.frameRate = frameRate
        self.resolution = resolution
        self.aspectRatio = aspectRatio
        self.colorSpace = colorSpace
    }
    
    public enum Resolution: String, Codable, CaseIterable {
        case hd720 = "1280x720"
        case hd1080 = "1920x1080"
        case uhd4K = "3840x2160"
        
        public var width: Int {
            switch self {
            case .hd720: return 1280
            case .hd1080: return 1920
            case .uhd4K: return 3840
            }
        }
        
        public var height: Int {
            switch self {
            case .hd720: return 720
            case .hd1080: return 1080
            case .uhd4K: return 2160
            }
        }
    }
    
    public enum AspectRatio: String, Codable {
        case sixteen9 = "16:9"
        case four3 = "4:3"
        case square = "1:1"
        case vertical = "9:16"
    }
}

// MARK: - Project Manager

public class ProjectManager {
    private var currentProject: Project?
    private let documentsURL: URL
    private let eventStore: EventStore
    
    public init() throws {
        self.documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.eventStore = try EventStore()
    }
    
    public func createProject(name: String) -> Project {
        let project = Project(name: name)
        currentProject = project
        return project
    }
    
    public func openProject(at url: URL) throws -> Project {
        let data = try Data(contentsOf: url)
        let project = try JSONDecoder().decode(Project.self, from: data)
        currentProject = project
        return project
    }
    
    public func saveProject(_ project: Project, to url: URL? = nil) throws {
        let saveURL = url ?? documentsURL.appendingPathComponent("\(project.name).arproj")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(project)
        try data.write(to: saveURL)
    }
    
    public func importMedia(urls: [URL]) throws -> [MediaItem] {
        var items: [MediaItem] = []
        
        for url in urls {
            var item = MediaItem(url: url)
            
            // TODO: Extract media metadata
            // This would use AVFoundation in the UI layer
            
            items.append(item)
            currentProject?.mediaPool.append(item)
        }
        
        return items
    }
    
    public func executeCommand(_ command: Command) throws {
        guard var project = currentProject else {
            throw ProjectError.noProjectOpen
        }
        
        // Record event
        _ = try eventStore.append(command)
        
        // Apply command to timeline
        switch command {
        case .blade(let tick, let trackIndex):
            project.timeline.blade(at: tick, trackIndex: trackIndex)
            
        case .delete(let clipId):
            project.timeline.deleteClip(clipId)
            
        case .rippleDelete(let clipId):
            project.timeline.deleteClip(clipId, ripple: true)
            
        case .trim(let clipId, let edge, let tick):
            project.timeline.trim(clipId: clipId, edge: edge, to: tick)
            
        default:
            break
        }
        
        project.modifiedAt = Date()
        currentProject = project
    }
    
    public func getCurrentProject() -> Project? {
        return currentProject
    }
}

// MARK: - Project Errors

public enum ProjectError: Error {
    case noProjectOpen
    case saveF

ailed(String)
    case loadFailed(String)
    case mediaImportFailed(String)
}