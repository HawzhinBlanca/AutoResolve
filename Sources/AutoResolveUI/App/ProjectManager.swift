import Foundation
import AutoResolveCore

public class ProjectManagerUI {
    private let projectManager = ProjectManager()
    
    public init() {}
    
    public func createProject(name: String) -> Project {
        return projectManager.createProject(name: name)
    }
    
    public func openProject(at url: URL) throws -> Project {
        return try projectManager.openProject(at: url)
    }
    
    public func saveProject(_ project: Project, to url: URL) throws {
        try projectManager.saveProject(project, to: url)
    }
}
