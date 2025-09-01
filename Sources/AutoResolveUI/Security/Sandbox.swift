import Foundation

public class Sandbox {
    public static let shared = Sandbox()
    
    private init() {}
    
    public func requestAccess(to url: URL) -> Bool {
        // Request sandbox access
        return url.startAccessingSecurityScopedResource()
    }
    
    public func releaseAccess(to url: URL) {
        url.stopAccessingSecurityScopedResource()
    }
    
    public func isAccessible(_ url: URL) -> Bool {
        return FileManager.default.isReadableFile(atPath: url.path)
    }
}
