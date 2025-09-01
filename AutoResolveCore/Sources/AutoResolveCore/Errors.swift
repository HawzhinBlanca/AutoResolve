import Foundation

// MARK: - AutoResolve Error Codes (Blueprint Section 16)

public enum AutoResolveError: String, Error, LocalizedError {
    // Import errors
    case importUnsupported = "AR-E001"
    case importDRM = "AR-E001"
    
    // Network errors  
    case backendUnreachable = "AR-E010"
    case loopbackOnly = "AR-E010"
    
    // ASR errors
    case asrUnavailable = "AR-E020"
    case midWordGateDisabled = "AR-E020"
    
    // Export errors
    case exportEDLFailed = "AR-E030"
    case exportFCPXMLFailed = "AR-E030"
    
    // Notarization errors
    case notarizationRequired = "AR-E040"
    
    // Timeline errors
    case invalidTick = "AR-E050"
    case clipNotFound = "AR-E051"
    case trackNotFound = "AR-E052"
    
    // Command errors
    case commandFailed = "AR-E060"
    case undoStackEmpty = "AR-E061"
    case redoStackEmpty = "AR-E062"
    
    // Project errors
    case projectCorrupted = "AR-E070"
    case projectVersionMismatch = "AR-E071"
    
    // Media errors
    case mediaOffline = "AR-E080"
    case codecUnsupported = "AR-E081"
    
    // Performance errors
    case memoryExceeded = "AR-E090"
    case frameDropped = "AR-E091"
    
    public var errorDescription: String? {
        switch self {
        case .importUnsupported, .importDRM:
            return "Import failure: File format unsupported or protected by DRM"
            
        case .backendUnreachable, .loopbackOnly:
            return "Backend unreachable: Only localhost connections allowed"
            
        case .asrUnavailable, .midWordGateDisabled:
            return "ASR unavailable: MidWord gate has been disabled"
            
        case .exportEDLFailed:
            return "Export failure: Could not generate EDL file"
            
        case .exportFCPXMLFailed:
            return "Export failure: Could not generate FCPXML file"
            
        case .notarizationRequired:
            return "Notarization required: App must be notarized for distribution"
            
        case .invalidTick:
            return "Invalid timeline position"
            
        case .clipNotFound:
            return "Clip not found in timeline"
            
        case .trackNotFound:
            return "Track not found in timeline"
            
        case .commandFailed:
            return "Command execution failed"
            
        case .undoStackEmpty:
            return "Nothing to undo"
            
        case .redoStackEmpty:
            return "Nothing to redo"
            
        case .projectCorrupted:
            return "Project file is corrupted"
            
        case .projectVersionMismatch:
            return "Project was created with an incompatible version"
            
        case .mediaOffline:
            return "Media file is offline or moved"
            
        case .codecUnsupported:
            return "Media codec is not supported"
            
        case .memoryExceeded:
            return "Memory limit exceeded"
            
        case .frameDropped:
            return "Performance warning: Frames dropped"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .importUnsupported, .importDRM:
            return "Try converting the file to a supported format (H.264/HEVC)"
            
        case .backendUnreachable:
            return "Ensure the backend is running on localhost:8000"
            
        case .asrUnavailable:
            return "Install CoreML Whisper model or use backend ASR"
            
        case .exportEDLFailed, .exportFCPXMLFailed:
            return "Check file permissions and available disk space"
            
        case .notarizationRequired:
            return "Run 'make notarize' to notarize the application"
            
        case .mediaOffline:
            return "Relink media or restore to original location"
            
        case .codecUnsupported:
            return "Transcode media to H.264 or ProRes"
            
        case .memoryExceeded:
            return "Close other applications or reduce timeline complexity"
            
        case .frameDropped:
            return "Reduce playback quality or enable proxy mode"
            
        default:
            return nil
        }
    }
}

// MARK: - Result Extensions

public extension Result where Failure == AutoResolveError {
    var isSuccess: Bool {
        switch self {
        case .success:
            return true
        case .failure:
            return false
        }
    }
    
    func getError() -> AutoResolveError? {
        switch self {
        case .success:
            return nil
        case .failure(let error):
            return error
        }
    }
}

// MARK: - Error Logger

public struct ErrorLogger {
    public static func log(_ error: AutoResolveError, file: String = #file, line: Int = #line) {
        let filename = URL(fileURLWithPath: file).lastPathComponent
        print("[\(error.rawValue)] \(error.localizedDescription) at \(filename):\(line)")
        
        if let recovery = error.recoverySuggestion {
            print("  → \(recovery)")
        }
    }
}