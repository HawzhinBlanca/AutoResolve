import Foundation

public enum AutoResolveError: Error, LocalizedError {
    case backendUnavailable
    case invalidResponse
    case authenticationFailed
    case networkError(String)
    case processingError(String)
    case exportError(String)
    case fileNotFound
    case invalidData
    
    public var errorDescription: String? {
        switch self {
        case .backendUnavailable:
            return "Backend service is unavailable"
        case .invalidResponse:
            return "Invalid response from backend"
        case .authenticationFailed:
            return "Authentication failed"
        case .networkError(let message):
            return "Network error: \(message)"
        case .processingError(let message):
            return "Processing error: \(message)"
        case .exportError(let message):
            return "Export error: \(message)"
        case .fileNotFound:
            return "File not found"
        case .invalidData:
            return "Invalid data"
        }
    }
}