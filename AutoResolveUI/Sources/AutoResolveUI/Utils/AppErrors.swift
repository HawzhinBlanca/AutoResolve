// AUTORESOLVE V3.0 - COMPREHENSIVE ERROR HANDLING
// Unified error types with localized descriptions

import Foundation

/// Main error type for the entire application
public enum AutoResolveError: LocalizedError, Equatable {
    case authentication(AppAuthError)
    case network(AppNetworkError)
    case media(AppMediaError)
    case timeline(AppTimelineError)
    case storage(AppStorageError)
    case validation(AppValidationError)
    case backend(AppBackendError)
    
    public var errorDescription: String? {
        switch self {
        case .authentication(let error):
            return error.localizedDescription
        case .network(let error):
            return error.localizedDescription
        case .media(let error):
            return error.localizedDescription
        case .timeline(let error):
            return error.localizedDescription
        case .storage(let error):
            return error.localizedDescription
        case .validation(let error):
            return error.localizedDescription
        case .backend(let error):
            return error.localizedDescription
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .authentication(let error):
            return error.recoverySuggestion
        case .network(let error):
            return error.recoverySuggestion
        case .media(let error):
            return error.recoverySuggestion
        case .timeline(let error):
            return error.recoverySuggestion
        case .storage(let error):
            return error.recoverySuggestion
        case .validation(let error):
            return error.recoverySuggestion
        case .backend(let error):
            return error.recoverySuggestion
        }
    }
    
    public var isRetryable: Bool {
        switch self {
        case .network(let error):
            return error.isRetryable
        case .backend(let error):
            return error.isRetryable
        case .storage(let error):
            return error.isRetryable
        default:
            return false
        }
    }
}

// MARK: - Authentication Errors

public enum AppAuthError: LocalizedError, Equatable {
    case invalidCredentials
    case tokenExpired
    case tokenInvalid
    case biometricFailed
    case biometricNotAvailable
    case accountLocked
    case twoFactorRequired
    case networkError(String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidCredentials:
            return "Invalid username or password"
        case .tokenExpired:
            return "Your session has expired"
        case .tokenInvalid:
            return "Authentication token is invalid"
        case .biometricFailed:
            return "Biometric authentication failed"
        case .biometricNotAvailable:
            return "Biometric authentication is not available"
        case .accountLocked:
            return "Your account has been locked"
        case .twoFactorRequired:
            return "Two-factor authentication is required"
        case .networkError(let message):
            return "Authentication failed: \(message)"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .invalidCredentials:
            return "Please check your username and password"
        case .tokenExpired:
            return "Please log in again"
        case .tokenInvalid:
            return "Please log in again"
        case .biometricFailed:
            return "Try again or use your password"
        case .biometricNotAvailable:
            return "Use your password to log in"
        case .accountLocked:
            return "Contact support to unlock your account"
        case .twoFactorRequired:
            return "Enter your two-factor authentication code"
        case .networkError:
            return "Check your internet connection and try again"
        }
    }
}

// MARK: - Network Errors

public enum AppNetworkError: LocalizedError, Equatable {
    case noConnection
    case timeout
    case serverError(Int, String?)
    case invalidResponse
    case decodingError(String)
    case requestCancelled
    case rateLimited(retryAfter: TimeInterval?)
    
    public var errorDescription: String? {
        switch self {
        case .noConnection:
            return "No internet connection"
        case .timeout:
            return "Request timed out"
        case .serverError(let code, let message):
            return "Server error (\(code)): \(message ?? "Unknown error")"
        case .invalidResponse:
            return "Invalid server response"
        case .decodingError(let details):
            return "Failed to decode response: \(details)"
        case .requestCancelled:
            return "Request was cancelled"
        case .rateLimited(let retryAfter):
            if let retryAfter = retryAfter {
                return "Rate limited. Try again in \(Int(retryAfter)) seconds"
            }
            return "Rate limited. Please try again later"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .noConnection:
            return "Check your internet connection"
        case .timeout:
            return "Try again with a better connection"
        case .serverError(let code, _):
            if code >= 500 {
                return "The server is experiencing issues. Please try again later"
            }
            return "Check your request and try again"
        case .invalidResponse:
            return "Try updating the app to the latest version"
        case .decodingError:
            return "Contact support if this persists"
        case .requestCancelled:
            return "Try your request again"
        case .rateLimited:
            return "Wait a moment before trying again"
        }
    }
    
    public var isRetryable: Bool {
        switch self {
        case .timeout, .rateLimited:
            return true
        case .serverError(let code, _):
            return code >= 500 || code == 429
        case .noConnection, .requestCancelled:
            return true
        default:
            return false
        }
    }
}

// MARK: - Media Errors

public enum AppMediaError: LocalizedError, Equatable {
    case fileNotFound(URL)
    case unsupportedFormat(String)
    case corruptedFile(URL)
    case accessDenied(URL)
    case fileTooLarge(Int64, maxSize: Int64)
    case metadataExtractionFailed(String)
    case thumbnailGenerationFailed
    case codecNotSupported(String)
    
    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "File not found: \(url.lastPathComponent)"
        case .unsupportedFormat(let format):
            return "Unsupported format: \(format)"
        case .corruptedFile(let url):
            return "File is corrupted: \(url.lastPathComponent)"
        case .accessDenied(let url):
            return "Access denied: \(url.lastPathComponent)"
        case .fileTooLarge(let size, let maxSize):
            let formatter = ByteCountFormatter()
            return "File too large: \(formatter.string(fromByteCount: size)) (max: \(formatter.string(fromByteCount: maxSize)))"
        case .metadataExtractionFailed(let reason):
            return "Failed to extract metadata: \(reason)"
        case .thumbnailGenerationFailed:
            return "Failed to generate thumbnail"
        case .codecNotSupported(let codec):
            return "Codec not supported: \(codec)"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .fileNotFound:
            return "Check if the file has been moved or deleted"
        case .unsupportedFormat:
            return "Convert the file to a supported format"
        case .corruptedFile:
            return "Try re-downloading or re-exporting the file"
        case .accessDenied:
            return "Check file permissions and security settings"
        case .fileTooLarge:
            return "Use a smaller file or compress it"
        case .metadataExtractionFailed:
            return "The file may be corrupted or in an unsupported format"
        case .thumbnailGenerationFailed:
            return "Try again or use a different frame"
        case .codecNotSupported:
            return "Install required codecs or convert the file"
        }
    }
}

// MARK: - Timeline Errors

public enum AppTimelineError: LocalizedError, Equatable {
    case invalidTimeRange(start: TimeInterval, end: TimeInterval)
    case clipOverlap(trackId: UUID)
    case trackNotFound(UUID)
    case clipNotFound(UUID)
    case insufficientSpace(required: TimeInterval, available: TimeInterval)
    case renderFailed(String)
    case exportFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidTimeRange(let start, let end):
            return "Invalid time range: \(start) - \(end)"
        case .clipOverlap(let trackId):
            return "Clips overlap on track \(trackId)"
        case .trackNotFound(let id):
            return "Track not found: \(id)"
        case .clipNotFound(let id):
            return "Clip not found: \(id)"
        case .insufficientSpace(let required, let available):
            return "Insufficient space: need \(required)s, have \(available)s"
        case .renderFailed(let reason):
            return "Render failed: \(reason)"
        case .exportFailed(let reason):
            return "Export failed: \(reason)"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .invalidTimeRange:
            return "Ensure the end time is after the start time"
        case .clipOverlap:
            return "Move or trim overlapping clips"
        case .trackNotFound, .clipNotFound:
            return "The item may have been deleted"
        case .insufficientSpace:
            return "Make room by moving or deleting clips"
        case .renderFailed:
            return "Check your render settings and try again"
        case .exportFailed:
            return "Check disk space and export settings"
        }
    }
}

// MARK: - Storage Errors

public enum AppStorageError: LocalizedError, Equatable {
    case insufficientSpace(required: Int64, available: Int64)
    case writePermissionDenied(URL)
    case readPermissionDenied(URL)
    case fileSystemError(String)
    case cacheCorrupted
    case quotaExceeded
    
    public var errorDescription: String? {
        switch self {
        case .insufficientSpace(let required, let available):
            let formatter = ByteCountFormatter()
            return "Insufficient storage: need \(formatter.string(fromByteCount: required)), have \(formatter.string(fromByteCount: available))"
        case .writePermissionDenied(let url):
            return "Cannot write to: \(url.path)"
        case .readPermissionDenied(let url):
            return "Cannot read from: \(url.path)"
        case .fileSystemError(let error):
            return "File system error: \(error)"
        case .cacheCorrupted:
            return "Cache is corrupted"
        case .quotaExceeded:
            return "Storage quota exceeded"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .insufficientSpace:
            return "Free up disk space and try again"
        case .writePermissionDenied, .readPermissionDenied:
            return "Check folder permissions in System Preferences"
        case .fileSystemError:
            return "Restart the app and try again"
        case .cacheCorrupted:
            return "Clear cache in settings"
        case .quotaExceeded:
            return "Delete old projects or upgrade your plan"
        }
    }
    
    public var isRetryable: Bool {
        switch self {
        case .fileSystemError, .cacheCorrupted:
            return true
        default:
            return false
        }
    }
}

// MARK: - Validation Errors

public enum AppValidationError: LocalizedError, Equatable {
    case missingRequiredField(String)
    case invalidFormat(field: String, expected: String)
    case valueTooLarge(field: String, max: String)
    case valueTooSmall(field: String, min: String)
    case invalidPath(String)
    case invalidURL(String)
    
    public var errorDescription: String? {
        switch self {
        case .missingRequiredField(let field):
            return "\(field) is required"
        case .invalidFormat(let field, let expected):
            return "\(field) must be in \(expected) format"
        case .valueTooLarge(let field, let max):
            return "\(field) must be less than \(max)"
        case .valueTooSmall(let field, let min):
            return "\(field) must be greater than \(min)"
        case .invalidPath(let path):
            return "Invalid path: \(path)"
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .missingRequiredField:
            return "Fill in all required fields"
        case .invalidFormat(_, let expected):
            return "Use the format: \(expected)"
        case .valueTooLarge, .valueTooSmall:
            return "Enter a valid value within the allowed range"
        case .invalidPath:
            return "Check the file path and try again"
        case .invalidURL:
            return "Enter a valid URL"
        }
    }
}

// MARK: - Backend Errors

public enum AppBackendError: LocalizedError, Equatable {
    case pipelineFailed(String)
    case taskNotFound(String)
    case invalidConfiguration(String)
    case serviceUnavailable
    case maintenanceMode
    
    public var errorDescription: String? {
        switch self {
        case .pipelineFailed(let reason):
            return "Pipeline failed: \(reason)"
        case .taskNotFound(let id):
            return "Task not found: \(id)"
        case .invalidConfiguration(let error):
            return "Invalid configuration: \(error)"
        case .serviceUnavailable:
            return "Backend service is unavailable"
        case .maintenanceMode:
            return "Service is under maintenance"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .pipelineFailed:
            return "Check your settings and try again"
        case .taskNotFound:
            return "The task may have expired or been cancelled"
        case .invalidConfiguration:
            return "Review your configuration settings"
        case .serviceUnavailable:
            return "Try again in a few moments"
        case .maintenanceMode:
            return "Please wait for maintenance to complete"
        }
    }
    
    public var isRetryable: Bool {
        switch self {
        case .serviceUnavailable, .maintenanceMode:
            return true
        default:
            return false
        }
    }
}

// MARK: - Error Recovery

public struct ErrorRecovery {
    
    /// Attempt to recover from an error
    public static func recover(from error: AutoResolveError, retry: @escaping () async throws -> Void) async throws {
        guard error.isRetryable else {
            throw error
        }
        
        // Exponential backoff for retryable errors
        let delays: [TimeInterval] = [1, 2, 4, 8, 16]
        var lastError: Error = error
        
        for (index, delay) in delays.enumerated() {
            do {
                if index > 0 {
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
                try await retry()
                return
            } catch {
                lastError = error
                
                // Don't retry if it's not retryable
                if let autoResolveError = error as? AutoResolveError, !autoResolveError.isRetryable {
                    throw error
                }
            }
        }
        
        throw lastError
    }
}