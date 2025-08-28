// AUTORESOLVE V3.0 - COMMON VALIDATORS
// Centralized validation logic to ensure consistency

import Foundation
import AVFoundation
import CoreMedia

/// Centralized validation utilities
public struct Validators {
    
    // MARK: - File Validation
    
    /// Validate file URL exists and is accessible
    public static func isValidFile(at url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }
    
    /// Validate file is readable
    public static func isReadable(at url: URL) -> Bool {
        FileManager.default.isReadableFile(atPath: url.path)
    }
    
    /// Validate file is writable
    public static func isWritable(at url: URL) -> Bool {
        FileManager.default.isWritableFile(atPath: url.path)
    }
    
    /// Validate file extension
    public static func hasValidExtension(_ url: URL, extensions: [String]) -> Bool {
        let fileExtension = url.pathExtension.lowercased()
        return extensions.contains(fileExtension)
    }
    
    /// Check if file is a video
    public static func isVideoFile(_ url: URL) -> Bool {
        let videoExtensions = ["mp4", "mov", "avi", "mkv", "m4v", "webm", "flv", "wmv", "mpg", "mpeg", "3gp"]
        return hasValidExtension(url, extensions: videoExtensions)
    }
    
    /// Check if file is audio
    public static func isAudioFile(_ url: URL) -> Bool {
        let audioExtensions = ["mp3", "wav", "aac", "m4a", "flac", "ogg", "wma", "aiff", "alac"]
        return hasValidExtension(url, extensions: audioExtensions)
    }
    
    /// Check if file is an image
    public static func isImageFile(_ url: URL) -> Bool {
        let imageExtensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "heic", "webp", "svg"]
        return hasValidExtension(url, extensions: imageExtensions)
    }
    
    /// Validate file size is within limits
    public static func isFileSizeValid(_ url: URL, maxSize: Int64) -> Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let fileSize = attributes[.size] as? Int64 else {
            return false
        }
        return fileSize <= maxSize
    }
    
    // MARK: - Path Validation
    
    /// Validate path doesn't contain traversal attempts
    public static func isSafePath(_ path: String) -> Bool {
        // Check for path traversal attempts
        if path.contains("../") || path.contains("..\\") {
            return false
        }
        
        // Check for absolute paths to system directories
        let dangerousPaths = ["/etc", "/usr", "/bin", "/sbin", "/var", "/System", "C:\\Windows", "C:\\Program Files"]
        for dangerous in dangerousPaths {
            if path.hasPrefix(dangerous) {
                return false
            }
        }
        
        return true
    }
    
    /// Validate path is within allowed directory
    public static func isPathWithinDirectory(_ path: String, directory: String) -> Bool {
        let normalizedPath = URL(fileURLWithPath: path).standardized.path
        let normalizedDir = URL(fileURLWithPath: directory).standardized.path
        return normalizedPath.hasPrefix(normalizedDir)
    }
    
    // MARK: - String Validation
    
    /// Validate email format
    public static func isValidEmail(_ email: String) -> Bool {
        let pattern = "^[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$"
        let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive)
        let range = NSRange(location: 0, length: email.utf16.count)
        return regex?.firstMatch(in: email, options: [], range: range) != nil
    }
    
    /// Validate URL format
    public static func isValidURL(_ urlString: String) -> Bool {
        guard let url = URL(string: urlString) else { return false }
        return url.scheme != nil && url.host != nil
    }
    
    /// Validate username format
    public static func isValidUsername(_ username: String) -> Bool {
        // Username: 3-20 characters, alphanumeric and underscore only
        let pattern = "^[a-zA-Z0-9_]{3,20}$"
        let regex = try? NSRegularExpression(pattern: pattern)
        let range = NSRange(location: 0, length: username.utf16.count)
        return regex?.firstMatch(in: username, options: [], range: range) != nil
    }
    
    /// Validate password strength
    public static func isStrongPassword(_ password: String) -> Bool {
        // At least 8 characters, one uppercase, one lowercase, one digit, one special
        guard password.count >= 8 else { return false }
        
        let patterns = [
            ".*[A-Z].*",      // Uppercase
            ".*[a-z].*",      // Lowercase
            ".*[0-9].*",      // Digit
            ".*[^A-Za-z0-9].*" // Special character
        ]
        
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern),
                  regex.firstMatch(in: password, options: [], range: NSRange(location: 0, length: password.utf16.count)) != nil else {
                return false
            }
        }
        
        return true
    }
    
    // MARK: - Time Validation
    
    /// Validate CMTime is valid and numeric
    public static func isValidTime(_ time: CMTime) -> Bool {
        time.isValid && time.isNumeric && time.seconds.isFinite
    }
    
    /// Validate time range
    public static func isValidTimeRange(start: CMTime, end: CMTime) -> Bool {
        isValidTime(start) && isValidTime(end) && CMTimeCompare(start, end) < 0
    }
    
    /// Validate duration is positive
    public static func isValidDuration(_ duration: TimeInterval) -> Bool {
        duration.isFinite && duration > 0
    }
    
    // MARK: - Number Validation
    
    /// Validate number is within range
    public static func isInRange<T: Comparable>(_ value: T, min: T, max: T) -> Bool {
        value >= min && value <= max
    }
    
    /// Validate positive number
    public static func isPositive<T: Numeric & Comparable>(_ value: T) -> Bool {
        value > 0
    }
    
    /// Validate non-negative number
    public static func isNonNegative<T: Numeric & Comparable>(_ value: T) -> Bool {
        value >= 0
    }
    
    // MARK: - Media Validation
    
    /// Validate frame rate is reasonable
    public static func isValidFrameRate(_ fps: Double) -> Bool {
        isInRange(fps, min: 1.0, max: 240.0) && fps.isFinite
    }
    
    /// Validate resolution
    public static func isValidResolution(width: Int, height: Int) -> Bool {
        width > 0 && height > 0 && width <= 8192 && height <= 8192
    }
    
    /// Validate aspect ratio
    public static func isValidAspectRatio(_ ratio: Double) -> Bool {
        ratio.isFinite && ratio > 0 && isInRange(ratio, min: 0.1, max: 10.0)
    }
    
    /// Validate audio sample rate
    public static func isValidSampleRate(_ rate: Double) -> Bool {
        let validRates = [8000.0, 16000.0, 22050.0, 32000.0, 44100.0, 48000.0, 88200.0, 96000.0, 192000.0]
        return validRates.contains(rate)
    }
    
    /// Validate bitrate
    public static func isValidBitrate(_ bitrate: Int) -> Bool {
        bitrate > 0 && bitrate <= 100_000_000 // Max 100 Mbps
    }
    
    // MARK: - JSON Validation
    
    /// Validate JSON string
    public static func isValidJSON(_ jsonString: String) -> Bool {
        guard let data = jsonString.data(using: .utf8) else { return false }
        
        do {
            _ = try JSONSerialization.jsonObject(with: data, options: [])
            return true
        } catch {
            return false
        }
    }
    
    /// Validate JSON data
    public static func isValidJSON(_ data: Data) -> Bool {
        do {
            _ = try JSONSerialization.jsonObject(with: data, options: [])
            return true
        } catch {
            return false
        }
    }
    
    // MARK: - UUID Validation
    
    /// Validate UUID string
    public static func isValidUUID(_ string: String) -> Bool {
        UUID(uuidString: string) != nil
    }
    
    // MARK: - Network Validation
    
    /// Validate port number
    public static func isValidPort(_ port: Int) -> Bool {
        isInRange(port, min: 1, max: 65535)
    }
    
    /// Validate IP address (IPv4)
    public static func isValidIPv4(_ address: String) -> Bool {
        let parts = address.split(separator: ".")
        guard parts.count == 4 else { return false }
        
        for part in parts {
            guard let num = Int(part),
                  isInRange(num, min: 0, max: 255) else {
                return false
            }
        }
        
        return true
    }
    
    // MARK: - Export Format Validation
    
    /// Validate export format
    public static func isValidExportFormat(_ format: String) -> Bool {
        let validFormats = ["mp4", "mov", "prores", "h264", "h265", "fcpxml", "edl", "xml"]
        return validFormats.contains(format.lowercased())
    }
    
    /// Validate codec
    public static func isValidCodec(_ codec: String) -> Bool {
        let validCodecs = ["h264", "h265", "hevc", "prores", "prores422", "prores4444", "dnxhd", "dnxhr"]
        return validCodecs.contains(codec.lowercased())
    }
}

// MARK: - Validation Result

/// Result type for validation with detailed error information
public struct AppValidationResult {
    public let isValid: Bool
    public let errors: [AppValidationError]
    
    public init(isValid: Bool, errors: [AppValidationError] = []) {
        self.isValid = isValid
        self.errors = errors
    }
    
    public static var valid: AppValidationResult {
        AppValidationResult(isValid: true)
    }
    
    public static func invalid(_ error: AppValidationError) -> AppValidationResult {
        AppValidationResult(isValid: false, errors: [error])
    }
    
    public static func invalid(_ errors: [AppValidationError]) -> AppValidationResult {
        AppValidationResult(isValid: false, errors: errors)
    }
}

// MARK: - Batch Validators

public extension Validators {
    
    /// Validate media file comprehensively
    static func validateMediaFile(_ url: URL) -> AppValidationResult {
        var errors: [AppValidationError] = []
        
        if !isValidFile(at: url) {
            errors.append(.missingRequiredField("File does not exist"))
        }
        
        if !isReadable(at: url) {
            errors.append(.invalidPath(url.path))
        }
        
        if !isVideoFile(url) && !isAudioFile(url) {
            errors.append(.invalidFormat(field: "file", expected: "video or audio"))
        }
        
        // Check file size (max 10GB)
        if !isFileSizeValid(url, maxSize: 10 * 1024 * 1024 * 1024) {
            errors.append(.valueTooLarge(field: "fileSize", max: "10GB"))
        }
        
        return errors.isEmpty ? .valid : .invalid(errors)
    }
    
    /// Validate export settings
    static func validateExportSettings(
        format: String,
        codec: String?,
        resolution: (width: Int, height: Int)?,
        frameRate: Double?,
        bitrate: Int?
    ) -> AppValidationResult {
        var errors: [AppValidationError] = []
        
        if !isValidExportFormat(format) {
            errors.append(.invalidFormat(field: "format", expected: "mp4, mov, prores, etc."))
        }
        
        if let codec = codec, !isValidCodec(codec) {
            errors.append(.invalidFormat(field: "codec", expected: "h264, h265, prores, etc."))
        }
        
        if let resolution = resolution,
           !isValidResolution(width: resolution.width, height: resolution.height) {
            errors.append(.invalidFormat(field: "resolution", expected: "valid dimensions"))
        }
        
        if let fps = frameRate, !isValidFrameRate(fps) {
            errors.append(.valueTooLarge(field: "frameRate", max: "240"))
        }
        
        if let bitrate = bitrate, !isValidBitrate(bitrate) {
            errors.append(.valueTooLarge(field: "bitrate", max: "100000000"))
        }
        
        return errors.isEmpty ? .valid : .invalid(errors)
    }
    
    /// Validate user credentials
    static func validateCredentials(username: String, password: String) -> AppValidationResult {
        var errors: [AppValidationError] = []
        
        if username.isEmpty {
            errors.append(.missingRequiredField("username"))
        } else if !isValidUsername(username) {
            errors.append(.invalidFormat(field: "username", expected: "3-20 alphanumeric characters"))
        }
        
        if password.isEmpty {
            errors.append(.missingRequiredField("password"))
        } else if !isStrongPassword(password) {
            errors.append(.invalidFormat(field: "password", expected: "8+ chars with upper, lower, digit, special"))
        }
        
        return errors.isEmpty ? .valid : .invalid(errors)
    }
}