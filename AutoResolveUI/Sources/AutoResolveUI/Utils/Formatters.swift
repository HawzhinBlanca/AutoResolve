// AUTORESOLVE V3.0 - COMMON FORMATTERS
// Centralized formatting utilities to eliminate code duplication

import Foundation
import AVFoundation
import CoreMedia

/// Centralized formatting utilities
public struct Formatters {
    
    // MARK: - Time & Duration
    
    /// Format duration in seconds to readable string (e.g., "1:23:45")
    public static func formatDuration(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite && seconds >= 0 else { return "0:00" }
        
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, secs)
        } else {
            return String(format: "%d:%02d", minutes, secs)
        }
    }
    
    /// Format CMTime to readable string with frame precision
    public static func formatTime(_ time: CMTime, fps: Double = 30.0) -> String {
        guard time.isValid && time.isNumeric else { return "00:00:00:00" }
        
        let totalSeconds = CMTimeGetSeconds(time)
        guard totalSeconds.isFinite && totalSeconds >= 0 else { return "00:00:00:00" }
        
        let hours = Int(totalSeconds) / 3600
        let minutes = (Int(totalSeconds) % 3600) / 60
        let seconds = Int(totalSeconds) % 60
        let frames = Int((totalSeconds.truncatingRemainder(dividingBy: 1)) * fps)
        
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, seconds, frames)
    }
    
    /// Format timecode with optional frame display
    public static func formatTimecode(_ seconds: TimeInterval, showFrames: Bool = false, fps: Double = 30.0) -> String {
        guard seconds.isFinite && seconds >= 0 else {
            return showFrames ? "00:00:00:00" : "00:00:00"
        }
        
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        
        if showFrames {
            let frames = Int((seconds.truncatingRemainder(dividingBy: 1)) * fps)
            return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
        } else {
            return String(format: "%02d:%02d:%02d", hours, minutes, secs)
        }
    }
    
    /// Convert seconds to user-friendly string (e.g., "2 hours 15 minutes")
    public static func humanReadableDuration(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite && seconds >= 0 else { return "0 seconds" }
        
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        
        var components: [String] = []
        
        if hours > 0 {
            components.append(hours == 1 ? "1 hour" : "\(hours) hours")
        }
        if minutes > 0 {
            components.append(minutes == 1 ? "1 minute" : "\(minutes) minutes")
        }
        if secs > 0 || components.isEmpty {
            components.append(secs == 1 ? "1 second" : "\(secs) seconds")
        }
        
        return components.joined(separator: " ")
    }
    
    // MARK: - File Size
    
    /// Format bytes to human-readable string
    public static func formatFileSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        formatter.allowedUnits = [.useAll]
        return formatter.string(fromByteCount: bytes)
    }
    
    /// Format bytes with specific precision
    public static func formatBytes(_ bytes: Int64, precision: Int = 2) -> String {
        guard bytes > 0 else { return "0 B" }
        
        let units = ["B", "KB", "MB", "GB", "TB"]
        let k: Double = 1024.0
        var magnitude = Double(bytes)
        var unitIndex = 0
        
        while magnitude >= k && unitIndex < units.count - 1 {
            magnitude /= k
            unitIndex += 1
        }
        
        if unitIndex == 0 {
            return "\(bytes) B"
        } else {
            return String(format: "%.\(precision)f %@", magnitude, units[unitIndex])
        }
    }
    
    // MARK: - Numbers
    
    /// Format percentage with specified decimal places
    public static func formatPercentage(_ value: Double, decimals: Int = 1) -> String {
        guard value.isFinite else { return "0%" }
        
        let percentage = value * 100
        
        if decimals == 0 {
            return "\(Int(percentage.rounded()))%"
        } else {
            return String(format: "%.\(decimals)f%%", percentage)
        }
    }
    
    /// Format large numbers with abbreviations (e.g., "1.2M", "450K")
    public static func formatLargeNumber(_ number: Int) -> String {
        guard number != 0 else { return "0" }
        
        let absNumber = abs(number)
        let sign = number < 0 ? "-" : ""
        
        switch absNumber {
        case 0..<1000:
            return "\(sign)\(absNumber)"
        case 1000..<1_000_000:
            let value = Double(absNumber) / 1000.0
            return String(format: "%@%.1fK", sign, value)
        case 1_000_000..<1_000_000_000:
            let value = Double(absNumber) / 1_000_000.0
            return String(format: "%@%.1fM", sign, value)
        default:
            let value = Double(absNumber) / 1_000_000_000.0
            return String(format: "%@%.1fB", sign, value)
        }
    }
    
    /// Format frame rate to common representation
    public static func formatFrameRate(_ fps: Double) -> String {
        guard fps.isFinite && fps > 0 else { return "Unknown" }
        
        // Common frame rates
        let commonRates: [(Double, String)] = [
            (23.976, "23.98"),
            (24.0, "24"),
            (25.0, "25"),
            (29.97, "29.97"),
            (30.0, "30"),
            (48.0, "48"),
            (50.0, "50"),
            (59.94, "59.94"),
            (60.0, "60"),
            (120.0, "120")
        ]
        
        for (rate, display) in commonRates {
            if abs(fps - rate) < 0.01 {
                return "\(display) fps"
            }
        }
        
        return String(format: "%.2f fps", fps)
    }
    
    // MARK: - Dates
    
    /// Format date for display
    public static func formatDate(_ date: Date, style: DateFormatter.Style = .medium) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = style
        formatter.timeStyle = .none
        return formatter.string(from: date)
    }
    
    /// Format date and time for display
    public static func formatDateTime(_ date: Date, dateStyle: DateFormatter.Style = .medium, timeStyle: DateFormatter.Style = .short) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = dateStyle
        formatter.timeStyle = timeStyle
        return formatter.string(from: date)
    }
    
    /// Format relative time (e.g., "2 hours ago", "in 5 minutes")
    public static func formatRelativeTime(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter.localizedString(for: date, relativeTo: Date())
    }
    
    // MARK: - Media Specific
    
    /// Format video resolution
    public static func formatResolution(width: Int, height: Int) -> String {
        // Common resolutions
        let resolutions: [(Int, Int, String)] = [
            (3840, 2160, "4K"),
            (2560, 1440, "1440p"),
            (1920, 1080, "1080p"),
            (1280, 720, "720p"),
            (854, 480, "480p"),
            (640, 360, "360p")
        ]
        
        for (w, h, name) in resolutions {
            if width == w && height == h {
                return "\(name) (\(width)×\(height))"
            }
        }
        
        return "\(width)×\(height)"
    }
    
    /// Format audio sample rate
    public static func formatSampleRate(_ rate: Double) -> String {
        guard rate > 0 else { return "Unknown" }
        
        if rate.truncatingRemainder(dividingBy: 1000) == 0 {
            return "\(Int(rate / 1000)) kHz"
        } else {
            return String(format: "%.1f kHz", rate / 1000)
        }
    }
    
    /// Format bitrate
    public static func formatBitrate(_ bitsPerSecond: Int) -> String {
        guard bitsPerSecond > 0 else { return "Unknown" }
        
        let kbps = Double(bitsPerSecond) / 1000.0
        
        if kbps >= 1000 {
            return String(format: "%.1f Mbps", kbps / 1000.0)
        } else {
            return String(format: "%.0f kbps", kbps)
        }
    }
    
    // MARK: - Validation Helpers
    
    /// Check if string is a valid timecode
    public static func isValidTimecode(_ string: String) -> Bool {
        let pattern = "^\\d{1,2}:\\d{2}:\\d{2}(:\\d{2})?$"
        let regex = try? NSRegularExpression(pattern: pattern)
        let range = NSRange(location: 0, length: string.utf16.count)
        return regex?.firstMatch(in: string, options: [], range: range) != nil
    }
    
    /// Parse timecode string to seconds
    public static func parseTimecode(_ timecode: String) -> TimeInterval? {
        let components = timecode.split(separator: ":")
        
        guard components.count >= 3 else { return nil }
        
        guard let hours = Int(components[0]),
              let minutes = Int(components[1]),
              let seconds = Int(components[2]),
              minutes < 60,
              seconds < 60 else {
            return nil
        }
        
        var result = TimeInterval(hours * 3600 + minutes * 60 + seconds)
        
        // Handle frames if present
        if components.count == 4,
           let frames = Int(components[3]) {
            result += TimeInterval(frames) / 30.0 // Assume 30fps as default
        }
        
        return result
    }
}

// MARK: - Extensions for Convenience

public extension TimeInterval {
    /// Format as duration string
    var durationString: String {
        Formatters.formatDuration(self)
    }
    
    /// Format as timecode
    var timecodeString: String {
        Formatters.formatTimecode(self)
    }
    
    /// Format as human-readable duration
    var humanReadableString: String {
        Formatters.humanReadableDuration(self)
    }
}

public extension Int64 {
    /// Format as file size
    var fileSizeString: String {
        Formatters.formatFileSize(self)
    }
}

public extension Double {
    /// Format as percentage
    var percentageString: String {
        Formatters.formatPercentage(self)
    }
    
    /// Format as frame rate
    var frameRateString: String {
        Formatters.formatFrameRate(self)
    }
}

public extension Date {
    /// Format as relative time
    var relativeTimeString: String {
        Formatters.formatRelativeTime(self)
    }
}