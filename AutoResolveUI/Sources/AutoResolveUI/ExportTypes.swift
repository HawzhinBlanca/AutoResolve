// Imports
import Foundation
import SwiftUI
import AVFoundation

// MARK: - Export Formats

public enum ExportFormat: String, CaseIterable, Codable, Identifiable, Sendable {
    public var id: String { rawValue }
    
    // Timeline formats
    case fcpxml = "fcpxml"
    case edl = "edl"
    case xml = "xml"
    case aaf = "aaf"
    case drp = "drp"      // DaVinci Resolve Project
    case otio = "otio"    // OpenTimelineIO
    
    // Video formats
    case proresHQ = "prores_hq"
    case proresStandard = "prores_standard"
    case prores422 = "prores_422"           // Standard ProRes 422
    case prores4444 = "prores_4444"         // ProRes 4444
    case proresLT = "prores_lt"
    case proresProxy = "prores_proxy"
    case prores_mov = "prores_mov"  // ProRes MOV container
    case h264 = "h264"
    case h264_mp4 = "h264_mp4"      // H.264 MP4 container
    case h265 = "h265"
    case h265_mp4 = "h265_mp4"      // H.265 MP4 container
    case vp9 = "vp9"                // VP9 codec
    case av1 = "av1"                // AV1 codec
    case dnxhd = "dnxhd"
    case dnxhr = "dnxhr"
    case dnxhd_mov = "dnxhd_mov"    // DNxHD MOV container
    
    // Image/Animation formats
    case gif = "gif"
    case image_sequence = "image_sequence"
    
    // Container formats
    case mp4 = "mp4"
    case mov = "mov"  
    case webm = "webm"
    
    // Audio formats
    case wav = "wav"
    case aiff = "aiff"
    case mp3 = "mp3"
    case m4a = "m4a"
    
    public var displayName: String {
        switch self {
        case .fcpxml: return "Final Cut Pro XML"
        case .edl: return "EDL"
        case .xml: return "XML"
        case .aaf: return "AAF"
        case .drp: return "DaVinci Resolve Project"
        case .otio: return "OpenTimelineIO"
        case .proresHQ: return "ProRes 422 HQ"
        case .proresStandard: return "ProRes 422"
        case .prores422: return "ProRes 422"
        case .prores4444: return "ProRes 4444"
        case .proresLT: return "ProRes 422 LT"
        case .proresProxy: return "ProRes Proxy"
        case .prores_mov: return "ProRes MOV"
        case .h264: return "H.264"
        case .h264_mp4: return "H.264 MP4"
        case .h265: return "H.265 (HEVC)"
        case .h265_mp4: return "H.265 MP4"
        case .vp9: return "VP9"
        case .av1: return "AV1"
        case .dnxhd: return "DNxHD"
        case .dnxhr: return "DNxHR"
        case .dnxhd_mov: return "DNxHD MOV"
        case .gif: return "GIF Animation"
        case .image_sequence: return "Image Sequence"
        case .mp4: return "MP4 Container"
        case .mov: return "MOV Container"
        case .webm: return "WebM"
        case .wav: return "WAV"
        case .aiff: return "AIFF"
        case .mp3: return "MP3"
        case .m4a: return "M4A"
        }
    }
    
    public var fileExtension: String {
        switch self {
        case .fcpxml: return "fcpxml"
        case .edl: return "edl"
        case .xml: return "xml"
        case .aaf: return "aaf"
        case .drp: return "drp"
        case .otio: return "otio"
        case .proresHQ, .proresStandard, .prores422, .prores4444, .proresLT, .proresProxy, .prores_mov: return "mov"
        case .h264: return "mp4"
        case .h264_mp4: return "mp4"
        case .h265: return "mp4"
        case .h265_mp4: return "mp4"
        case .vp9: return "webm"
        case .av1: return "mp4"
        case .dnxhd, .dnxhr: return "mxf"
        case .dnxhd_mov: return "mov"
        case .gif: return "gif"
        case .image_sequence: return "dpx"
        case .mp4: return "mp4"
        case .mov: return "mov"
        case .webm: return "webm"
        case .wav: return "wav"
        case .aiff: return "aiff"
        case .mp3: return "mp3"
        case .m4a: return "m4a"
        }
    }
    
    public var icon: String {
        switch self {
        case .h264_mp4, .h265_mp4: return "play.rectangle"
        case .prores_mov, .dnxhd_mov: return "film"
        case .gif: return "photo.on.rectangle.angled"
        case .image_sequence: return "photo.stack"
        case .fcpxml: return "doc.text"
        case .drp: return "doc.badge.gearshape"
        case .aaf: return "doc.richtext"
        case .edl: return "doc.plaintext"
        case .otio: return "doc.badge.arrow.up.arrow.down"
        default: return "doc"
        }
    }
}

// MARK: - Resolution

public enum ResolutionPreset: String, Codable, CaseIterable, Sendable {
    case hd = "1280x720p"         // Alias for hd720 with different raw value
    case hd720 = "1280x720"
    case hd1080 = "1920x1080"
    case fullHD = "1920x1080p"    // Unique raw value for alias
    case uhd4k = "3840x2160"
    case uhd4K = "3840x2160p"     // Alias with different raw value
    case dci4k = "4096x2160"
    case dci4K = "4096x2160p"     // Alias with different raw value
    case uhd8k = "7680x4320"
    
    // Social media formats
    case square = "1080x1080"
    case vertical9_16 = "1080x1920"
    case vertical4_5 = "1080x1350"
    
    public var width: Int {
        let components = rawValue.split(separator: "x")
        return Int(components[0]) ?? 1920
    }
    
    public var height: Int {
        let components = rawValue.split(separator: "x")
        return Int(components[1]) ?? 1080
    }
    
    public var size: CGSize {
        CGSize(width: width, height: height)
    }
    
    public var displayName: String {
        switch self {
        case .hd: return "HD"
        case .hd720: return "HD 720p"
        case .hd1080: return "Full HD 1080p"
        case .fullHD: return "Full HD 1080p"
        case .uhd4k: return "4K UHD"
        case .uhd4K: return "4K UHD"
        case .dci4k: return "4K DCI"
        case .dci4K: return "4K DCI"
        case .uhd8k: return "8K UHD"
        case .square: return "Square (1:1)"
        case .vertical9_16: return "Vertical (9:16)"
        case .vertical4_5: return "Vertical (4:5)"
        }
    }
}

// MARK: - Frame Rate (Uncommented and fixed)

public enum FrameRate: Double, Codable, CaseIterable {
    case fps23_976 = 23.976
    case fps24 = 24.0
    case fps25 = 25.0
    case fps29_97 = 29.97
    case fps30 = 30.0
    case fps48 = 48.0
    case fps50 = 50.0
    case fps59_94 = 59.94
    case fps60 = 60.0
    case fps120 = 120.0
    
    public var displayName: String {
        switch self {
        case .fps23_976: return "23.976 fps"
        case .fps24: return "24 fps"
        case .fps25: return "25 fps"
        case .fps29_97: return "29.97 fps"
        case .fps30: return "30 fps"
        case .fps48: return "48 fps"
        case .fps50: return "50 fps"
        case .fps59_94: return "59.94 fps"
        case .fps60: return "60 fps"
        case .fps120: return "120 fps"
        }
    }
    
    public var value: Double {
        return self.rawValue
    }
    
    public var isDropFrame: Bool {
        switch self {
        case .fps23_976, .fps29_97, .fps59_94:
            return true
        default:
            return false
        }
    }
}

// MARK: - Aspect Ratio

public enum AspectRatio: String, Codable, CaseIterable {
    case standard = "4:3"
    case widescreen = "16:9"
    case cinema = "21:9"
    case vertical = "9:16"
    case square = "1:1"
    
    public var value: Double {
        switch self {
        case .standard: return 4.0 / 3.0
        case .widescreen: return 16.0 / 9.0
        case .cinema: return 21.0 / 9.0
        case .vertical: return 9.0 / 16.0
        case .square: return 1.0
        }
    }
}
