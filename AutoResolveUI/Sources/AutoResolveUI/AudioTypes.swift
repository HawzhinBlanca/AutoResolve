// Imports
import Foundation
import AVFoundation

// MARK: - Audio Configuration

public enum AudioSampleRate: Int, Codable, CaseIterable {
    case rate44100 = 44100
    case rate48000 = 48000
    case rate88200 = 88200
    case rate96000 = 96000
    case rate192000 = 192000
    
    public var displayName: String {
        "\(rawValue / 1000) kHz"
    }
}

public enum AudioChannels: Int, Codable, CaseIterable {
    case mono = 1
    case stereo = 2
    case surround51 = 6
    case surround71 = 8
    
    public var displayName: String {
        switch self {
        case .mono: return "Mono"
        case .stereo: return "Stereo"
        case .surround51: return "5.1 Surround"
        case .surround71: return "7.1 Surround"
        }
    }
}

// MARK: - Audio Processing Types

public struct AudioProcessingBuffer: Codable, Sendable {
    public let samples: [Float]
    public let sampleRate: Double
    public let channels: Int
    
    public init(samples: [Float], sampleRate: Double, channels: Int = 1) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.channels = channels
    }
}

public struct WaveformData: Codable, Sendable {
    public let samples: [Float]
    public let sampleRate: Double
    public let channels: Int
    public let duration: TimeInterval
    
    public init(samples: [Float] = [], sampleRate: Double = 44100, channels: Int = 2, duration: TimeInterval = 0) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.channels = channels
        self.duration = duration
    }
    
    public func resample(to targetSamples: Int) -> [Float] {
        guard !samples.isEmpty else { return [] }
        
        if samples.count == targetSamples {
            return samples
        }
        
        var resampled = [Float](repeating: 0, count: targetSamples)
        let ratio = Float(samples.count) / Float(targetSamples)
        
        for i in 0..<targetSamples {
            let sourceIndex = Double(i) * Double(ratio)
            let lowerIndex = Int(sourceIndex)
            let upperIndex = min(lowerIndex + 1, samples.count - 1)
            let fraction = Float(sourceIndex - Double(lowerIndex))
            
            resampled[i] = samples[lowerIndex] * (1 - fraction) + samples[upperIndex] * fraction
        }
        
        return resampled
    }
}

public struct AudioEffect: Codable, Sendable, Identifiable {
    public let id = UUID()
    public let name: String
    public let type: AudioEffectType
    public let parameters: [String: Double]
    public let isEnabled: Bool
    
    public init(name: String, type: AudioEffectType, parameters: [String: Double] = [:], isEnabled: Bool = true) {
        self.name = name
        self.type = type
        self.parameters = parameters
        self.isEnabled = isEnabled
    }
}

public enum AudioEffectType: String, Codable, CaseIterable, Sendable {
    case equalizer = "equalizer"
    case compressor = "compressor"
    case reverb = "reverb"
    case delay = "delay"
    case chorus = "chorus"
    case distortion = "distortion"
    case filter = "filter"
    case noise_reduction = "noise_reduction"
    case normalize = "normalize"
    case limiter = "limiter"
}

// Audio Analysis Extensions
public struct AudioFeatureExtraction: Codable {
    public let spectrograms: [SpectrogramFrame]
    public let acousticFeatures: AcousticFeatures
}
