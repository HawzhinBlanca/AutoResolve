import Foundation

public struct PipelineOptions: Codable {
    public var enableSilenceDetection: Bool = true
    public var enableTranscription: Bool = true
    public var enableStoryBeats: Bool = true
    public var enableBRollSelection: Bool = true
    public var silenceThreshold: Double = 0.01
    public var minSilenceDuration: Double = 0.5
    
    public init() {}
}