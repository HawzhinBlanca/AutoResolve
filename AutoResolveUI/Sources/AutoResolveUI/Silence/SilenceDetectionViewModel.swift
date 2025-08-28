import Foundation
import SwiftUI
import AVFoundation
import Accelerate
import Combine

// MARK: - Silence Detection View Model
@MainActor
public class SilenceDetectionViewModel: ObservableObject {
    // Audio Data
    @Published public var audioURL: URL?
    @Published public var audioDuration: TimeInterval = 0
    @Published public var waveformData: [Float] = []
    
    // Detection Parameters
    @Published public var silenceThresholdDB: Double = -40.0
    @Published public var minSilenceDuration: TimeInterval = 0.5
    @Published public var paddingBefore: TimeInterval = 0.1
    @Published public var paddingAfter: TimeInterval = 0.1
    
    // Advanced Settings
    @Published public var useRMSAnalysis = true
    @Published public var useAdaptiveThreshold = false
    @Published public var mergeAdjacentRegions = true
    @Published public var mergeGapThreshold: TimeInterval = 0.3
    @Published public var processingQuality = ProcessingQuality.balanced
    @Published public var realtimePreview = false
    
    // Detection Results
    @Published public var detectedSilenceRegions: [SilenceRegion] = []
    @Published public var selectedRegions: Set<UUID> = []
    @Published public var totalSilenceDuration: TimeInterval = 0
    @Published public var silencePercentage: Double = 0
    
    // Processing State
    @Published public var isProcessing = false
    @Published public var processingProgress: Double = 0
    @Published public var zoomLevel: Double = 1.0
    
    private var audioFile: AVAudioFile?
    private var audioFormat: AVAudioFormat?
    private let backendService = AutoResolveService()
    private var cancellables = Set<AnyCancellable>()
    
    public init() {}
    
    // MARK: - Audio Loading
    
    public func loadAudioFromVideo(url: URL) {
        Task {
            do {
                // Extract audio from video
                let asset = AVAsset(url: url)
                guard let _ = try await asset.loadTracks(withMediaType: .audio).first else {
                    print("No audio track found in video")
                    return
                }
                
                // Get audio duration
                let duration = try await asset.load(.duration)
                self.audioDuration = CMTimeGetSeconds(duration)
                
                // Store audio URL for processing
                self.audioURL = url
                
                // Load waveform data
                await loadWaveformData(from: url)
                
            } catch {
                print("Error loading audio: \(error)")
            }
        }
    }
    
    private func loadWaveformData(from url: URL) async {
        do {
            // Create audio file
            audioFile = try AVAudioFile(forReading: url)
            audioFormat = audioFile?.processingFormat
            
            guard let file = audioFile,
                  let format = audioFormat else { return }
            
            // Read audio data
            let frameCount = AVAudioFrameCount(file.length)
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
            
            try file.read(into: buffer, frameCount: frameCount)
            
            // Convert to waveform data
            if let floatData = buffer.floatChannelData?[0] {
                let samples = Array(UnsafeBufferPointer(start: floatData, count: Int(frameCount)))
                
                // Downsample for visualization
                let downsampledData = downsample(samples, targetSampleCount: 1000)
                
                await MainActor.run {
                    self.waveformData = downsampledData
                }
            }
            
        } catch {
            print("Error loading waveform: \(error)")
        }
    }
    
    private func downsample(_ samples: [Float], targetSampleCount: Int) -> [Float] {
        guard samples.count > targetSampleCount else { return samples }
        
        let ratio = samples.count / targetSampleCount
        var downsampled: [Float] = []
        
        for i in 0..<targetSampleCount {
            let startIdx = i * ratio
            let endIdx = min((i + 1) * ratio, samples.count)
            
            if startIdx < endIdx {
                let slice = samples[startIdx..<endIdx]
                let peak = slice.map { abs($0) }.max() ?? 0
                downsampled.append(peak)
            }
        }
        
        return downsampled
    }
    
    // MARK: - Silence Detection
    
    public func detectSilence() async {
        guard let url = audioURL else { return }
        
        await MainActor.run {
            self.isProcessing = true
            self.processingProgress = 0
            self.detectedSilenceRegions.removeAll()
        }
        
        do {
            // Use backend service for detection
            let settings = ViewModelSilenceDetectionSettings(
                thresholdDB: silenceThresholdDB,
                minDuration: minSilenceDuration,
                paddingBefore: paddingBefore,
                paddingAfter: paddingAfter
            )
            
            let results = try await backendService.detectSilence(
                videoPath: url.path,
                settings: BackendSilenceDetectionSettings(
                    threshold: settings.threshold,
                    minDuration: settings.minDuration,
                    padding: settings.paddingAfter
                )
            )
            
            // Convert results to UI format
            let regions = results.silenceSegments.map { range in
                SilenceRegion(
                    startTime: range.start,
                    endTime: range.end,
                    duration: range.end - range.start,
                    averageLevel: silenceThresholdDB,
                    confidence: 0.8,
                    isSelected: false
                )
            }
            
            await MainActor.run {
                self.detectedSilenceRegions = regions
                self.calculateStatistics()
                self.isProcessing = false
                self.processingProgress = 1.0
            }
            
        } catch {
            print("Silence detection failed: \(error)")
            
            // Fallback to local detection
            await performLocalSilenceDetection()
        }
    }
    
    private func performLocalSilenceDetection() async {
        guard let file = audioFile,
              let format = audioFormat else { return }
        
        let frameCount = AVAudioFrameCount(file.length)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        
        do {
            try file.read(into: buffer, frameCount: frameCount)
            
            guard let floatData = buffer.floatChannelData?[0] else { return }
            
            let sampleRate = format.sampleRate
            let samples = Array(UnsafeBufferPointer(start: floatData, count: Int(frameCount)))
            
            // Detect silence regions
            var regions: [SilenceRegion] = []
            var inSilence = false
            var silenceStart: TimeInterval = 0
            
            let windowSize = Int(sampleRate * 0.01) // 10ms windows
            let threshold = dbToLinear(silenceThresholdDB)
            
            for i in stride(from: 0, to: samples.count, by: windowSize) {
                let endIdx = min(i + windowSize, samples.count)
                let window = Array(samples[i..<endIdx])
                
                let rms = useRMSAnalysis ? calculateRMS(window) : calculatePeak(window)
                let currentTime = Double(i) / sampleRate
                
                // Update progress
                await MainActor.run {
                    self.processingProgress = Double(i) / Double(samples.count)
                }
                
                if rms < threshold {
                    if !inSilence {
                        inSilence = true
                        silenceStart = currentTime
                    }
                } else {
                    if inSilence {
                        let silenceEnd = currentTime
                        let duration = silenceEnd - silenceStart
                        
                        if duration >= minSilenceDuration {
                            // Apply padding
                            let paddedStart = max(0, silenceStart - paddingBefore)
                            let paddedEnd = min(audioDuration, silenceEnd + paddingAfter)
                            
                            regions.append(SilenceRegion(
                                startTime: paddedStart,
                                endTime: paddedEnd,
                                duration: paddedEnd - paddedStart,
                                averageLevel: linearToDb(rms),
                                confidence: 0.8,
                                isSelected: false
                            ))
                        }
                        
                        inSilence = false
                    }
                }
            }
            
            // Handle silence at the end
            if inSilence {
                let silenceEnd = audioDuration
                let duration = silenceEnd - silenceStart
                
                if duration >= minSilenceDuration {
                    regions.append(SilenceRegion(
                        startTime: silenceStart,
                        endTime: silenceEnd,
                        duration: duration,
                        averageLevel: silenceThresholdDB,
                        confidence: 0.8,
                        isSelected: false
                    ))
                }
            }
            
            // Merge adjacent regions if enabled
            if mergeAdjacentRegions {
                regions = mergeNearbyRegions(regions)
            }
            
            await MainActor.run {
                self.detectedSilenceRegions = regions
                self.calculateStatistics()
                self.isProcessing = false
                self.processingProgress = 1.0
            }
            
        } catch {
            print("Local silence detection failed: \(error)")
            await MainActor.run {
                self.isProcessing = false
            }
        }
    }
    
    private func calculateRMS(_ samples: [Float]) -> Float {
        let squaredSum = samples.reduce(0) { $0 + $1 * $1 }
        return sqrt(squaredSum / Float(samples.count))
    }
    
    private func calculatePeak(_ samples: [Float]) -> Float {
        return samples.map { abs($0) }.max() ?? 0
    }
    
    private func dbToLinear(_ db: Double) -> Float {
        return Float(pow(10, db / 20))
    }
    
    private func linearToDb(_ linear: Float) -> Double {
        return 20 * log10(Double(max(linear, 0.00001)))
    }
    
    private func mergeNearbyRegions(_ regions: [SilenceRegion]) -> [SilenceRegion] {
        guard !regions.isEmpty else { return regions }
        
        var merged: [SilenceRegion] = []
        var currentRegion = regions[0]
        
        for i in 1..<regions.count {
            let nextRegion = regions[i]
            let gap = nextRegion.startTime - currentRegion.endTime
            
            if gap <= mergeGapThreshold {
                // Merge regions
                currentRegion = SilenceRegion(
                    startTime: currentRegion.startTime,
                    endTime: nextRegion.endTime,
                    duration: nextRegion.endTime - currentRegion.startTime,
                    averageLevel: (currentRegion.averageLevel + nextRegion.averageLevel) / 2,
                    confidence: (currentRegion.confidence + nextRegion.confidence) / 2,
                    isSelected: false
                )
            } else {
                merged.append(currentRegion)
                currentRegion = nextRegion
            }
        }
        
        merged.append(currentRegion)
        return merged
    }
    
    // MARK: - Statistics
    
    private func calculateStatistics() {
        totalSilenceDuration = detectedSilenceRegions.reduce(0) { $0 + $1.duration }
        
        if audioDuration > 0 {
            silencePercentage = (totalSilenceDuration / audioDuration) * 100
        } else {
            silencePercentage = 0
        }
    }
    
    // MARK: - Selection Management
    
    public func toggleRegionSelection(_ region: SilenceRegion) {
        if let index = detectedSilenceRegions.firstIndex(where: { $0.id == region.id }) {
            detectedSilenceRegions[index].isSelected.toggle()
            
            if detectedSilenceRegions[index].isSelected {
                selectedRegions.insert(region.id)
            } else {
                selectedRegions.remove(region.id)
            }
        }
    }
    
    public func selectAllRegions() {
        for i in 0..<detectedSilenceRegions.count {
            detectedSilenceRegions[i].isSelected = true
            selectedRegions.insert(detectedSilenceRegions[i].id)
        }
    }
    
    public func deselectAllRegions() {
        for i in 0..<detectedSilenceRegions.count {
            detectedSilenceRegions[i].isSelected = false
        }
        selectedRegions.removeAll()
    }
    
    // MARK: - Timeline Integration
    
    public func removeSelectedSilenceFromTimeline(timeline: TimelineModel) {
        let selectedSilenceRegions = detectedSilenceRegions.filter { $0.isSelected }
        
        for region in selectedSilenceRegions {
            removeTimeRange(from: region.startTime, to: region.endTime, in: timeline)
        }
        
        // Clear selection after removal
        deselectAllRegions()
    }
    
    private func removeTimeRange(from startTime: TimeInterval, to endTime: TimeInterval, in timeline: TimelineModel) {
        let duration = endTime - startTime
        
        // Adjust clips in all tracks
        for trackIndex in timeline.tracks.indices {
            guard timeline.tracks[trackIndex].type == .video else { continue }
            
            for clipIndex in timeline.tracks[trackIndex].clips.indices {
                let clip = timeline.tracks[trackIndex].clips[clipIndex]
                if clip.startTime >= endTime {
                    // Shift clips after the silence region
                    timeline.tracks[trackIndex].clips[clipIndex].startTime -= duration
                } else if clip.startTime + clip.duration ?? 0 > startTime && clip.startTime < endTime {
                    // Clip overlaps with silence region
                    if clip.startTime >= startTime {
                        // Clip starts within silence - remove it
                        timeline.tracks[trackIndex].clips.removeAll { $0.id == clip.id }
                    } else {
                        // Trim clip that extends into silence
                        timeline.tracks[trackIndex].clips[clipIndex].duration = startTime - clip.startTime
                    }
                }
            }
        }
        
        // Adjust timeline duration
        timeline.duration -= duration
    }
    
    public func applySilenceDetectionToTimeline(timeline: TimelineModel) {
        // Mark silence regions on timeline without removing them
        for region in detectedSilenceRegions {
            let marker = UITimelineMarker(time: region.startTime, type: .silence, name: "Silence")
            timeline.markers.append(marker)
        }
    }
    
    // MARK: - Export
    
    public func exportEDL(to url: URL) {
        var edlContent = "TITLE: Silence Detection EDL\n"
        edlContent += "FCM: NON-DROP FRAME\n\n"
        
        for (index, region) in detectedSilenceRegions.enumerated() {
            let startTC = timecodeFromSeconds(region.startTime)
            let endTC = timecodeFromSeconds(region.endTime)
            
            edlContent += String(format: "%03d  AX       V     C        ", index + 1)
            edlContent += "\(startTC) \(endTC) "
            edlContent += "\(startTC) \(endTC)\n"
            edlContent += "* SILENCE REGION\n\n"
        }
        
        do {
            try edlContent.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            print("Failed to export EDL: \(error)")
        }
    }
    
    private func timecodeFromSeconds(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        let frames = Int((seconds - Double(Int(seconds))) * 30)
        
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
    }
    
    // MARK: - Settings
    
    public func resetToDefaults() {
        silenceThresholdDB = -40.0
        minSilenceDuration = 0.5
        paddingBefore = 0.1
        paddingAfter = 0.1
        useRMSAnalysis = true
        useAdaptiveThreshold = false
        mergeAdjacentRegions = true
        mergeGapThreshold = 0.3
        processingQuality = .balanced
        realtimePreview = false
    }
}

// MARK: - Data Models

// SilenceRegion is now defined in AnalysisTypes.swift
// Additional properties for UI state
// SilenceRegion now includes id/averageLevel/isSelected in AnalysisTypes

public enum ProcessingQuality: String, CaseIterable {
    case fast = "Fast"
    case balanced = "Balanced"
    case high = "High"
}

public struct SilenceTimelineMarker: Identifiable {
    public let id = UUID()
    public let time: TimeInterval
    public let name: String
    public let color: Color
    public let type: MarkerType
    
    public enum MarkerType {
        case silence
        case sceneChange
        case manual
        case auto
    }
}

// Backend communication structures
struct ViewModelSilenceDetectionSettings: Codable {
    let thresholdDB: Double
    let minDuration: TimeInterval
    let paddingBefore: TimeInterval
    let paddingAfter: TimeInterval
    
    // Additional properties for compatibility
    var threshold: Double { thresholdDB }
    var maxDuration: TimeInterval { 10.0 }  // Default max duration
    var leadingPadding: TimeInterval { paddingBefore }
    var trailingPadding: TimeInterval { paddingAfter }
}

struct ViewModelSilenceDetectionResult: Codable {
    let silenceRanges: [SilenceRange]
    let totalDuration: TimeInterval
    let silencePercentage: Double
}

struct SilenceRange: Codable {
    let start: TimeInterval
    let end: TimeInterval
    let averageLevel: Double?
}
