import Foundation
import Combine
import AVFoundation
import Accelerate
import SwiftUI

// MARK: - Audio Waveform Generator

public class AudioWaveformGenerator {
    
    public struct WaveformData {
        let samples: [Float]
        let duration: TimeInterval
        let sampleRate: Double
        
        func resample(to targetSamples: Int) -> [Float] {
            guard !samples.isEmpty else { return [] }
            
            if samples.count == targetSamples {
                return samples
            }
            
            var resampled = [Float](repeating: 0, count: targetSamples)
            let ratio = Float(samples.count) / Float(targetSamples)
            
            for i in 0..<targetSamples {
                let sourceIndex = Int(Float(i) * ratio)
                let endIndex = min(Int(Float(i + 1) * ratio), samples.count)
                
                // Take the max value in the range for better visualization
                var maxValue: Float = 0
                for j in sourceIndex..<endIndex {
                    maxValue = max(maxValue, abs(samples[j]))
                }
                resampled[i] = maxValue
            }
            
            return resampled
        }
    }
    
    private let downsampleFactor = 100 // Process every 100th sample for performance
    
    public init() {}
    
    /// Generate waveform data from audio URL
    public func generateWaveform(from url: URL, completion: @escaping (Result<WaveformData, Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let waveform = try self.processAudioFile(url: url)
                DispatchQueue.main.async {
                    completion(.success(waveform))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    private func processAudioFile(url: URL) throws -> WaveformData {
        let asset = AVAsset(url: url)
        
        guard let track = asset.tracks(withMediaType: .audio).first else {
            throw WaveformError.noAudioTrack
        }
        
        let reader = try AVAssetReader(asset: asset)
        
        let outputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
        
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        
        var samples: [Float] = []
        
        reader.startReading()
        
        while reader.status == .reading {
            guard let sampleBuffer = output.copyNextSampleBuffer() else { continue }
            
            let processedSamples = processSampleBuffer(sampleBuffer)
            samples.append(contentsOf: processedSamples)
        }
        
        if reader.status == .failed {
            throw reader.error ?? WaveformError.processingFailed
        }
        
        // Normalize samples
        let normalizedSamples = normalizeSamples(samples)
        
        return WaveformData(
            samples: normalizedSamples,
            duration: asset.duration.seconds,
            sampleRate: Double(track.naturalTimeScale)
        )
    }
    
    private func processSampleBuffer(_ sampleBuffer: CMSampleBuffer) -> [Float] {
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
            return []
        }
        
        let length = CMBlockBufferGetDataLength(blockBuffer)
        var data = Data(count: length)
        
        data.withUnsafeMutableBytes { bytes in
            CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: length, destination: bytes.baseAddress!)
        }
        
        // Convert Int16 samples to Float
        let int16Samples = data.withUnsafeBytes { bytes in
            bytes.bindMemory(to: Int16.self)
        }
        
        var floatSamples: [Float] = []
        
        // Downsample for performance
        for i in stride(from: 0, to: int16Samples.count, by: downsampleFactor) {
            let sample = Float(int16Samples[i]) / Float(Int16.max)
            floatSamples.append(sample)
        }
        
        return floatSamples
    }
    
    private func normalizeSamples(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return [] }
        
        // Find peak
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        
        guard peak > 0 else { return samples }
        
        // Normalize to 0...1 range
        var normalized = [Float](repeating: 0, count: samples.count)
        var scale = 1.0 / peak
        vDSP_vsmul(samples, 1, &scale, &normalized, 1, vDSP_Length(samples.count))
        
        return normalized
    }
    
    enum WaveformError: LocalizedError {
        case noAudioTrack
        case processingFailed
        
        var errorDescription: String? {
            switch self {
            case .noAudioTrack:
                return "No audio track found in file"
            case .processingFailed:
                return "Failed to process audio file"
            }
        }
    }
}

// MARK: - Waveform View

public struct WaveformView: View {
    let waveformData: AudioWaveformGenerator.WaveformData
    let color: Color
    let backgroundColor: Color
    
    @State private var displaySamples: [Float] = []
    
    public init(waveformData: AudioWaveformGenerator.WaveformData,
                color: Color = .blue,
                backgroundColor: Color = .clear) {
        self.waveformData = waveformData
        self.color = color
        self.backgroundColor = backgroundColor
    }
    
    public var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                // Draw background
                context.fill(
                    Path(CGRect(origin: .zero, size: size)),
                    with: .color(backgroundColor)
                )
                
                // Calculate samples to display
                let targetSamples = Int(size.width)
                let samples = displaySamples.isEmpty ? 
                    waveformData.resample(to: targetSamples) : displaySamples
                
                guard !samples.isEmpty else { return }
                
                let midY = size.height / 2
                let amplitudeScale = size.height / 2 * 0.8 // 80% of half height
                
                // Draw waveform
                var path = Path()
                
                // Top half
                path.move(to: CGPoint(x: 0, y: midY))
                
                for (index, sample) in samples.enumerated() {
                    let x = CGFloat(index)
                    let y = midY - CGFloat(abs(sample)) * amplitudeScale
                    
                    if index == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                
                // Bottom half (mirror)
                for (index, sample) in samples.enumerated().reversed() {
                    let x = CGFloat(index)
                    let y = midY + CGFloat(abs(sample)) * amplitudeScale
                    path.addLine(to: CGPoint(x: x, y: y))
                }
                
                path.closeSubpath()
                
                // Draw the waveform
                context.fill(path, with: .color(color.opacity(0.6)))
                context.stroke(path, with: .color(color), lineWidth: 0.5)
                
                // Draw center line
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: 0, y: midY))
                        path.addLine(to: CGPoint(x: size.width, y: midY))
                    },
                    with: .color(color.opacity(0.3)),
                    lineWidth: 0.5
                )
            }
            .onAppear {
                updateDisplaySamples(for: geometry.size.width)
            }
            .onChange(of: geometry.size.width) { newWidth in
                updateDisplaySamples(for: newWidth)
            }
        }
    }
    
    private func updateDisplaySamples(for width: CGFloat) {
        let targetSamples = Int(width)
        displaySamples = waveformData.resample(to: targetSamples)
    }
}

// MARK: - Audio Level Meter

public struct AudioLevelMeter: View {
    let level: Float // 0.0 to 1.0
    let peakLevel: Float
    let isClipping: Bool
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background
                Rectangle()
                    .fill(Color.black.opacity(0.3))
                
                // Level bar
                Rectangle()
                    .fill(levelColor)
                    .frame(width: geometry.size.width * CGFloat(level))
                
                // Peak indicator
                Rectangle()
                    .fill(Color.white)
                    .frame(width: 2)
                    .offset(x: geometry.size.width * CGFloat(peakLevel) - 1)
                
                // Clipping indicator
                if isClipping {
                    Rectangle()
                        .fill(Color.red)
                        .frame(width: 4)
                        .offset(x: geometry.size.width - 4)
                }
            }
        }
        .frame(height: 8)
    }
    
    private var levelColor: LinearGradient {
        LinearGradient(
            colors: [
                Color.green,
                Color.green,
                Color.yellow,
                Color.orange,
                Color.red
            ],
            startPoint: .leading,
            endPoint: .trailing
        )
    }
}

// MARK: - Audio Analysis

public class AudioLevelAnalyzer: ObservableObject {
    @Published var currentLevel: Float = 0
    @Published var peakLevel: Float = 0
    @Published var isClipping = false
    
    private var audioEngine: AVAudioEngine?
    private var analyzerNode: AVAudioMixerNode?
    private let updateInterval: TimeInterval = 1.0 / 30.0 // 30 FPS
    private var updateTimer: Timer?
    
    public init() {}
    
    public func startAnalyzing(player: AVPlayer) {
        setupAudioEngine()
        startLevelMonitoring()
    }
    
    public func stopAnalyzing() {
        updateTimer?.invalidate()
        audioEngine?.stop()
    }
    
    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        analyzerNode = AVAudioMixerNode()
        
        guard let engine = audioEngine,
              let analyzer = analyzerNode else { return }
        
        engine.attach(analyzer)
        engine.connect(analyzer, to: engine.mainMixerNode, format: nil)
        
        do {
            try engine.start()
        } catch {
            print("Failed to start audio engine: \(error)")
        }
    }
    
    private func startLevelMonitoring() {
        updateTimer = Timer.scheduledTimer(withTimeInterval: updateInterval, repeats: true) { [weak self] _ in
            self?.updateLevels()
        }
    }
    
    private func updateLevels() {
        guard let analyzer = analyzerNode else { return }
        
        // Install tap to get audio data
        analyzer.installTap(onBus: 0, bufferSize: 1024, format: nil) { [weak self] buffer, _ in
            guard let self = self,
                  let channelData = buffer.floatChannelData else { return }
            
            let channelCount = Int(buffer.format.channelCount)
            let frameLength = Int(buffer.frameLength)
            
            var maxLevel: Float = 0
            
            for channel in 0..<channelCount {
                let samples = channelData[channel]
                
                for i in 0..<frameLength {
                    let sample = abs(samples[i])
                    maxLevel = max(maxLevel, sample)
                }
            }
            
            DispatchQueue.main.async {
                self.currentLevel = maxLevel
                
                if maxLevel > self.peakLevel {
                    self.peakLevel = maxLevel
                }
                
                self.isClipping = maxLevel >= 0.99
                
                // Decay peak level slowly
                self.peakLevel *= 0.99
            }
        }
    }
}
