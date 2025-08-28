// AUTORESOLVE V3.0 - AUDIO ENGINE
import Combine
// Enterprise-grade audio processing with AVAudioEngine for real-time scrubbing

import Foundation
import AVFoundation
import Accelerate
import CoreAudio
import AppKit

// MARK: - Audio Engine Protocol
protocol AudioEngineProtocol: AnyObject {
    var isPlaying: Bool { get }
    var currentTime: TimeInterval { get }
    var duration: TimeInterval { get }
    var volume: Float { get set }
    var rate: Float { get set }
    
    func loadAudio(from url: URL) async throws
    func play()
    func pause()
    func stop()
    func seek(to time: TimeInterval)
    func scrub(at rate: Float)
    func extractWaveform(resolution: Int) async -> [Float]
    func detectSilence(threshold: Float, minDuration: TimeInterval) async -> [AudioTimeRange]
}

// MARK: - Audio Time Range
struct AudioTimeRange: Equatable {
    let start: TimeInterval
    let end: TimeInterval
    var duration: TimeInterval { end - start }
}

// MARK: - Audio Engine Implementation
@MainActor
final class AudioEngine: NSObject, AudioEngineProtocol, ObservableObject {
    // MARK: - Published Properties
    @Published private(set) var isPlaying = false
    @Published private(set) var currentTime: TimeInterval = 0
    @Published private(set) var duration: TimeInterval = 0
    @Published var volume: Float = 1.0 {
        didSet { updateVolume() }
    }
    @Published var rate: Float = 1.0 {
        didSet { updateRate() }
    }
    @Published private(set) var waveformData: [Float] = []
    @Published private(set) var silenceSegments: [AudioTimeRange] = []
    @Published private(set) var audioLevels: AudioLevels = AudioLevels()
    
    // Audio levels for VU meters
    struct AudioLevels {
        var leftChannel: Float = 0
        var rightChannel: Float = 0
        var leftPeak: Float = 0
        var rightPeak: Float = 0
    }
    
    // MARK: - Private Properties
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private var audioFile: AVAudioFile?
    private var audioBuffer: AVAudioPCMBuffer?
    private let mixerNode = AVAudioMixerNode()
    private let timeEffect = AVAudioUnitTimePitch()
    
    // Scrubbing support
    private var scrubbingTimer: Timer?
    private var lastScrubbingPosition: AVAudioFramePosition = 0
    private var isScrubbing = false
    
    // Audio analysis
    private let fftSetup: FFTSetup
    private let fftLength = 2048
    private var audioFormat: AVAudioFormat?
    
    // Tap for real-time audio levels
    private var audioTap: UUID?
    
    // MARK: - Initialization
    override init() {
        // Setup FFT for audio analysis
        self.fftSetup = vDSP_create_fftsetup(vDSP_Length(log2(Float(fftLength))), FFTRadix(kFFTRadix2))!
        
        super.init()
        setupAudioEngine()
        setupAudioSession()
    }
    
    deinit {
        vDSP_destroy_fftsetup(fftSetup)
        engine.stop()
    }
    
    // MARK: - Setup
    private func setupAudioEngine() {
        // Attach nodes
        engine.attach(playerNode)
        engine.attach(timeEffect)
        engine.attach(mixerNode)
        
        // Connect nodes: player -> time effect -> mixer -> output
        engine.connect(playerNode, to: timeEffect, format: nil)
        engine.connect(timeEffect, to: mixerNode, format: nil)
        engine.connect(mixerNode, to: engine.mainMixerNode, format: nil)
        
        // Setup audio tap for level monitoring
        setupAudioTap()
        
        // Prepare engine
        engine.prepare()
    }
    
    private func setupAudioSession() {
        #if os(macOS)
        // macOS doesn't use AVAudioSession
        #else
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
        } catch {
            print("Failed to setup audio session: \(error)")
        }
        #endif
    }
    
    private func setupAudioTap() {
        let format = mixerNode.outputFormat(forBus: 0)
        
        // Remove existing tap if any
        if audioTap != nil {
            mixerNode.removeTap(onBus: 0)
        }
        
        audioTap = UUID()
        
        mixerNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer)
        }
    }
    
    // MARK: - Public Methods
    
    func loadAudio(from url: URL) async throws {
        stop()
        
        // Load audio file
        audioFile = try AVAudioFile(forReading: url)
        guard let audioFile = audioFile else {
            throw AudioEngineError.fileLoadFailed
        }
        
        audioFormat = audioFile.processingFormat
        duration = Double(audioFile.length) / audioFile.processingFormat.sampleRate
        
        // Load entire file into buffer for scrubbing support
        let frameCount = AVAudioFrameCount(audioFile.length)
        audioBuffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameCount)
        
        guard let buffer = audioBuffer else {
            throw AudioEngineError.bufferCreationFailed
        }
        
        try audioFile.read(into: buffer)
        
        // Extract waveform asynchronously
        Task {
            self.waveformData = await extractWaveform(resolution: 1000)
        }
        
        // Detect silence regions
        Task {
            self.silenceSegments = await detectSilence(threshold: -40, minDuration: 0.5)
        }
    }
    
    func play() {
        guard !isPlaying else { return }
        
        do {
            if !engine.isRunning {
                try engine.start()
            }
            
            if let buffer = audioBuffer {
                // Schedule buffer
                playerNode.scheduleBuffer(buffer, at: nil, options: .loops)
            }
            
            playerNode.play()
            isPlaying = true
            startTimeUpdates()
            
        } catch {
            print("Failed to start playback: \(error)")
        }
    }
    
    func pause() {
        playerNode.pause()
        isPlaying = false
        stopTimeUpdates()
    }
    
    func stop() {
        playerNode.stop()
        isPlaying = false
        currentTime = 0
        stopTimeUpdates()
    }
    
    func seek(to time: TimeInterval) {
        guard let audioFile = audioFile else { return }
        
        let wasPlaying = isPlaying
        
        if wasPlaying {
            playerNode.stop()
        }
        
        // Calculate frame position
        let sampleRate = audioFile.processingFormat.sampleRate
        let framePosition = AVAudioFramePosition(time * sampleRate)
        
        // Clamp to valid range
        let clampedPosition = max(0, min(framePosition, audioFile.length))
        
        currentTime = Double(clampedPosition) / sampleRate
        
        if wasPlaying {
            // Restart playback from new position
            playerNode.stop()
            
            if let buffer = audioBuffer {
                let remainingFrames = AVAudioFrameCount(audioFile.length - clampedPosition)
                if remainingFrames > 0 {
                    // Create new buffer from seek position
                    if let seekBuffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: remainingFrames) {
                        seekBuffer.frameLength = remainingFrames
                        
                        // Copy audio data starting from seek position
                        if let sourceBuffer = audioBuffer {
                            for channel in 0..<Int(audioFile.processingFormat.channelCount) {
                                let sourceData = sourceBuffer.floatChannelData![channel]
                                let destData = seekBuffer.floatChannelData![channel]
                                
                                // Copy from seek position to end
                                for frame in 0..<Int(remainingFrames) {
                                    destData[frame] = sourceData[Int(clampedPosition) + frame]
                                }
                            }
                        }
                        
                        playerNode.scheduleBuffer(seekBuffer, at: nil)
                        playerNode.play()
                    }
                }
            }
        }
    }
    
    func scrub(at rate: Float) {
        if !isScrubbing {
            isScrubbing = true
            lastScrubbingPosition = AVAudioFramePosition(currentTime * (audioFile?.processingFormat.sampleRate ?? 44100))
        }
        
        // Stop regular playback
        if isPlaying {
            pause()
        }
        
        // Start scrubbing timer
        scrubbingTimer?.invalidate()
        scrubbingTimer = Timer.scheduledTimer(withTimeInterval: 0.02, repeats: true) { [weak self] _ in
            self?.updateScrubbingPosition(rate: rate)
        }
    }
    
    func stopScrubbing() {
        isScrubbing = false
        scrubbingTimer?.invalidate()
        scrubbingTimer = nil
    }
    
    // MARK: - Waveform Extraction
    
    func extractWaveform(resolution: Int) async -> [Float] {
        guard let audioFile = audioFile,
              let buffer = audioBuffer else { return [] }
        
        let channelData = buffer.floatChannelData![0]
        let frameLength = Int(buffer.frameLength)
        let samplesPerPixel = max(1, frameLength / resolution)
        
        var waveform: [Float] = []
        waveform.reserveCapacity(resolution)
        
        await withTaskGroup(of: (Int, Float).self) { group in
            for i in 0..<resolution {
                group.addTask {
                    let startSample = i * samplesPerPixel
                    let endSample = min(startSample + samplesPerPixel, frameLength)
                    
                    var maxValue: Float = 0
                    for sample in startSample..<endSample {
                        maxValue = max(maxValue, abs(channelData[sample]))
                    }
                    
                    return (i, maxValue)
                }
            }
            
            // Collect results in order
            var results = Array(repeating: Float(0), count: resolution)
            for await (index, value) in group {
                results[index] = value
            }
            waveform = results
        }
        
        // Normalize waveform
        if let maxValue = waveform.max(), maxValue > 0 {
            waveform = waveform.map { $0 / maxValue }
        }
        
        return waveform
    }
    
    // MARK: - Silence Detection
    
    func detectSilence(threshold: Float, minDuration: TimeInterval) async -> [AudioTimeRange] {
        guard let buffer = audioBuffer,
              let format = audioFormat else { return [] }
        
        let channelData = buffer.floatChannelData![0]
        let frameLength = Int(buffer.frameLength)
        let sampleRate = format.sampleRate
        
        // Convert threshold from dB to linear
        let linearThreshold = pow(10, threshold / 20)
        
        var silenceSegments: [AudioTimeRange] = []
        var silenceStart: Int?
        
        // Window size for RMS calculation (10ms)
        let windowSize = Int(sampleRate * 0.01)
        let minSilenceSamples = Int(sampleRate * minDuration)
        
        for i in stride(from: 0, to: frameLength - windowSize, by: windowSize / 2) {
            // Calculate RMS for window
            var rms: Float = 0
            vDSP_rmsqv(channelData.advanced(by: i), 1, &rms, vDSP_Length(windowSize))
            
            if rms < linearThreshold {
                // Below threshold - potential silence
                if silenceStart == nil {
                    silenceStart = i
                }
            } else {
                // Above threshold - end of silence
                if let start = silenceStart {
                    let silenceDuration = i - start
                    if silenceDuration >= minSilenceSamples {
                        let startTime = Double(start) / sampleRate
                        let endTime = Double(i) / sampleRate
                        silenceSegments.append(AudioTimeRange(start: startTime, end: endTime))
                    }
                    silenceStart = nil
                }
            }
        }
        
        // Check for silence at end
        if let start = silenceStart {
            let silenceDuration = frameLength - start
            if silenceDuration >= minSilenceSamples {
                let startTime = Double(start) / sampleRate
                let endTime = Double(frameLength) / sampleRate
                silenceSegments.append(AudioTimeRange(start: startTime, end: endTime))
            }
        }
        
        return silenceSegments
    }
    
    // MARK: - Audio Analysis
    
    func analyzeFrequencySpectrum(at time: TimeInterval) -> [Float] {
        guard let buffer = audioBuffer,
              let format = audioFormat else { return [] }
        
        let sampleRate = format.sampleRate
        let framePosition = Int(time * sampleRate)
        
        guard framePosition + fftLength < buffer.frameLength else { return [] }
        
        let channelData = buffer.floatChannelData![0]
        
        // Prepare FFT buffers
        var realp = [Float](repeating: 0, count: fftLength / 2)
        var imagp = [Float](repeating: 0, count: fftLength / 2)
        var fftBuffer = DSPSplitComplex(realp: &realp, imagp: &imagp)
        
        // Copy and window audio data
        var windowedData = [Float](repeating: 0, count: fftLength)
        for i in 0..<fftLength {
            let sample = channelData[framePosition + i]
            // Apply Hanning window
            let window = 0.5 - 0.5 * cos(2 * .pi * Double(i) / Float(fftLength - 1))
            windowedData[i] = sample * window
        }
        
        // Convert to split complex format
        windowedData.withUnsafeBufferPointer { ptr in
            ptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftLength / 2) { complexPtr in
                vDSP_ctoz(complexPtr, 2, &fftBuffer, 1, vDSP_Length(fftLength / 2))
            }
        }
        
        // Perform FFT
        vDSP_fft_zrip(fftSetup, &fftBuffer, 1, vDSP_Length(log2(Float(fftLength))), FFTDirection(kFFTDirection_Forward))
        
        // Calculate magnitude
        var magnitudes = [Float](repeating: 0, count: fftLength / 2)
        vDSP_zvmags(&fftBuffer, 1, &magnitudes, 1, vDSP_Length(fftLength / 2))
        
        // Convert to dB
        var dbMagnitudes = [Float](repeating: 0, count: fftLength / 2)
        var reference: Float = 1.0
        vDSP_vdbcon(&magnitudes, 1, &reference, &dbMagnitudes, 1, vDSP_Length(fftLength / 2), 0)
        
        return dbMagnitudes
    }
    
    // MARK: - Private Methods
    
    private func updateVolume() {
        mixerNode.volume = volume
    }
    
    private func updateRate() {
        timeEffect.rate = rate
        timeEffect.pitch = 0 // Keep pitch constant when changing rate
    }
    
    private func startTimeUpdates() {
        Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] timer in
            guard let self = self else {
                timer.invalidate()
                return
            }
            
            if self.isPlaying {
                self.updateCurrentTime()
            } else {
                timer.invalidate()
            }
        }
    }
    
    private func stopTimeUpdates() {
        // Timer will auto-invalidate when isPlaying becomes false
    }
    
    private func updateCurrentTime() {
        guard let nodeTime = playerNode.lastRenderTime,
              let playerTime = playerNode.playerTime(forNodeTime: nodeTime),
              let format = audioFormat else { return }
        
        currentTime = Double(playerTime.sampleTime) / format.sampleRate
    }
    
    private func updateScrubbingPosition(rate: Float) {
        guard let format = audioFormat,
              let audioFile = audioFile else { return }
        
        let sampleRate = format.sampleRate
        let samplesPerUpdate = Int(sampleRate * 0.02 * Double(abs(rate)))
        
        if rate > 0 {
            lastScrubbingPosition += AVAudioFramePosition(samplesPerUpdate)
        } else {
            lastScrubbingPosition -= AVAudioFramePosition(samplesPerUpdate)
        }
        
        // Clamp position
        lastScrubbingPosition = max(0, min(lastScrubbingPosition, audioFile.length))
        
        currentTime = Double(lastScrubbingPosition) / sampleRate
        
        // Play small chunk for audio feedback
        playScrubbingChunk(at: lastScrubbingPosition)
    }
    
    private func playScrubbingChunk(at position: AVAudioFramePosition) {
        guard let buffer = audioBuffer,
              let format = audioFormat else { return }
        
        let chunkSize = AVAudioFrameCount(format.sampleRate * 0.05) // 50ms chunk
        let startFrame = max(0, position)
        let endFrame = min(startFrame + AVAudioFramePosition(chunkSize), AVAudioFramePosition(buffer.frameLength))
        let actualChunkSize = AVAudioFrameCount(endFrame - startFrame)
        
        guard actualChunkSize > 0 else { return }
        
        if let chunkBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: actualChunkSize) {
            chunkBuffer.frameLength = actualChunkSize
            
            // Copy chunk data
            for channel in 0..<Int(format.channelCount) {
                let sourceData = buffer.floatChannelData![channel]
                let destData = chunkBuffer.floatChannelData![channel]
                
                for i in 0..<Int(actualChunkSize) {
                    destData[i] = sourceData[Int(startFrame) + i]
                }
            }
            
            playerNode.stop()
            playerNode.scheduleBuffer(chunkBuffer, at: nil)
            playerNode.play()
        }
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        let channelData = buffer.floatChannelData!
        let channelCount = Int(buffer.format.channelCount)
        let frameLength = Int(buffer.frameLength)
        
        guard frameLength > 0 else { return }
        
        // Calculate RMS levels for each channel
        var leftRMS: Float = 0
        var rightRMS: Float = 0
        var leftPeak: Float = 0
        var rightPeak: Float = 0
        
        if channelCount >= 1 {
            vDSP_rmsqv(channelData[0], 1, &leftRMS, vDSP_Length(frameLength))
            vDSP_maxv(channelData[0], 1, &leftPeak, vDSP_Length(frameLength))
        }
        
        if channelCount >= 2 {
            vDSP_rmsqv(channelData[1], 1, &rightRMS, vDSP_Length(frameLength))
            vDSP_maxv(channelData[1], 1, &rightPeak, vDSP_Length(frameLength))
        } else {
            rightRMS = leftRMS
            rightPeak = leftPeak
        }
        
        // Update audio levels (convert to dB)
        Task { @MainActor in
            self.audioLevels = AudioLevels(
                leftChannel: 20 * log10(max(0.00001, leftRMS)),
                rightChannel: 20 * log10(max(0.00001, rightRMS)),
                leftPeak: 20 * log10(max(0.00001, abs(leftPeak))),
                rightPeak: 20 * log10(max(0.00001, abs(rightPeak)))
            )
        }
    }
}

// MARK: - Audio Engine Error
enum AudioEngineError: LocalizedError {
    case fileLoadFailed
    case bufferCreationFailed
    case engineStartFailed
    case formatMismatch
    
    var errorDescription: String? {
        switch self {
        case .fileLoadFailed:
            return "Failed to load audio file"
        case .bufferCreationFailed:
            return "Failed to create audio buffer"
        case .engineStartFailed:
            return "Failed to start audio engine"
        case .formatMismatch:
            return "Audio format mismatch"
        }
    }
}

// MARK: - Audio Scrubbing Controller
@MainActor
final class AudioScrubbingController: ObservableObject {
    @Published var isScrubbing = false
    @Published var scrubbingRate: Float = 1.0
    @Published var scrubbingPosition: TimeInterval = 0
    
    private let audioEngine: AudioEngine
    private var scrubbingGesture: NSPanGestureRecognizer?
    
    init(audioEngine: AudioEngine) {
        self.audioEngine = audioEngine
    }
    
    func startScrubbing(at position: TimeInterval) {
        isScrubbing = true
        scrubbingPosition = position
        audioEngine.seek(to: position)
    }
    
    func updateScrubbing(deltaX: CGFloat, viewWidth: CGFloat) {
        guard isScrubbing else { return }
        
        // Calculate scrubbing rate based on gesture velocity
        let normalizedVelocity = deltaX / viewWidth
        scrubbingRate = Float(normalizedVelocity * 4) // 4x max scrubbing speed
        
        // Update position
        let timeDelta = Double(scrubbingRate) * 0.016 // 60fps update rate
        scrubbingPosition += timeDelta
        scrubbingPosition = max(0, min(scrubbingPosition, audioEngine.duration))
        
        // Scrub audio
        audioEngine.scrub(at: scrubbingRate)
    }
    
    func endScrubbing() {
        isScrubbing = false
        audioEngine.stopScrubbing()
        audioEngine.seek(to: scrubbingPosition)
    }
}
