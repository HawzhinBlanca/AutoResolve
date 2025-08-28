import Foundation
import SwiftUI
import Combine
import AVFoundation
import Accelerate
import OSLog

/// Advanced audio analysis system for speech recognition, music analysis, and acoustic feature extraction
/// Provides comprehensive audio content understanding for intelligent editing
@MainActor
public class AudioAnalyzer: ObservableObject {
    private let logger = Logger.shared
    
    // Audio processing components
    private let audioEngine = AVAudioEngine()
    private let speechRecognizer = SpeechRecognizer()
    private var musicAnalyzer: MusicAnalyzer!
    private var acousticAnalyzer: AcousticAnalyzer!
    
    // Analysis state
    @Published public var isAnalyzing = false
    @Published public var analysisProgress = 0.0
    @Published public var currentOperation = ""
    
    // Results cache
    private var analysisCache: [String: AudioAnalysis] = [:]
    
    // Audio configuration
    private let sampleRate: Double = 44100.0
    private let bufferSize: AVAudioFrameCount = 4096
    private let analysisWindowSize: Double = 0.5 // 500ms windows
    
    public init() {
        setupAudioEngine()
        Task {
            await initializeAnalyzers()
        }
    }
    
    private func initializeAnalyzers() async {
        musicAnalyzer = await MusicAnalyzer()
        acousticAnalyzer = await AcousticAnalyzer()
    }
    
    // MARK: - Public API
    
    public func analyzeAudio(_ audioURL: URL) async throws -> AudioAnalysis {
        let cacheKey = audioURL.absoluteString
        if let cached = analysisCache[cacheKey] {
            return cached
        }
        
        logger.info("Starting audio analysis for: \(audioURL.lastPathComponent)")
        
        isAnalyzing = true
        analysisProgress = 0.0
        currentOperation = "Initializing audio analysis..."
        
        defer {
            isAnalyzing = false
            analysisProgress = 1.0
            currentOperation = ""
        }
        
        do {
            let asset = AVAsset(url: audioURL)
            let duration = try await asset.load(.duration)
            
            // Extract audio data
            currentOperation = "Extracting audio data..."
            analysisProgress = 0.1
            
            let audioData = try await extractAudioData(from: asset)
            
            // Perform speech recognition
            currentOperation = "Performing speech recognition..."
            analysisProgress = 0.3
            
            let speechAnalysis = try await speechRecognizer.analyze(audioData: audioData, duration: duration)
            
            // Analyze music content
            currentOperation = "Analyzing music content..."
            analysisProgress = 0.5
            
            let musicAnalysis = try await musicAnalyzer.analyze(audioData: audioData, duration: duration)
            
            // Extract acoustic features
            currentOperation = "Extracting acoustic features..."
            analysisProgress = 0.7
            
            let acousticFeatures = try await acousticAnalyzer.analyze(audioData: audioData, duration: duration)
            
            // Detect silence regions
            currentOperation = "Detecting silence regions..."
            analysisProgress = 0.85
            
            let silenceSegments = detectSilenceRegions(audioData: audioData, duration: duration)
            
            // Calculate energy levels
            currentOperation = "Calculating energy levels..."
            analysisProgress = 0.95
            
            let energyLevels = calculateEnergyLevels(audioData: audioData, duration: duration)
            
            let audioAnalysis = AudioAnalysis(
                speechRecognition: speechAnalysis,
                musicAnalysis: musicAnalysis,
                acousticFeatures: acousticFeatures,
                silenceSegments: silenceSegments,
                energyLevels: energyLevels,
                peakLocations: findPeakLocations(energyLevels),
                spectralAnalysis: performSpectralAnalysis(audioData: audioData),
                rhythmAnalysis: analyzeRhythm(audioData: audioData, duration: duration),
                emotionalTone: analyzeEmotionalTone(speechAnalysis, musicAnalysis),
                audioQuality: assessAudioQuality(audioData: audioData)
            )
            
            // Cache results
            analysisCache[cacheKey] = audioAnalysis
            
            logger.info("Audio analysis completed successfully")
            return audioAnalysis
            
        } catch {
            logger.error("Audio analysis failed: \(error)")
            throw error
        }
    }
    
    public func extractAudioFeatures(_ audioBuffer: AVAudioPCMBuffer) async throws -> AcousticFeatures {
        guard let floatChannelData = audioBuffer.floatChannelData else {
            throw AudioAnalysisError.invalidAudioBuffer
        }
        
        let frameLength = Int(audioBuffer.frameLength)
        let audioData = Array(UnsafeBufferPointer(start: floatChannelData[0], count: frameLength))
        
        return try await acousticAnalyzer.extractFeatures(from: audioData, sampleRate: sampleRate)
    }
    
    // MARK: - Private Implementation
    
    private func setupAudioEngine() {
        #if os(iOS)
        do {
            try AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .measurement, options: [])
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            logger.error("Failed to setup audio session: \(error)")
        }
        #else
        // AVAudioSession unavailable on macOS; nothing to configure here
        #endif
    }
    
    private func extractAudioData(from asset: AVAsset) async throws -> [Float] {
        guard let audioTrack = try await asset.loadTracks(withMediaType: .audio).first else {
            throw AudioAnalysisError.noAudioTrack
        }
        
        let reader = try AVAssetReader(asset: asset)
        
        let outputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
            AVSampleRateKey: sampleRate
        ]
        
        let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: outputSettings)
        reader.add(output)
        reader.startReading()
        
        var audioData: [Float] = []
        
        while reader.status == .reading {
            if let sampleBuffer = output.copyNextSampleBuffer() {
                if let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) {
                    let length = CMBlockBufferGetDataLength(blockBuffer)
                    let floatCount = length / MemoryLayout<Float>.size
                    
                    var audioBuffer = [Float](repeating: 0, count: floatCount)
                    CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: length, destination: &audioBuffer)
                    
                    audioData.append(contentsOf: audioBuffer)
                }
                CMSampleBufferInvalidate(sampleBuffer)
            } else {
                break
            }
        }
        
        return audioData
    }
    
    private func detectSilenceRegions(audioData: [Float], duration: CMTime) -> [SilenceRegion] {
        let silenceThreshold: Float = 0.01 // Adjustable threshold
        let minSilenceDuration: Double = 0.3 // Minimum 300ms for silence
        
        let samplesPerSecond = Int(sampleRate)
        let minSilenceSamples = Int(minSilenceDuration * sampleRate)
        
        var silenceSegments: [SilenceRegion] = []
        var currentSilenceStart: Int? = nil
        var consecutiveSilentSamples = 0
        
        for (index, sample) in audioData.enumerated() {
            let isQuiet = abs(sample) < silenceThreshold
            
            if isQuiet {
                if currentSilenceStart == nil {
                    currentSilenceStart = index
                }
                consecutiveSilentSamples += 1
            } else {
                if let silenceStart = currentSilenceStart, consecutiveSilentSamples >= minSilenceSamples {
                    let startTime = Double(silenceStart) / sampleRate
                    let endTime = Double(index) / sampleRate
                    let confidence = calculateSilenceConfidence(audioData, start: silenceStart, end: index)
                    
                    silenceSegments.append(SilenceRegion(
                        startTime: startTime,
                        endTime: endTime,
                        duration: endTime - startTime,
                        confidence: confidence
                    ))
                }
                currentSilenceStart = nil
                consecutiveSilentSamples = 0
            }
        }
        
        // Handle silence at the end
        if let silenceStart = currentSilenceStart, consecutiveSilentSamples >= minSilenceSamples {
            let startTime = Double(silenceStart) / sampleRate
            let endTime = duration.seconds
            let confidence = calculateSilenceConfidence(audioData, start: silenceStart, end: audioData.count)
            
            silenceSegments.append(SilenceRegion(
                startTime: startTime,
                endTime: endTime,
                duration: endTime - startTime,
                confidence: confidence
            ))
        }
        
        return silenceSegments
    }
    
    private func calculateSilenceConfidence(_ audioData: [Float], start: Int, end: Int) -> Double {
        let silenceThreshold: Float = 0.01
        let segment = Array(audioData[start..<min(end, audioData.count)])
        let quietSamples = segment.filter { abs($0) < silenceThreshold }.count
        return Double(quietSamples) / Double(segment.count)
    }
    
    private func calculateEnergyLevels(audioData: [Float], duration: CMTime) -> [EnergyLevel] {
        let windowSize = Int(analysisWindowSize * sampleRate)
        let hopSize = windowSize / 2
        var energyLevels: [EnergyLevel] = []
        
        var index = 0
        while index + windowSize < audioData.count {
            let window = Array(audioData[index..<index + windowSize])
            let rms = sqrt(window.reduce(0) { $0 + $1 * $1 } / Float(windowSize))
            let peak = window.max() ?? 0.0
            let timestamp = Double(index) / sampleRate
            
            energyLevels.append(EnergyLevel(
                timestamp: timestamp,
                rms: Double(rms),
                spectralCentroid: calculateSpectralCentroid(window)
            ))
            
            index += hopSize
        }
        
        return energyLevels
    }
    
    private func calculateSpectralCentroid(_ audioData: [Float]) -> Double {
        let fft = performFFT(audioData)
        var weightedSum = 0.0
        var magnitudeSum = 0.0
        
        for (index, magnitude) in fft.enumerated() {
            let frequency = Double(index) * sampleRate / Double(fft.count)
            weightedSum += frequency * magnitude
            magnitudeSum += magnitude
        }
        
        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0.0
    }
    
    private func calculateZeroCrossingRate(_ audioData: [Float]) -> Double {
        var crossings = 0
        for i in 1..<audioData.count {
            if (audioData[i] >= 0) != (audioData[i-1] >= 0) {
                crossings += 1
            }
        }
        return Double(crossings) / Double(audioData.count - 1)
    }
    
    private func performFFT(_ audioData: [Float]) -> [Double] {
        let length = audioData.count
        let log2n = vDSP_Length(log2(Float(length)))
        let fftSetup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(length), .forward)
        
        var realPart = audioData
        var imaginaryPart = [Float](repeating: 0.0, count: length)
        
        realPart.withUnsafeMutableBufferPointer { realPtr in
            imaginaryPart.withUnsafeMutableBufferPointer { imagPtr in
                vDSP_DFT_Execute(fftSetup!, realPtr.baseAddress!, imagPtr.baseAddress!, realPtr.baseAddress!, imagPtr.baseAddress!)
            }
        }
        
        vDSP_DFT_DestroySetup(fftSetup)
        
        var magnitudes = [Double](repeating: 0.0, count: length / 2)
        for i in 0..<length / 2 {
            magnitudes[i] = sqrt(Double(realPart[i] * realPart[i] + imaginaryPart[i] * imaginaryPart[i]))
        }
        
        return magnitudes
    }
    
    private func findPeakLocations(_ energyLevels: [EnergyLevel]) -> [RhythmBeatData] {
        var peaks: [RhythmBeatData] = []
        let windowSize = 5 // Look for peaks in a 5-sample window
        
        for i in windowSize..<(energyLevels.count - windowSize) {
            let current = energyLevels[i]
            var isPeak = true
            
            // Check if current sample is higher than all neighbors
            for j in (i - windowSize)..<(i + windowSize) {
                if j != i && energyLevels[j].rms >= current.rms {
                    isPeak = false
                    break
                }
            }
            
            if isPeak && current.rms > 0.1 {
                peaks.append(RhythmBeatData(timestamp: current.timestamp, confidence: min(1.0, current.rms)))
            }
        }
        
        return peaks
    }
    
    private func classifyPeakType(_ peak: EnergyLevel, surrounding: [EnergyLevel]) -> PeakType {
        let avgRMS = surrounding.reduce(0.0) { $0 + $1.rms } / Double(surrounding.count)
        let peakRatio = peak.rms / avgRMS
        
        if peak.spectralCentroid > 2000 && peakRatio > 3.0 {
            return .transient
        } else if peak.spectralCentroid < 500 && peakRatio > 2.0 {
            return .bass
        } else if peakRatio > 2.5 {
            return .emphasis
        } else {
            return .normal
        }
    }
    
    private func performSpectralAnalysis(audioData: [Float]) -> SpectralAnalysis {
        let windowSize = 2048
        let hopSize = windowSize / 2
        var spectrograms: [SpectrogramFrame] = []
        
        var index = 0
        while index + windowSize < audioData.count {
            let window = Array(audioData[index..<index + windowSize])
            let fft = performFFT(window)
            let timestamp = Double(index) / sampleRate
            
            spectrograms.append(SpectrogramFrame(
                frequencies: fft,
                magnitudes: fft,
                timestamp: timestamp
            ))
            
            index += hopSize
        }
        
        return SpectralAnalysis(
            spectrograms: spectrograms,
            frequencyRange: (min: 20.0, max: sampleRate / 2.0),
            dominantFrequencies: findDominantFrequencies(spectrograms),
            spectralFeatures: aggregateSpectralFeatures(spectrograms)
        )
    }
    
    private func findFundamentalFrequency(_ fft: [Double]) -> Double {
        // Find the frequency with the highest magnitude (simplified approach)
        if let maxIndex = fft.enumerated().max(by: { $0.element < $1.element })?.offset {
            return Double(maxIndex) * sampleRate / Double(fft.count * 2)
        }
        return 0.0
    }
    
    private func findHarmonics(_ fft: [Double]) -> [Double] {
        let fundamental = findFundamentalFrequency(fft)
        var harmonics: [Double] = []
        
        // Look for harmonics (multiples of fundamental frequency)
        for multiplier in 2...10 {
            let harmonicFreq = fundamental * Double(multiplier)
            if harmonicFreq < sampleRate / 2 {
                harmonics.append(harmonicFreq)
            }
        }
        
        return harmonics
    }
    
    private func calculateSpectralRolloff(_ fft: [Double]) -> Double {
        let totalEnergy = fft.reduce(0, +)
        let threshold = 0.85 * totalEnergy
        var cumulativeEnergy = 0.0
        
        for (index, magnitude) in fft.enumerated() {
            cumulativeEnergy += magnitude
            if cumulativeEnergy >= threshold {
                return Double(index) * sampleRate / Double(fft.count * 2)
            }
        }
        
        return sampleRate / 2.0
    }
    
    private func calculateMFCC(_ fft: [Double]) -> [Double] {
        // Simplified MFCC calculation
        // In production, would use a proper MFCC implementation
        let numCoefficients = 13
        let melFilters = createMelFilterBank(fftSize: fft.count, sampleRate: sampleRate, numFilters: 26)
        
        var mfcc = [Double](repeating: 0.0, count: numCoefficients)
        
        // Apply mel filter bank
        for (filterIndex, filter) in melFilters.enumerated() {
            var filterOutput = 0.0
            for (freqIndex, magnitude) in fft.enumerated() {
                if freqIndex < filter.count {
                    filterOutput += magnitude * filter[freqIndex]
                }
            }
            
            // DCT to get MFCC coefficients
            for coeffIndex in 0..<numCoefficients {
                if coeffIndex < mfcc.count && filterIndex < melFilters.count {
                    mfcc[coeffIndex] += log(max(filterOutput, 1e-10)) * cos(Double(coeffIndex) * (Double(filterIndex) + 0.5) * Double.pi / Double(melFilters.count))
                }
            }
        }
        
        return mfcc
    }
    
    private func createMelFilterBank(fftSize: Int, sampleRate: Double, numFilters: Int) -> [[Double]] {
        // Create mel filter bank (simplified implementation)
        var filters: [[Double]] = []
        
        for filterIndex in 0..<numFilters {
            var filter = [Double](repeating: 0.0, count: fftSize / 2)
            
            // Triangular mel filter
            let centerFreq = Double(filterIndex + 1) * (sampleRate / 2.0) / Double(numFilters + 1)
            let centerBin = Int(centerFreq * Double(fftSize) / sampleRate)
            
            let leftBin = max(0, centerBin - fftSize / (numFilters * 4))
            let rightBin = min(fftSize / 2 - 1, centerBin + fftSize / (numFilters * 4))
            
            // Create triangular response
            for bin in leftBin...rightBin {
                if bin <= centerBin {
                    filter[bin] = Double(bin - leftBin) / Double(centerBin - leftBin)
                } else {
                    filter[bin] = Double(rightBin - bin) / Double(rightBin - centerBin)
                }
            }
            
            filters.append(filter)
        }
        
        return filters
    }
    
    private func findDominantFrequencies(_ spectrograms: [SpectrogramFrame]) -> [Double] {
        var frequencyHistogram: [Int: Double] = [:]
        
        for frame in spectrograms {
            for (index, magnitude) in frame.frequencies.enumerated() {
                let frequency = Int(Double(index) * sampleRate / Double(frame.frequencies.count * 2))
                frequencyHistogram[frequency, default: 0.0] += magnitude
            }
        }
        
        return frequencyHistogram.sorted(by: { $0.value > $1.value }).prefix(10).map { Double($0.key) }
    }
    
    private func aggregateSpectralFeatures(_ spectrograms: [SpectrogramFrame]) -> SpectralFeatures {
        let avgCentroid = spectrograms.reduce(0.0) { $0 + $1.fundamentalFrequency } / Double(spectrograms.count)
        let avgRolloff = spectrograms.reduce(0.0) { $0 + $1.spectralRolloff } / Double(spectrograms.count)
        
        let avgMFCC = spectrograms.first?.mfcc.indices.map { index in
            spectrograms.reduce(0.0) { $0 + $1.mfcc[index] } / Double(spectrograms.count)
        } ?? []
        
        return SpectralFeatures(
            spectralCentroid: avgCentroid,
            spectralRolloff: avgRolloff,
            spectralFlux: calculateSpectralFlux(spectrograms),
            mfcc: avgMFCC,
            chroma: calculateChromaFeatures(spectrograms),
            tonnetz: calculateTonnetz(spectrograms)
        )
    }
    
    private func calculateSpectralFlux(_ spectrograms: [SpectrogramFrame]) -> Double {
        guard spectrograms.count > 1 else { return 0.0 }
        
        var totalFlux = 0.0
        for i in 1..<spectrograms.count {
            let prev = spectrograms[i-1].frequencies
            let curr = spectrograms[i].frequencies
            
            var flux = 0.0
            for j in 0..<min(prev.count, curr.count) {
                flux += abs(curr[j] - prev[j])
            }
            totalFlux += flux
        }
        
        return totalFlux / Double(spectrograms.count - 1)
    }
    
    private func calculateChromaFeatures(_ spectrograms: [SpectrogramFrame]) -> [Double] {
        // Simplified chroma feature calculation
        // Maps frequencies to 12 pitch classes
        var chroma = [Double](repeating: 0.0, count: 12)
        
        for frame in spectrograms {
            for (index, magnitude) in frame.frequencies.enumerated() {
                let frequency = Double(index) * sampleRate / Double(frame.frequencies.count * 2)
                if frequency > 0 {
                    let pitchClass = Int(log2(frequency / 440.0) * 12 + 69) % 12
                    chroma[pitchClass] += magnitude
                }
            }
        }
        
        // Normalize
        let total = chroma.reduce(0, +)
        return total > 0 ? chroma.map { $0 / total } : chroma
    }
    
    private func calculateTonnetz(_ spectrograms: [SpectrogramFrame]) -> [Double] {
        // Simplified Tonnetz calculation
        // In production, would implement proper harmonic network analysis
        return [Double](repeating: 0.5, count: 6)
    }
    
    private func analyzeRhythm(audioData: [Float], duration: CMTime) -> RhythmAnalysis {
        let energyLevels = calculateEnergyLevels(audioData: audioData, duration: duration)
        let tempo = estimateTempo(energyLevels)
        let beats = detectBeats(energyLevels)
        let meter = estimateMeter(beats)
        
        return RhythmAnalysis(
            tempo: tempo,
            beats: beats,
            timeSignature: meter,
            rhythmComplexity: calculateRhythmComplexity(beats),
            syncopation: calculateSyncopation(beats),
            groove: analyzeGroove(beats)
        )
    }
    
    private func estimateTempo(_ energyLevels: [EnergyLevel]) -> Double {
        // Autocorrelation-based tempo estimation
        let maxLag = Int(3.0 / analysisWindowSize) // Up to 3 seconds
        var autocorrelation = [Double](repeating: 0.0, count: maxLag)
        
        for lag in 1..<maxLag {
            var sum = 0.0
            var count = 0
            
            for i in lag..<energyLevels.count {
                sum += energyLevels[i].rms * energyLevels[i - lag].rms
                count += 1
            }
            
            autocorrelation[lag] = count > 0 ? sum / Double(count) : 0.0
        }
        
        // Find peak in autocorrelation (corresponds to beat period)
        if let maxIndex = autocorrelation.enumerated().max(by: { $0.element < $1.element })?.offset {
            let beatPeriod = Double(maxIndex) * analysisWindowSize
            return 60.0 / beatPeriod // Convert to BPM
        }
        
        return 120.0 // Default tempo
    }
    
    private func detectBeats(_ energyLevels: [EnergyLevel]) -> [BeatData] {
        var beats: [BeatData] = []
        let threshold = energyLevels.reduce(0.0) { $0 + $1.rms } / Double(energyLevels.count) * 1.5
        
        var lastBeatTime = 0.0
        let minBeatInterval = 0.3 // Minimum 300ms between beats
        
        for energy in energyLevels {
            if energy.rms > threshold && energy.timestamp - lastBeatTime > minBeatInterval {
                beats.append(Beat(
                    timestamp: energy.timestamp,
                    strength: energy.rms / threshold,
                    confidence: calculateBeatConfidence(energy, surrounding: energyLevels)
                ))
                lastBeatTime = energy.timestamp
            }
        }
        
        return beats
    }
    
    private func calculateBeatConfidence(_ beat: EnergyLevel, surrounding: [EnergyLevel]) -> Double {
        // Calculate confidence based on how much the beat stands out from surrounding energy
        let windowSize = 10
        let beatIndex = surrounding.firstIndex(where: { $0.timestamp == beat.timestamp }) ?? 0
        
        let startIndex = max(0, beatIndex - windowSize)
        let endIndex = min(surrounding.count, beatIndex + windowSize)
        
        let localAverage = surrounding[startIndex..<endIndex].reduce(0.0) { $0 + $1.rms } / Double(endIndex - startIndex)
        
        return min(1.0, beat.rms / (localAverage + 0.001)) // Avoid division by zero
    }
    
    private func estimateMeter(_ beats: [BeatData]) -> TimeSignature {
        // Simple meter estimation based on beat intervals
        guard beats.count > 4 else { return TimeSignature(numerator: 4, denominator: 4) }
        
        var intervals: [Double] = []
        for i in 1..<beats.count {
            intervals.append(beats[i].timestamp - beats[i-1].timestamp)
        }
        
        // Cluster intervals to find common beat patterns
        let avgInterval = intervals.reduce(0, +) / Double(intervals.count)
        
        // Simple heuristic: if most intervals are similar, likely 4/4 time
        let uniformIntervals = intervals.filter { abs($0 - avgInterval) < avgInterval * 0.2 }.count
        
        if Double(uniformIntervals) / Double(intervals.count) > 0.7 {
            return TimeSignature(numerator: 4, denominator: 4)
        } else {
            return TimeSignature(numerator: 3, denominator: 4) // Default to waltz time
        }
    }
    
    private func calculateRhythmComplexity(_ beats: [BeatData]) -> Double {
        guard beats.count > 2 else { return 0.0 }
        
        var intervals: [Double] = []
        for i in 1..<beats.count {
            intervals.append(beats[i].timestamp - beats[i-1].timestamp)
        }
        
        // Calculate coefficient of variation
        let mean = intervals.reduce(0, +) / Double(intervals.count)
        let variance = intervals.reduce(0) { $0 + pow($1 - mean, 2) } / Double(intervals.count)
        let standardDeviation = sqrt(variance)
        
        return mean > 0 ? standardDeviation / mean : 0.0
    }
    
    private func calculateSyncopation(_ beats: [BeatData]) -> Double {
        // Simplified syncopation detection
        // In production, would implement more sophisticated analysis
        return 0.3 // Placeholder
    }
    
    private func analyzeGroove(_ beats: [BeatData]) -> GrooveAnalysis {
        guard beats.count > 4 else {
            return GrooveAnalysis(
                swingRatio: 0.5,
                microtiming: [],
                grooveTemplate: "straight",
                humanization: 0.0
            )
        }
        
        // Analyze swing ratio
        var evenBeatIntervals: [Double] = []
        var oddBeatIntervals: [Double] = []
        
        for i in stride(from: 1, to: beats.count, by: 2) {
            if i + 1 < beats.count {
                evenBeatIntervals.append(beats[i].timestamp - beats[i-1].timestamp)
                oddBeatIntervals.append(beats[i+1].timestamp - beats[i].timestamp)
            }
        }
        
        let evenAvg = evenBeatIntervals.isEmpty ? 0.5 : evenBeatIntervals.reduce(0, +) / Double(evenBeatIntervals.count)
        let oddAvg = oddBeatIntervals.isEmpty ? 0.5 : oddBeatIntervals.reduce(0, +) / Double(oddBeatIntervals.count)
        
        let swingRatio = evenAvg / (evenAvg + oddAvg)
        
        return GrooveAnalysis(
            swingRatio: swingRatio,
            microtiming: calculateMicrotiming(beats),
            grooveTemplate: classifyGrooveTemplate(swingRatio),
            humanization: calculateHumanization(beats)
        )
    }
    
    private func calculateMicrotiming(_ beats: [BeatData]) -> [Double] {
        // Calculate timing deviations from a perfect grid
        guard beats.count > 2 else { return [] }
        
        let averageInterval = (beats.last!.timestamp - beats.first!.timestamp) / Double(beats.count - 1)
        var microtiming: [Double] = []
        
        for (index, beat) in beats.enumerated() {
            let expectedTime = beats.first!.timestamp + Double(index) * averageInterval
            let deviation = beat.timestamp - expectedTime
            microtiming.append(deviation)
        }
        
        return microtiming
    }
    
    private func classifyGrooveTemplate(_ swingRatio: Double) -> String {
        if swingRatio < 0.45 {
            return "heavy_swing"
        } else if swingRatio < 0.48 {
            return "moderate_swing"
        } else if swingRatio > 0.52 {
            return "reverse_swing"
        } else {
            return "straight"
        }
    }
    
    private func calculateHumanization(_ beats: [BeatData]) -> Double {
        guard beats.count > 2 else { return 0.0 }
        
        let microtiming = calculateMicrotiming(beats)
        let variance = microtiming.reduce(0) { $0 + $1 * $1 } / Double(microtiming.count)
        
        // Normalize humanization score (0.0 = perfectly quantized, 1.0 = very human)
        return min(1.0, sqrt(variance) * 100.0)
    }
    
    private func analyzeEmotionalTone(_ speechAnalysis: SpeechRecognitionResult, _ musicAnalysis: MusicAnalysisResult) -> EmotionalTone {
        // Combine speech and music analysis to determine emotional content
        var valence = 0.5 // Positive/negative emotion
        var arousal = 0.5 // Energy/intensity
        var dominance = 0.5 // Control/power
        
        // Adjust based on speech sentiment
        if let sentiment = speechAnalysis.overallSentiment {
            valence = sentiment.positiveScore
            arousal = sentiment.confidence
        }
        
        // Adjust based on music characteristics
        if musicAnalysis.tempo > 120 {
            arousal += 0.2
        }
        
        if musicAnalysis.key?.contains("major") == true {
            valence += 0.1
        } else if musicAnalysis.key?.contains("minor") == true {
            valence -= 0.1
        }
        
        // Clamp values
        valence = max(0.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        dominance = max(0.0, min(1.0, dominance))
        
        return EmotionalTone(
            valence: valence,
            arousal: arousal,
            dominance: dominance,
            emotionType: classifyPrimaryEmotion(valence: valence, arousal: arousal),
            confidence: (speechAnalysis.overallSentiment?.confidence ?? 0.5 + musicAnalysis.confidence) / 2.0
        )
    }
    
    private func classifyPrimaryEmotion(valence: Double, arousal: Double) -> String {
        if valence > 0.6 && arousal > 0.6 {
            return "excited"
        } else if valence > 0.6 && arousal < 0.4 {
            return "content"
        } else if valence < 0.4 && arousal > 0.6 {
            return "angry"
        } else if valence < 0.4 && arousal < 0.4 {
            return "sad"
        } else {
            return "neutral"
        }
    }
    
    private func assessAudioQuality(_ audioData: [Float]) -> AudioQuality {
        // Signal-to-noise ratio estimation
        let signal = audioData.reduce(0.0) { $0 + Double($1 * $1) } / Double(audioData.count)
        let noise = estimateNoise(audioData)
        let snr = signal > 0 ? 10 * log10(signal / max(noise, 1e-10)) : 0.0
        
        // Dynamic range analysis
        let peak = audioData.max() ?? 0.0
        let rms = sqrt(audioData.reduce(0.0) { $0 + Double($1 * $1) } / Double(audioData.count))
        let dynamicRange = rms > 0 ? 20 * log10(Double(peak) / rms) : 0.0
        
        // Clipping detection
        let clippingThreshold: Float = 0.99
        let clippedSamples = audioData.filter { abs($0) > clippingThreshold }.count
        let clippingPercentage = Double(clippedSamples) / Double(audioData.count) * 100.0
        
        return AudioQuality(
            signalToNoiseRatio: snr,
            dynamicRange: dynamicRange,
            clippingPercentage: clippingPercentage,
            bitDepth: 32, // Assuming 32-bit float
            sampleRate: sampleRate,
            overallScore: calculateOverallQualityScore(snr: snr, dynamicRange: dynamicRange, clipping: clippingPercentage)
        )
    }
    
    private func estimateNoise(_ audioData: [Float]) -> Double {
        // Find quietest 10% of samples to estimate noise floor
        let sortedSamples = audioData.map { abs($0) }.sorted()
        let noiseFloorIndex = Int(Double(sortedSamples.count) * 0.1)
        let noiseFloorSamples = Array(sortedSamples[0..<noiseFloorIndex])
        
        return noiseFloorSamples.isEmpty ? 0.001 : noiseFloorSamples.reduce(0.0) { $0 + Double($1 * $1) } / Double(noiseFloorSamples.count)
    }
    
    private func calculateOverallQualityScore(snr: Double, dynamicRange: Double, clipping: Double) -> Double {
        var score = 0.0
        
        // SNR contribution (0-40 dB range)
        score += min(1.0, max(0.0, snr / 40.0)) * 0.4
        
        // Dynamic range contribution (0-30 dB range)
        score += min(1.0, max(0.0, dynamicRange / 30.0)) * 0.4
        
        // Clipping penalty
        score += max(0.0, 1.0 - clipping / 5.0) * 0.2 // Penalize >5% clipping
        
        return score
    }
}

// MARK: - Supporting Classes

private class SpeechRecognizer {
    func analyze(audioData: [Float], duration: CMTime) async throws -> SpeechRecognitionResult {
        // Placeholder implementation - would integrate with Speech framework
        return SpeechRecognitionResult(
            transcription: "Sample transcription",
            wordTimings: [],
            speakerCount: 1,
            languages: ["en"],
            overallSentiment: SentimentResult(positiveScore: 0.6, negativeScore: 0.4, neutralScore: 0.0, confidence: 0.8)
        )
    }
}

private class MusicAnalyzer {
    func analyze(audioData: [Float], duration: CMTime) async throws -> MusicAnalysisResult {
        // Placeholder implementation - would implement music analysis
        return MusicAnalysisResult(
            tempo: 120.0,
            key: "C major",
            genre: "unknown",
            instruments: [],
            confidence: 0.7
        )
    }
}

private class AcousticAnalyzer {
    func analyze(audioData: [Float], duration: CMTime) async throws -> AcousticFeatures {
        return AcousticFeatures(
            mfcc: [Double](repeating: 0.5, count: 13),
            spectralCentroid: 1500.0,
            spectralRolloff: 8000.0,
            zeroCrossingRate: 0.1,
            spectralContrast: [Double](repeating: 0.6, count: 7),
            chroma: [Double](repeating: 0.08, count: 12),
            tonnetz: [Double](repeating: 0.5, count: 6)
        )
    }
    
    func extractFeatures(from audioData: [Float], sampleRate: Double) async throws -> AcousticFeatures {
        return AcousticFeatures(
            mfcc: [Double](repeating: 0.5, count: 13),
            spectralCentroid: 1500.0,
            spectralRolloff: 8000.0,
            zeroCrossingRate: 0.1,
            spectralContrast: [Double](repeating: 0.6, count: 7),
            chroma: [Double](repeating: 0.08, count: 12),
            tonnetz: [Double](repeating: 0.5, count: 6)
        )
    }
}

// MARK: - Audio Analysis Errors

public enum AudioAnalysisError: Error, LocalizedError {
    case noAudioTrack
    case invalidAudioBuffer
    case audioProcessingFailed
    case speechRecognitionFailed
    case musicAnalysisFailed
    case analysisTimeout
    
    public var errorDescription: String? {
        switch self {
        case .noAudioTrack:
            return "No audio track found in the media file"
        case .invalidAudioBuffer:
            return "Invalid audio buffer for analysis"
        case .audioProcessingFailed:
            return "Failed to process audio data"
        case .speechRecognitionFailed:
            return "Speech recognition failed"
        case .musicAnalysisFailed:
            return "Music analysis failed"
        case .analysisTimeout:
            return "Audio analysis timed out"
        }
    }
}
