import AppKit
import SwiftUI
import AVFoundation
import CoreImage
import Metal
import MetalKit

// MARK: - Video Effects Processor

public class VideoEffectsProcessor: ObservableObject {
    @Published var hasActiveEffects: Bool = false
    @Published var currentEffects: [VideoProcessorEffect] = []
    @Published var processingLoad: Double = 0.0
    
    private var player: AVPlayer?
    private var videoOutput: AVPlayerItemVideoOutput?
    private var displayTimer: Timer?
    private var metalDevice: MTLDevice?
    private var ciContext: CIContext?
    private var effectChain: [CIFilter] = []
    private let processingQueue = DispatchQueue(label: "com.autoresolve.effects", qos: .userInitiated)
    
    // Performance metrics
    private var frameProcessingTime: TimeInterval = 0
    private var droppedFrames: Int = 0
    private var lastProcessedTime: CMTime = .zero
    
    public init() {
        setupMetal()
    }
    
    deinit {
        stopProcessing()
    }
    
    private func setupMetal() {
        metalDevice = MTLCreateSystemDefaultDevice()
        if let device = metalDevice {
            ciContext = CIContext(mtlDevice: device)
        }
    }
    
    // MARK: - Player Attachment
    
    public func attachToPlayer(_ player: AVPlayer) {
        self.player = player
        setupVideoOutput()
        startProcessing()
    }
    
    public func detachFromPlayer() {
        stopProcessing()
        player?.currentItem?.remove(videoOutput!)
        player = nil
    }
    
    private func setupVideoOutput() {
        let pixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferOpenGLCompatibilityKey as String: true,
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        
        videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: pixelBufferAttributes)
        player?.currentItem?.add(videoOutput!)
    }
    
    // MARK: - Effect Management
    
    public func addEffect(_ effect: VideoProcessorEffect) {
        currentEffects.append(effect)
        rebuildEffectChain()
        hasActiveEffects = !currentEffects.isEmpty
    }
    
    public func removeEffect(_ effect: VideoProcessorEffect) {
        currentEffects.removeAll { $0.id == effect.id }
        rebuildEffectChain()
        hasActiveEffects = !currentEffects.isEmpty
    }
    
    public func updateEffect(_ effect: VideoProcessorEffect) {
        if let index = currentEffects.firstIndex(where: { $0.id == effect.id }) {
            currentEffects[index] = effect
            rebuildEffectChain()
        }
    }
    
    private func rebuildEffectChain() {
        effectChain = currentEffects.compactMap { effect in
            createFilter(for: effect)
        }
    }
    
    private func createFilter(for effect: VideoProcessorEffect) -> CIFilter? {
        switch effect.type {
        case .colorCorrection(let settings):
            return createColorCorrectionFilter(settings)
        case .blur(let radius):
            return CIFilter(name: "CIGaussianBlur", parameters: ["inputRadius": radius])
        case .brightness(let amount):
            return CIFilter(name: "CIColorControls", parameters: ["inputBrightness": amount])
        case .contrast(let amount):
            return CIFilter(name: "CIColorControls", parameters: ["inputContrast": amount])
        case .saturation(let amount):
            return CIFilter(name: "CIColorControls", parameters: ["inputSaturation": amount])
        case .vignette(let intensity):
            return CIFilter(name: "CIVignette", parameters: ["inputIntensity": intensity])
        case .sharpen(let intensity):
            return CIFilter(name: "CISharpenLuminance", parameters: ["inputSharpness": intensity])
        case .lut(let lutImage):
            return createLUTFilter(lutImage)
        }
    }
    
    private func createColorCorrectionFilter(_ settings: ColorCorrectionSettings) -> CIFilter? {
        let filter = CIFilter(name: "CIColorMatrix")
        
        // Apply color correction matrix
        filter?.setValue(CIVector(x: settings.redGain, y: 0, z: 0, w: 0), forKey: "inputRVector")
        filter?.setValue(CIVector(x: 0, y: settings.greenGain, z: 0, w: 0), forKey: "inputGVector")
        filter?.setValue(CIVector(x: 0, y: 0, z: settings.blueGain, w: 0), forKey: "inputBVector")
        filter?.setValue(CIVector(x: 0, y: 0, z: 0, w: 1), forKey: "inputAVector")
        filter?.setValue(CIVector(x: settings.redOffset, y: settings.greenOffset, z: settings.blueOffset, w: 0), forKey: "inputBiasVector")
        
        return filter
    }
    
    private func createLUTFilter(_ lutImage: NSImage) -> CIFilter? {
        guard lutImage.tiffRepresentation != nil else { return nil }
        
        let filter = CIFilter(name: "CIColorCube")
        filter?.setValue(64, forKey: "inputCubeDimension")
        // Process LUT image to extract color cube data
        // This is simplified - real implementation would parse the LUT properly
        return filter
    }
    
    // MARK: - Frame Processing
    
    private func startProcessing() {
        stopProcessing() // Stop any existing timer
        
        // Use Timer instead of CVDisplayLink for macOS compatibility
        displayTimer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { [weak self] _ in
            self?.processingQueue.async {
                self?.processFrameAsync()
            }
        }
        
        // Ensure timer runs on the common run loop
        if let timer = displayTimer {
            RunLoop.current.add(timer, forMode: .common)
        }
    }
    
    private func stopProcessing() {
        displayTimer?.invalidate()
        displayTimer = nil
    }
    
    private func processFrameAsync() {
        guard let videoOutput = videoOutput,
              let player = player,
              !effectChain.isEmpty else { return }
        
        let currentTime = player.currentTime()
        
        // Skip if we've already processed this frame
        if currentTime == lastProcessedTime {
            return
        }
        
        guard videoOutput.hasNewPixelBuffer(forItemTime: currentTime),
              let pixelBuffer = videoOutput.copyPixelBuffer(forItemTime: currentTime, itemTimeForDisplay: nil) else {
            droppedFrames += 1
            return
        }
        
        lastProcessedTime = currentTime
        let startTime = CACurrentMediaTime()
        
        autoreleasepool {
            // Convert to CIImage
            var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            
            // Apply effect chain
            for filter in effectChain {
                filter.setValue(ciImage, forKey: kCIInputImageKey)
                if let outputImage = filter.outputImage {
                    ciImage = outputImage
                }
            }
            
            // Render back to pixel buffer if we have a context
            if let context = ciContext {
                // Create output pixel buffer (in production, reuse buffers)
                context.render(ciImage, to: pixelBuffer)
            }
        }
        
        let endTime = CACurrentMediaTime()
        frameProcessingTime = endTime - startTime
        
        // Update performance metrics on main thread
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.processingLoad = min(self.frameProcessingTime * 60, 1.0) // Assuming 60fps
        }
    }
}

// MARK: - Video Effect Model

public struct VideoProcessorEffect: Identifiable, Equatable {
    public let id = UUID()
    public var name: String
    public var type: EffectType
    public var intensity: Double = 1.0
    public var enabled: Bool = true
    
    public nonisolated static func == (lhs: VideoProcessorEffect, rhs: VideoProcessorEffect) -> Bool {
        lhs.id == rhs.id
    }
    
    public enum EffectType: Equatable {
        case colorCorrection(ColorCorrectionSettings)
        case blur(radius: Double)
        case brightness(amount: Double)
        case contrast(amount: Double)
        case saturation(amount: Double)
        case vignette(intensity: Double)
        case sharpen(intensity: Double)
        case lut(NSImage)
        
        public nonisolated static func == (lhs: EffectType, rhs: EffectType) -> Bool {
            switch (lhs, rhs) {
            case (.blur(let l), .blur(let r)): return l == r
            case (.brightness(let l), .brightness(let r)): return l == r
            case (.contrast(let l), .contrast(let r)): return l == r
            case (.saturation(let l), .saturation(let r)): return l == r
            case (.vignette(let l), .vignette(let r)): return l == r
            case (.sharpen(let l), .sharpen(let r)): return l == r
            case (.colorCorrection(let l), .colorCorrection(let r)): return l == r
            case (.lut(_), .lut(_)): return true // NSImage comparison simplified
            default: return false
            }
        }
    }
}

public struct ColorCorrectionSettings: Equatable {
    public var temperature: Double = 6500
    public var tint: Double = 0
    public var exposure: Double = 0
    public var highlights: Double = 0
    public var shadows: Double = 0
    public var whites: Double = 0
    public var blacks: Double = 0
    public var vibrance: Double = 0
    public var redGain: CGFloat = 1.0
    public var greenGain: CGFloat = 1.0
    public var blueGain: CGFloat = 1.0
    public var redOffset: CGFloat = 0
    public var greenOffset: CGFloat = 0
    public var blueOffset: CGFloat = 0
}

// MARK: - Effects Preview Panel

public struct EffectsPreviewPanel: View {
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    @State private var selectedEffect: VideoProcessorEffect?
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Active Effects")
                .font(.caption.bold())
            
            ForEach(effectsProcessor.currentEffects) { effect in
                HStack {
                    Image(systemName: iconForEffect(effect))
                        .font(.caption)
                        .frame(width: 16)
                    
                    Text(effect.name)
                        .font(.caption)
                    
                    Spacer()
                    
                    Text("\(Int(effect.intensity * 100))%")
                        .font(.caption.monospacedDigit())
                        .foregroundColor(.secondary)
                    
                    Toggle("", isOn: .constant(effect.enabled))
                        .toggleStyle(.switch)
                        .scaleEffect(0.7)
                }
                .padding(4)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .fill(selectedEffect?.id == effect.id ? 
                              Color.accentColor.opacity(0.2) : Color.clear)
                )
                .onTapGesture {
                    selectedEffect = effect
                }
            }
            
            Divider()
            
            // Processing load indicator
            HStack {
                Text("GPU Load")
                    .font(.caption)
                
                ProgressView(value: effectsProcessor.processingLoad)
                    .progressViewStyle(.linear)
                    .frame(width: 80)
                
                Text("\(Int(effectsProcessor.processingLoad * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundColor(effectsProcessor.processingLoad > 0.8 ? .red : .green)
            }
        }
        .padding(12)
        .frame(width: 250)
        .background(Color.black.opacity(0.8))
        .cornerRadius(8)
    }
    
    private func iconForEffect(_ effect: VideoProcessorEffect) -> String {
        switch effect.type {
        case .colorCorrection: return "slider.horizontal.3"
        case .blur: return "drop.circle"
        case .brightness: return "sun.max"
        case .contrast: return "circle.lefthalf.filled"
        case .saturation: return "paintpalette"
        case .vignette: return "camera.filters"
        case .sharpen: return "sparkles"
        case .lut: return "square.grid.3x3"
        }
    }
}

// MARK: - Preview Rendering Pipeline

public class PreviewRenderingPipeline: ObservableObject {
    @Published var isRendering = false
    @Published var renderProgress: Double = 0
    @Published var estimatedTimeRemaining: TimeInterval = 0
    
    private let exportSession: AVAssetExportSession?
    private var renderTimer: Timer?
    
    init(asset: AVAsset? = nil) {
        if let asset = asset {
            exportSession = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHighestQuality)
        } else {
            exportSession = nil
        }
    }
    
    public func renderPreview(
        from startTime: TimeInterval,
        to endTime: TimeInterval,
        effects: [VideoProcessorEffect],
        completion: @escaping (URL?) -> Void
    ) async {
        guard let exportSession = exportSession else {
            completion(nil)
            return
        }
        
        isRendering = true
        renderProgress = 0
        
        // Configure export
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("preview_\(UUID().uuidString).mov")
        
        exportSession.outputURL = outputURL
        exportSession.outputFileType = .mov
        exportSession.timeRange = CMTimeRange(
            start: CMTime(seconds: startTime, preferredTimescale: 600),
            end: CMTime(seconds: endTime, preferredTimescale: 600)
        )
        
        // Apply video composition with effects
        if !effects.isEmpty {
            let composition = await createComposition(with: effects, for: exportSession.asset)
            exportSession.videoComposition = composition
        }
        
        // Start render timer
        renderTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            DispatchQueue.main.async {
                self.renderProgress = Double(exportSession.progress)
                self.estimatedTimeRemaining = self.calculateTimeRemaining(progress: exportSession.progress)
            }
        }
        
        // Export
        exportSession.exportAsynchronously {
            DispatchQueue.main.async {
                self.renderTimer?.invalidate()
                self.isRendering = false
                self.renderProgress = 1.0
                
                switch exportSession.status {
                case .completed:
                    completion(outputURL)
                case .failed, .cancelled:
                    completion(nil)
                default:
                    break
                }
            }
        }
    }
    
    private func createComposition(with effects: [VideoProcessorEffect], for asset: AVAsset) async -> AVVideoComposition {
        let composition = AVMutableVideoComposition()
        composition.renderSize = CGSize(width: 1920, height: 1080)
        composition.frameDuration = CMTime(value: 1, timescale: 30)
        
        // Create instruction for applying effects
        let instruction = AVMutableVideoCompositionInstruction()
        // Use async loading for duration
        let duration = try? await asset.load(.duration)
        instruction.timeRange = CMTimeRange(start: .zero, duration: duration ?? .zero)
        
        // Add layer instruction with Core Image filters
        // Use async loading for video tracks
        if let videoTracks = try? await asset.loadTracks(withMediaType: .video),
           let videoTrack = videoTracks.first {
            let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTrack)
            
            // Apply transforms or effects here
            instruction.layerInstructions = [layerInstruction]
        }
        
        composition.instructions = [instruction]
        
        // Set up Core Image rendering
        composition.customVideoCompositorClass = VideoEffectCompositor.self
        
        return composition
    }
    
    private func calculateTimeRemaining(progress: Float) -> TimeInterval {
        guard progress > 0 else { return 0 }
        
        let elapsedTime = CACurrentMediaTime()
        let estimatedTotalTime = elapsedTime / Double(progress)
        return estimatedTotalTime - elapsedTime
    }
}

// Custom video compositor for effects
class VideoEffectCompositor: NSObject, AVVideoCompositing {
    var sourcePixelBufferAttributes: [String : Any]? {
        return [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
    }
    
    var requiredPixelBufferAttributesForRenderContext: [String : Any] {
        return [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
    }
    
    func renderContextChanged(_ newRenderContext: AVVideoCompositionRenderContext) {
        // Handle context changes
    }
    
    func startRequest(_ request: AVAsynchronousVideoCompositionRequest) {
        // Apply effects and render
        autoreleasepool {
            guard let sourceFrame = request.sourceFrame(byTrackID: Int32(request.sourceTrackIDs[0].intValue)),
                  let pixelBuffer = request.renderContext.newPixelBuffer() else {
                request.finish(with: NSError(domain: "VideoEffectCompositor", code: -1))
                return
            }
            
            // Apply effects using Core Image
            let ciImage = CIImage(cvPixelBuffer: sourceFrame)
            let context = CIContext()
            
            // Render processed image to output buffer
            context.render(ciImage, to: pixelBuffer)
            
            request.finish(withComposedVideoFrame: pixelBuffer)
        }
    }
    
    func cancelAllPendingVideoCompositionRequests() {
        // Clean up any pending requests
    }
}
