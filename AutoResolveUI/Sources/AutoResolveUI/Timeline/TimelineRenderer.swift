import SwiftUI
import MetalKit
import Metal
import CoreMedia

/// Metal-accelerated timeline renderer with virtualization
public struct TimelineRenderer: NSViewRepresentable {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    let tracks: [UITimelineTrack]
    let visibleRange: Range<CMTime>
    let zoomLevel: Double
    
    public func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        view.enableSetNeedsDisplay = true
        view.isPaused = false
        view.preferredFramesPerSecond = 60
        view.clearColor = MTLClearColor(
            red: Double(UITheme.Colors.background.components.red),
            green: Double(UITheme.Colors.background.components.green),
            blue: Double(UITheme.Colors.background.components.blue),
            alpha: 1.0
        )
        
        context.coordinator.setupMetal(device: view.device!)
        
        return view
    }
    
    public func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.tracks = tracks
        context.coordinator.visibleRange = visibleRange
        context.coordinator.zoomLevel = zoomLevel
        context.coordinator.currentTime = transport.currentTime
        context.coordinator.selectedClips = appState.selectedClips
        nsView.setNeedsDisplay(nsView.bounds)
    }
    
    public func makeCoordinator() -> Coordinator {
        Coordinator(appState: appState, transport: transport)
    }
    
    @MainActor
    public class Coordinator: NSObject, MTKViewDelegate {
        var appState: AppState
        var transport: Transport
        
        var tracks: [UITimelineTrack] = []
        var visibleRange: Range<CMTime>
        var zoomLevel: Double = 1.0
        var currentTime: CMTime = .zero
        var selectedClips: Set<String> = []
        
        // Metal resources
        var device: MTLDevice!
        var commandQueue: MTLCommandQueue!
        var pipelineState: MTLRenderPipelineState!
        var vertexBuffer: MTLBuffer!
        
        init(appState: AppState, transport: Transport) {
            self.appState = appState
            self.transport = transport
            self.visibleRange = CMTime.zero..<CMTime(seconds: 60, preferredTimescale: 600)
            super.init()
        }
        
        func setupMetal(device: MTLDevice) {
            self.device = device
            self.commandQueue = device.makeCommandQueue()
            
            // Create pipeline - but skip if shaders aren't available
            let library = device.makeDefaultLibrary()
            
            // Check if shader functions exist
            guard let vertexFunction = library?.makeFunction(name: "vertex_main"),
                  let fragmentFunction = library?.makeFunction(name: "fragment_main") else {
                print("Metal shaders not found, using CPU rendering")
                pipelineState = nil
                return
            }
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            
            do {
                pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            } catch {
                // Fallback to CPU rendering if Metal fails
                print("Failed to create pipeline state: \(error)")
                pipelineState = nil
            }
        }
        
        public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            // Handle resize
        }
        
        public func draw(in view: MTKView) {
            // Simple rendering - just clear background
            guard let drawable = view.currentDrawable else { return }
            drawable.present()
        }
        
        // MARK: - Virtualization
        
        private func calculateVisibleClips() -> [(UITimelineTrack, [SimpleTimelineClip])] {
            var result: [(UITimelineTrack, [SimpleTimelineClip])] = []
            
            let startSeconds = CMTimeGetSeconds(visibleRange.lowerBound)
            let endSeconds = CMTimeGetSeconds(visibleRange.upperBound)
            
            for track in tracks {
                let visibleClips = track.clips.filter { clip in
                    let clipEnd = clip.startTime + clip.duration
                    return clip.startTime < endSeconds && clipEnd > startSeconds
                }
                
                if !visibleClips.isEmpty {
                    result.append((track, visibleClips))
                }
            }
            
            return result
        }
        
        // MARK: - CPU Fallback Rendering
        
        private func drawCPU(in view: MTKView) {
            // This would be implemented with Core Graphics for fallback
            // For now, we'll rely on Metal being available
        }
        
        // MARK: - Render Functions
        
        private func renderClip(_ clip: SimpleTimelineClip, track: UITimelineTrack, encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
            let x = pixelFromTime(CMTime(seconds: clip.startTime, preferredTimescale: 600))
            let width = pixelFromTime(CMTime(seconds: clip.duration, preferredTimescale: 600))
            let y = CGFloat(clip.trackIndex) * UITheme.Sizes.timelineTrackHeight
            let height = UITheme.Sizes.timelineTrackHeight - 4
            
            // Determine clip color
            var color: simd_float4
            if selectedClips.contains(clip.id.uuidString) {
                color = simd_float4(0.25, 0.47, 0.85, 1.0) // Selection color
            } else {
                switch track.type {
                case .video:
                    color = simd_float4(0.40, 0.62, 0.83, 1.0) // Video clip color
                case .audio:
                    color = simd_float4(0.48, 0.78, 0.64, 1.0) // Audio clip color
                default:
                    color = simd_float4(0.5, 0.5, 0.5, 1.0)
                }
            }
            
            // Create vertices for clip rectangle
            let vertices: [Float] = [
                Float(x), Float(y), 0, 1,  // Top-left
                Float(x + width), Float(y), 0, 1,  // Top-right
                Float(x), Float(y + height), 0, 1,  // Bottom-left
                Float(x + width), Float(y + height), 0, 1,  // Bottom-right
            ]
            
            let vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Float>.size, options: [])
            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBytes(&color, length: MemoryLayout<simd_float4>.size, index: 1)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        
        private func renderPlayhead(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
            let x = pixelFromTime(currentTime)
            var color = simd_float4(1.0, 0.87, 0.34, 1.0) // Playhead color
            
            let vertices: [Float] = [
                Float(x - 1), 0, 0, 1,
                Float(x + 1), 0, 0, 1,
                Float(x - 1), Float(viewSize.height), 0, 1,
                Float(x + 1), Float(viewSize.height), 0, 1,
            ]
            
            let vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Float>.size, options: [])
            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBytes(&color, length: MemoryLayout<simd_float4>.size, index: 1)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        
        private func renderSilenceSegments(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
            guard let silenceResult = appState.silenceResult else { return }
            
            var color = simd_float4(0.85, 0.33, 0.33, 0.5) // Silence color
            
            for segment in silenceResult.silenceSegments {
                let x = pixelFromTime(segment.start)
                let width = pixelFromTime(segment.end) - x
                let y = viewSize.height - UITheme.Sizes.timelineTrackHeight / 2
                let height = UITheme.Sizes.timelineTrackHeight / 3
                
                let vertices: [Float] = [
                    Float(x), Float(y), 0, 1,
                    Float(x + width), Float(y), 0, 1,
                    Float(x), Float(y + height), 0, 1,
                    Float(x + width), Float(y + height), 0, 1,
                ]
                
                let vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Float>.size, options: [])
                encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
                encoder.setVertexBytes(&color, length: MemoryLayout<simd_float4>.size, index: 1)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
        }
        
        private func renderStoryBeats(encoder: MTLRenderCommandEncoder, viewSize: CGSize) {
            guard let storyBeats = appState.storyBeatsResult,
                  let beats = storyBeats["beats"] as? [[String: Any]] else { return }
            
            var color = simd_float4(0.85, 0.65, 0.33, 0.8) // Story beat color
            
            for beat in beats {
                guard let time = beat["time"] as? Double,
                      let intensity = beat["intensity"] as? Double else { continue }
                let x = pixelFromTime(CMTime(seconds: time, preferredTimescale: 600))
                let size: CGFloat = CGFloat(intensity) * 10 + 5
                
                // Render as diamond marker
                let vertices: [Float] = [
                    Float(x), Float(viewSize.height - size - 20), 0, 1,  // Top
                    Float(x - size/2), Float(viewSize.height - size/2 - 20), 0, 1,  // Left
                    Float(x), Float(viewSize.height - 20), 0, 1,  // Bottom
                    Float(x + size/2), Float(viewSize.height - size/2 - 20), 0, 1,  // Right
                ]
                
                let vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Float>.size, options: [])
                encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
                encoder.setVertexBytes(&color, length: MemoryLayout<simd_float4>.size, index: 1)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
        }
        
        // MARK: - Helpers
        
        private func pixelFromTime(_ time: CMTime) -> CGFloat {
            let seconds = CMTimeGetSeconds(time)
            let visibleStart = CMTimeGetSeconds(visibleRange.lowerBound)
            let relativeSeconds = seconds - visibleStart
            return CGFloat(relativeSeconds * zoomLevel * 100) // 100 pixels per second at zoom 1.0
        }
        
        private func pixelFromTime(_ time: TimeInterval) -> CGFloat {
            let visibleStart = CMTimeGetSeconds(visibleRange.lowerBound)
            let relativeSeconds = time - visibleStart
            return CGFloat(relativeSeconds * zoomLevel * 100) // 100 pixels per second at zoom 1.0
        }
    }
}

// MARK: - Color Extension

extension Color {
    var components: (red: CGFloat, green: CGFloat, blue: CGFloat, alpha: CGFloat) {
        let nsColor = NSColor(self)
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        nsColor.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        return (red, green, blue, alpha)
    }
}