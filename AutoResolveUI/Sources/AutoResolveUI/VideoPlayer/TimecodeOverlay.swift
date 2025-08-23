import SwiftUI
import AVFoundation

// MARK: - Timecode Overlay

public struct TimecodeOverlay: View {
    @ObservedObject var timeline: TimelineModel
    let currentTime: TimeInterval
    @State private var showExtendedInfo = false
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Main timecode display
            HStack(spacing: 12) {
                // Timecode
                Text(timeline.timecode(for: currentTime))
                    .font(.system(size: 24, weight: .medium, design: .monospaced))
                    .foregroundColor(.white)
                
                // Frame number
                Text("F: \(currentFrame)")
                    .font(.system(size: 14, weight: .regular, design: .monospaced))
                    .foregroundColor(.white.opacity(0.8))
                
                // Playback rate indicator
                if timeline.playbackRate != 1.0 {
                    Text("\(String(format: "%.1fx", timeline.playbackRate))")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(playbackRateColor)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.black.opacity(0.3)))
                }
            }
            
            // Extended info
            if showExtendedInfo {
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Duration:")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.6))
                        Text(timeline.timecode(for: timeline.duration))
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.white.opacity(0.8))
                    }
                    
                    HStack {
                        Text("In:")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.6))
                        Text(timeline.timecode(for: timeline.workAreaStart))
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.white.opacity(0.8))
                        
                        Text("Out:")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.6))
                        Text(timeline.timecode(for: timeline.workAreaEnd))
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.white.opacity(0.8))
                    }
                    
                    if let activeClip = getActiveClip() {
                        Divider()
                            .background(Color.white.opacity(0.3))
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text(activeClip.name)
                                .font(.caption.bold())
                                .foregroundColor(.white.opacity(0.9))
                                .lineLimit(1)
                            
                            HStack {
                                Text("Clip TC:")
                                    .font(.caption)
                                    .foregroundColor(.white.opacity(0.6))
                                Text(clipTimecode(for: activeClip))
                                    .font(.caption.monospacedDigit())
                                    .foregroundColor(.white.opacity(0.8))
                            }
                        }
                    }
                }
                .padding(.top, 4)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.7))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
        )
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.2)) {
                showExtendedInfo.toggle()
            }
        }
    }
    
    private var currentFrame: Int {
        Int(currentTime * Double(timeline.frameRate))
    }
    
    private var playbackRateColor: Color {
        if timeline.playbackRate > 1.0 {
            return .green
        } else if timeline.playbackRate < 1.0 {
            return .orange
        } else {
            return .white
        }
    }
    
    private func getActiveClip() -> TimelineClip? {
        for track in timeline.tracks {
            for clip in track.clips {
                if currentTime >= clip.startTime && 
                   currentTime < clip.startTime + clip.duration {
                    return clip
                }
            }
        }
        return nil
    }
    
    private func clipTimecode(for clip: TimelineClip) -> String {
        let clipTime = currentTime - clip.startTime + clip.inPoint
        return timeline.timecode(for: clipTime)
    }
}

// MARK: - Broadcast Safe Indicator

public struct BroadcastSafeIndicator: View {
    let isSafe: Bool
    let luminanceRange: ClosedRange<Float>
    let chromaRange: ClosedRange<Float>
    
    public var body: some View {
        HStack(spacing: 8) {
            Image(systemName: isSafe ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundColor(isSafe ? .green : .orange)
            
            VStack(alignment: .leading, spacing: 2) {
                Text("Broadcast Safe")
                    .font(.caption.bold())
                
                HStack(spacing: 12) {
                    Label("Luma", systemImage: "sun.max")
                        .font(.caption2)
                    Text("\(Int(luminanceRange.lowerBound))-\(Int(luminanceRange.upperBound))")
                        .font(.caption2.monospacedDigit())
                    
                    Label("Chroma", systemImage: "paintpalette")
                        .font(.caption2)
                    Text("\(Int(chromaRange.lowerBound))-\(Int(chromaRange.upperBound))")
                        .font(.caption2.monospacedDigit())
                }
            }
        }
        .padding(8)
        .background(Color.black.opacity(0.6))
        .cornerRadius(6)
        .foregroundColor(.white)
    }
}

// MARK: - Scopes Panel

public struct ScopesPanel: View {
    @ObservedObject var scopesManager: VideoScopesManager
    @State private var selectedScope: ScopeType = .waveform
    
    public enum ScopeType: String, CaseIterable {
        case waveform = "Waveform"
        case vectorscope = "Vectorscope"
        case histogram = "Histogram"
        case parade = "RGB Parade"
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Scope selector
            Picker("Scope", selection: $selectedScope) {
                ForEach(ScopeType.allCases, id: \.self) { scope in
                    Text(scope.rawValue).tag(scope)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding(8)
            
            // Scope display
            ZStack {
                Color.black
                
                switch selectedScope {
                case .waveform:
                    WaveformScope(data: scopesManager.waveformData)
                case .vectorscope:
                    VectorscopeView(data: scopesManager.vectorscopeData)
                case .histogram:
                    HistogramView(data: scopesManager.histogramData)
                case .parade:
                    RGBParadeView(data: scopesManager.rgbParadeData)
                }
            }
            .aspectRatio(16/9, contentMode: .fit)
            
            // Scope settings
            HStack {
                Toggle("IRE", isOn: $scopesManager.showIRE)
                    .toggleStyle(.button)
                
                Toggle("75%", isOn: $scopesManager.show75Percent)
                    .toggleStyle(.button)
                
                Spacer()
                
                Text("Opacity:")
                    .font(.caption)
                
                Slider(value: $scopesManager.scopeOpacity, in: 0.3...1.0)
                    .frame(width: 100)
            }
            .padding(8)
            .font(.caption)
        }
        .frame(width: 300, height: 250)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}

// MARK: - Video Scopes Manager

public class VideoScopesManager: ObservableObject {
    @Published var waveformData = WaveformData()
    @Published var vectorscopeData = VectorscopeData()
    @Published var histogramData = HistogramData()
    @Published var rgbParadeData = RGBParadeData()
    
    @Published var showIRE = true
    @Published var show75Percent = false
    @Published var scopeOpacity: Double = 0.8
    
    private var displayLink: CVDisplayLink?
    private weak var player: AVPlayer?
    
    public func startAnalyzing(player: AVPlayer) {
        self.player = player
        setupDisplayLink()
    }
    
    public func stopAnalyzing() {
        if let displayLink = displayLink {
            CVDisplayLinkStop(displayLink)
        }
    }
    
    private func setupDisplayLink() {
        CVDisplayLinkCreateWithActiveCGDisplays(&displayLink)
        // Setup callback for frame analysis
        CVDisplayLinkStart(displayLink!)
    }
    
    // Scope data structures
    public struct WaveformData {
        var luminance: [Float] = []
        var red: [Float] = []
        var green: [Float] = []
        var blue: [Float] = []
    }
    
    public struct VectorscopeData {
        var points: [(x: Float, y: Float)] = []
        var saturation: Float = 0
    }
    
    public struct HistogramData {
        var red: [Int] = Array(repeating: 0, count: 256)
        var green: [Int] = Array(repeating: 0, count: 256)
        var blue: [Int] = Array(repeating: 0, count: 256)
    }
    
    public struct RGBParadeData {
        var red: [Float] = []
        var green: [Float] = []
        var blue: [Float] = []
    }
}

// MARK: - Individual Scope Views

struct WaveformScope: View {
    let data: VideoScopesManager.WaveformData
    
    var body: some View {
        Canvas { context, size in
            // Draw waveform
            drawChannel(data.luminance, color: .white, in: context, size: size)
        }
    }
    
    private func drawChannel(_ values: [Float], color: Color, in context: GraphicsContext, size: CGSize) {
        guard !values.isEmpty else { return }
        
        var path = Path()
        let xStep = size.width / CGFloat(values.count)
        
        for (index, value) in values.enumerated() {
            let x = CGFloat(index) * xStep
            let y = size.height - (CGFloat(value) * size.height)
            
            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
        
        context.stroke(path, with: .color(color.opacity(0.8)), lineWidth: 1)
    }
}

struct VectorscopeView: View {
    let data: VideoScopesManager.VectorscopeData
    
    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) / 2 * 0.9
            
            // Draw graticule
            context.stroke(
                Circle().path(in: CGRect(
                    x: center.x - radius,
                    y: center.y - radius,
                    width: radius * 2,
                    height: radius * 2
                )),
                with: .color(.white.opacity(0.3)),
                lineWidth: 1
            )
            
            // Draw color targets
            let targets = [
                (angle: 0, color: Color.red, label: "R"),
                (angle: 60, color: Color.yellow, label: "Yl"),
                (angle: 120, color: Color.green, label: "G"),
                (angle: 180, color: Color.cyan, label: "Cy"),
                (angle: 240, color: Color.blue, label: "B"),
                (angle: 300, color: Color(red: 1, green: 0, blue: 1), label: "Mg")
            ]
            
            for target in targets {
                let angle = Angle(degrees: Double(target.angle))
                let x = center.x + cos(angle.radians) * radius * 0.75
                let y = center.y + sin(angle.radians) * radius * 0.75
                
                context.fill(
                    Circle().path(in: CGRect(x: x - 3, y: y - 3, width: 6, height: 6)),
                    with: .color(target.color)
                )
            }
            
            // Draw data points
            for point in data.points {
                let x = center.x + CGFloat(point.x) * radius
                let y = center.y + CGFloat(point.y) * radius
                
                context.fill(
                    Circle().path(in: CGRect(x: x - 1, y: y - 1, width: 2, height: 2)),
                    with: .color(.white.opacity(0.6))
                )
            }
        }
    }
}

struct HistogramView: View {
    let data: VideoScopesManager.HistogramData
    
    var body: some View {
        Canvas { context, size in
            let barWidth = size.width / 256
            
            // Draw RGB histograms
            drawHistogram(data.red, color: .red.opacity(0.5), in: context, size: size, barWidth: barWidth)
            drawHistogram(data.green, color: .green.opacity(0.5), in: context, size: size, barWidth: barWidth)
            drawHistogram(data.blue, color: .blue.opacity(0.5), in: context, size: size, barWidth: barWidth)
        }
    }
    
    private func drawHistogram(_ values: [Int], color: Color, in context: GraphicsContext, size: CGSize, barWidth: CGFloat) {
        let maxValue = values.max() ?? 1
        
        for (index, value) in values.enumerated() {
            let height = CGFloat(value) / CGFloat(maxValue) * size.height
            let x = CGFloat(index) * barWidth
            let y = size.height - height
            
            context.fill(
                Rectangle().path(in: CGRect(x: x, y: y, width: barWidth, height: height)),
                with: .color(color)
            )
        }
    }
}

struct RGBParadeView: View {
    let data: VideoScopesManager.RGBParadeData
    
    var body: some View {
        HStack(spacing: 2) {
            ParadeChannel(values: data.red, color: .red)
            ParadeChannel(values: data.green, color: .green)
            ParadeChannel(values: data.blue, color: .blue)
        }
    }
}

struct ParadeChannel: View {
    let values: [Float]
    let color: Color
    
    var body: some View {
        Canvas { context, size in
            guard !values.isEmpty else { return }
            
            var path = Path()
            let xStep = size.width / CGFloat(values.count)
            
            for (index, value) in values.enumerated() {
                let x = CGFloat(index) * xStep
                let y = size.height - (CGFloat(value) * size.height)
                
                if index == 0 {
                    path.move(to: CGPoint(x: x, y: y))
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
            
            context.stroke(path, with: .color(color), lineWidth: 1)
        }
    }
}