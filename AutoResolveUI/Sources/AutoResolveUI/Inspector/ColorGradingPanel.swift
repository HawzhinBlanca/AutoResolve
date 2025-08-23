import SwiftUI
import CoreImage

// MARK: - Professional Color Grading Panel

public struct ColorGradingPanel: View {
    @ObservedObject var timeline: TimelineModel
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    @State private var selectedTool: ColorTool = .wheels
    @State private var selectedWheel: ColorWheel = .lift
    @State private var showVectorscope = true
    @State private var showWaveform = true
    
    enum ColorTool: String, CaseIterable {
        case wheels = "Color Wheels"
        case curves = "Curves"
        case hsl = "HSL Qualifiers"
        case lut = "LUT"
        case scopes = "Scopes"
        
        var icon: String {
            switch self {
            case .wheels: return "circle.grid.2x2"
            case .curves: return "chart.line.uptrend.xyaxis"
            case .hsl: return "slider.horizontal.3"
            case .lut: return "square.grid.3x3"
            case .scopes: return "waveform"
            }
        }
    }
    
    enum ColorWheel: String, CaseIterable {
        case lift = "Lift"
        case gamma = "Gamma" 
        case gain = "Gain"
        case offset = "Offset"
        
        var description: String {
            switch self {
            case .lift: return "Shadows"
            case .gamma: return "Midtones"
            case .gain: return "Highlights"
            case .offset: return "Overall"
            }
        }
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Color Grading")
                    .font(.headline)
                
                Spacer()
                
                Button("Reset All") {
                    resetAllColorGrading()
                }
                .buttonStyle(.bordered)
            }
            .padding()
            
            // Tool selector
            Picker("Tool", selection: $selectedTool) {
                ForEach(ColorTool.allCases, id: \.self) { tool in
                    Label(tool.rawValue, systemImage: tool.icon)
                        .tag(tool)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding(.horizontal)
            
            Divider()
            
            // Content
            ScrollView {
                VStack(spacing: 16) {
                    switch selectedTool {
                    case .wheels:
                        ColorWheelsSection(
                            selectedWheel: $selectedWheel,
                            effectsProcessor: effectsProcessor
                        )
                        
                    case .curves:
                        CurvesSection(effectsProcessor: effectsProcessor)
                        
                    case .hsl:
                        HSLQualifierSection(effectsProcessor: effectsProcessor)
                        
                    case .lut:
                        LUTSection(effectsProcessor: effectsProcessor)
                        
                    case .scopes:
                        ScopesSection(
                            showVectorscope: $showVectorscope,
                            showWaveform: $showWaveform
                        )
                    }
                }
                .padding()
            }
        }
        .frame(width: 400)
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private func resetAllColorGrading() {
        // Reset all color grading parameters
    }
}

// MARK: - Color Wheels Section

struct ColorWheelsSection: View {
    @Binding var selectedWheel: ColorGradingPanel.ColorWheel
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var liftValues = ColorWheelValues()
    @State private var gammaValues = ColorWheelValues()
    @State private var gainValues = ColorWheelValues()
    @State private var offsetValues = ColorWheelValues()
    
    var body: some View {
        VStack(spacing: 16) {
            // Wheel selector
            Picker("Wheel", selection: $selectedWheel) {
                ForEach(ColorGradingPanel.ColorWheel.allCases, id: \.self) { wheel in
                    Text(wheel.description).tag(wheel)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            
            // Color wheels grid
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2)) {
                ColorWheelView(
                    title: "Lift",
                    subtitle: "Shadows",
                    values: $liftValues,
                    isSelected: selectedWheel == .lift
                ) {
                    selectedWheel = .lift
                }
                
                ColorWheelView(
                    title: "Gamma",
                    subtitle: "Midtones", 
                    values: $gammaValues,
                    isSelected: selectedWheel == .gamma
                ) {
                    selectedWheel = .gamma
                }
                
                ColorWheelView(
                    title: "Gain",
                    subtitle: "Highlights",
                    values: $gainValues,
                    isSelected: selectedWheel == .gain
                ) {
                    selectedWheel = .gain
                }
                
                ColorWheelView(
                    title: "Offset",
                    subtitle: "Overall",
                    values: $offsetValues,
                    isSelected: selectedWheel == .offset
                ) {
                    selectedWheel = .offset
                }
            }
            
            // Selected wheel controls
            VStack(spacing: 12) {
                Text("\(selectedWheel.rawValue) Controls")
                    .font(.headline)
                
                switch selectedWheel {
                case .lift:
                    ColorWheelControls(values: $liftValues)
                case .gamma:
                    ColorWheelControls(values: $gammaValues)
                case .gain:
                    ColorWheelControls(values: $gainValues)
                case .offset:
                    ColorWheelControls(values: $offsetValues)
                }
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))
            .cornerRadius(8)
        }
    }
}

// MARK: - Color Wheel Models

struct ColorWheelValues {
    var x: Double = 0.0  // Horizontal position (-1 to 1)
    var y: Double = 0.0  // Vertical position (-1 to 1)
    var luminance: Double = 0.0  // Luminance adjustment (-1 to 1)
    
    var saturation: Double {
        sqrt(x * x + y * y)
    }
    
    var hue: Double {
        atan2(y, x)
    }
}

// MARK: - Color Wheel View

struct ColorWheelView: View {
    let title: String
    let subtitle: String
    @Binding var values: ColorWheelValues
    let isSelected: Bool
    let onSelect: () -> Void
    
    @State private var isDragging = false
    
    var body: some View {
        VStack(spacing: 8) {
            // Title
            VStack(spacing: 2) {
                Text(title)
                    .font(.caption.bold())
                Text(subtitle)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            // Color wheel
            ZStack {
                // Background circle
                Circle()
                    .fill(colorWheelGradient)
                    .frame(width: 120, height: 120)
                    .overlay(
                        Circle()
                            .stroke(isSelected ? Color.accentColor : Color.secondary, lineWidth: 2)
                    )
                
                // Center indicator
                Circle()
                    .fill(Color.white)
                    .frame(width: 8, height: 8)
                    .shadow(radius: 2)
                    .offset(
                        x: CGFloat(values.x * 50),
                        y: CGFloat(-values.y * 50)
                    )
                    .gesture(
                        DragGesture()
                            .onChanged { drag in
                                isDragging = true
                                let center = CGPoint(x: 60, y: 60)
                                let offset = CGPoint(
                                    x: drag.location.x - center.x,
                                    y: center.y - drag.location.y
                                )
                                
                                let distance = sqrt(offset.x * offset.x + offset.y * offset.y)
                                let maxDistance: CGFloat = 50
                                
                                if distance <= maxDistance {
                                    values.x = Double(offset.x / maxDistance)
                                    values.y = Double(offset.y / maxDistance)
                                } else {
                                    let angle = atan2(offset.y, offset.x)
                                    values.x = Double(cos(angle))
                                    values.y = Double(sin(angle))
                                }
                            }
                            .onEnded { _ in
                                isDragging = false
                            }
                    )
            }
            .onTapGesture {
                onSelect()
            }
            
            // Luminance slider
            VStack(spacing: 4) {
                Text("Lum")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                Slider(value: $values.luminance, in: -1...1)
                    .frame(width: 100)
                
                Text("\(Int(values.luminance * 100))")
                    .font(.caption2)
                    .monospacedDigit()
            }
        }
        .padding(8)
        .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .cornerRadius(8)
    }
    
    private var colorWheelGradient: AngularGradient {
        AngularGradient(
            gradient: Gradient(colors: [
                .red, .yellow, .green, .cyan, .blue, .purple, .red
            ]),
            center: .center
        )
    }
}

// MARK: - Color Wheel Controls

struct ColorWheelControls: View {
    @Binding var values: ColorWheelValues
    
    var body: some View {
        VStack(spacing: 12) {
            // Hue and Saturation
            HStack {
                VStack(alignment: .leading) {
                    Text("Hue")
                        .font(.caption)
                    Text("\(Int(values.hue * 180 / .pi))°")
                        .font(.caption.monospacedDigit())
                }
                .frame(width: 60, alignment: .leading)
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("Saturation")
                        .font(.caption)
                    Text("\(Int(values.saturation * 100))%")
                        .font(.caption.monospacedDigit())
                }
                .frame(width: 80, alignment: .trailing)
            }
            
            // Luminance with fine control
            VStack(alignment: .leading, spacing: 4) {
                Text("Luminance")
                    .font(.caption)
                
                HStack {
                    Slider(value: $values.luminance, in: -1...1)
                    
                    Text("\(String(format: "%.2f", values.luminance))")
                        .font(.caption.monospacedDigit())
                        .frame(width: 40)
                }
            }
            
            // Reset button
            Button("Reset") {
                values = ColorWheelValues()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }
}

// MARK: - Curves Section

struct CurvesSection: View {
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    @State private var selectedCurve: CurveType = .rgb
    @State private var curvePoints: [CGPoint] = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 1)]
    
    enum CurveType: String, CaseIterable {
        case rgb = "RGB"
        case red = "Red"
        case green = "Green"
        case blue = "Blue"
        
        var color: Color {
            switch self {
            case .rgb: return .white
            case .red: return .red
            case .green: return .green
            case .blue: return .blue
            }
        }
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Curve selector
            Picker("Curve", selection: $selectedCurve) {
                ForEach(CurveType.allCases, id: \.self) { curve in
                    Text(curve.rawValue)
                        .foregroundColor(curve.color)
                        .tag(curve)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            
            // Curve editor
            CurveEditor(
                points: $curvePoints,
                curveColor: selectedCurve.color
            )
            .frame(height: 250)
            .background(Color.black.opacity(0.1))
            .cornerRadius(8)
            
            // Curve presets
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(CurvePreset.allPresets) { preset in
                        Button(preset.name) {
                            curvePoints = preset.points
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
                .padding(.horizontal)
            }
            
            // Manual input
            VStack(alignment: .leading, spacing: 8) {
                Text("Manual Input")
                    .font(.subheadline.bold())
                
                HStack {
                    Text("Input:")
                        .frame(width: 50, alignment: .leading)
                    TextField("0", text: .constant("128"))
                        .textFieldStyle(.roundedBorder)
                    
                    Text("Output:")
                        .frame(width: 50, alignment: .leading)
                    TextField("0", text: .constant("128"))
                        .textFieldStyle(.roundedBorder)
                }
                .font(.caption)
            }
        }
    }
}

// MARK: - Curve Editor

struct CurveEditor: View {
    @Binding var points: [CGPoint]
    let curveColor: Color
    
    @State private var selectedPointIndex: Int?
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Grid background
                CurveGrid()
                
                // Histogram background (simplified)
                HistogramBackground()
                
                // Curve path
                Path { path in
                    let scaledPoints = points.map { point in
                        CGPoint(
                            x: point.x * geometry.size.width,
                            y: (1 - point.y) * geometry.size.height
                        )
                    }
                    
                    if scaledPoints.count >= 2 {
                        path.move(to: scaledPoints[0])
                        for i in 1..<scaledPoints.count {
                            path.addLine(to: scaledPoints[i])
                        }
                    }
                }
                .stroke(curveColor, lineWidth: 2)
                
                // Control points
                ForEach(0..<points.count, id: \.self) { index in
                    Circle()
                        .fill(selectedPointIndex == index ? .white : curveColor)
                        .stroke(.black, lineWidth: 1)
                        .frame(width: 8, height: 8)
                        .position(
                            x: points[index].x * geometry.size.width,
                            y: (1 - points[index].y) * geometry.size.height
                        )
                        .gesture(
                            DragGesture()
                                .onChanged { drag in
                                    selectedPointIndex = index
                                    let newPoint = CGPoint(
                                        x: max(0, min(1, drag.location.x / geometry.size.width)),
                                        y: max(0, min(1, 1 - drag.location.y / geometry.size.height))
                                    )
                                    points[index] = newPoint
                                }
                                .onEnded { _ in
                                    selectedPointIndex = nil
                                }
                        )
                }
            }
            .gesture(
                TapGesture()
                    .onEnded { _ in
                        // Add new point on tap
                    }
            )
        }
    }
}

struct CurveGrid: View {
    var body: some View {
        Canvas { context, size in
            let gridSpacing: CGFloat = size.width / 4
            
            // Vertical lines
            for i in 1..<4 {
                let x = CGFloat(i) * gridSpacing
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: x, y: 0))
                        path.addLine(to: CGPoint(x: x, y: size.height))
                    },
                    with: .color(.secondary.opacity(0.3)),
                    lineWidth: 0.5
                )
            }
            
            // Horizontal lines
            for i in 1..<4 {
                let y = CGFloat(i) * gridSpacing
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: size.width, y: y))
                    },
                    with: .color(.secondary.opacity(0.3)),
                    lineWidth: 0.5
                )
            }
        }
    }
}

struct HistogramBackground: View {
    var body: some View {
        // Simplified histogram background
        Canvas { context, size in
            let barWidth = size.width / 256
            
            for i in 0..<256 {
                let height = CGFloat.random(in: 0...size.height * 0.3)
                let x = CGFloat(i) * barWidth
                let y = size.height - height
                
                context.fill(
                    Rectangle().path(in: CGRect(x: x, y: y, width: barWidth, height: height)),
                    with: .color(.secondary.opacity(0.1))
                )
            }
        }
    }
}

// MARK: - Curve Presets

struct CurvePreset: Identifiable {
    let id = UUID()
    let name: String
    let points: [CGPoint]
    
    static let allPresets: [CurvePreset] = [
        CurvePreset(name: "Linear", points: [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 1)]),
        CurvePreset(name: "S-Curve", points: [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 0.25, y: 0.2),
            CGPoint(x: 0.75, y: 0.8),
            CGPoint(x: 1, y: 1)
        ]),
        CurvePreset(name: "Brighten", points: [
            CGPoint(x: 0, y: 0.1),
            CGPoint(x: 0.5, y: 0.6),
            CGPoint(x: 1, y: 1)
        ]),
        CurvePreset(name: "Darken", points: [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 0.5, y: 0.4),
            CGPoint(x: 1, y: 0.9)
        ]),
        CurvePreset(name: "High Contrast", points: [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 0.2, y: 0.1),
            CGPoint(x: 0.8, y: 0.9),
            CGPoint(x: 1, y: 1)
        ])
    ]
}

// MARK: - HSL Qualifier Section

struct HSLQualifierSection: View {
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    
    @State private var hueRange: ClosedRange<Double> = 0...360
    @State private var saturationRange: ClosedRange<Double> = 0...100
    @State private var luminanceRange: ClosedRange<Double> = 0...100
    @State private var softness: Double = 10
    
    var body: some View {
        VStack(spacing: 16) {
            Text("HSL Qualifiers")
                .font(.headline)
            
            // Hue qualifier
            HSLRangeControl(
                title: "Hue",
                range: $hueRange,
                overallRange: 0...360,
                unit: "°",
                color: .red
            )
            
            // Saturation qualifier
            HSLRangeControl(
                title: "Saturation",
                range: $saturationRange,
                overallRange: 0...100,
                unit: "%",
                color: .green
            )
            
            // Luminance qualifier
            HSLRangeControl(
                title: "Luminance",
                range: $luminanceRange,
                overallRange: 0...100,
                unit: "%",
                color: .blue
            )
            
            // Softness
            VStack(alignment: .leading, spacing: 8) {
                Text("Edge Softness")
                    .font(.subheadline.bold())
                
                HStack {
                    Slider(value: $softness, in: 0...50)
                    Text("\(Int(softness))%")
                        .monospacedDigit()
                        .frame(width: 40)
                }
            }
            
            // Qualifier preview
            HSLQualifierPreview(
                hueRange: hueRange,
                saturationRange: saturationRange,
                luminanceRange: luminanceRange
            )
            .frame(height: 100)
            .cornerRadius(8)
        }
    }
}

struct HSLRangeControl: View {
    let title: String
    @Binding var range: ClosedRange<Double>
    let overallRange: ClosedRange<Double>
    let unit: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline.bold())
            
            // Range slider (simplified)
            HStack {
                Text("\(Int(range.lowerBound))\(unit)")
                    .font(.caption.monospacedDigit())
                    .frame(width: 50)
                
                ZStack {
                    Rectangle()
                        .fill(color.opacity(0.2))
                        .frame(height: 20)
                    
                    Rectangle()
                        .fill(color)
                        .frame(height: 20)
                        .scaleEffect(
                            x: (range.upperBound - range.lowerBound) / (overallRange.upperBound - overallRange.lowerBound),
                            anchor: .leading
                        )
                        .offset(
                            x: (range.lowerBound - overallRange.lowerBound) / (overallRange.upperBound - overallRange.lowerBound) * 200
                        )
                }
                .frame(width: 200)
                .cornerRadius(4)
                
                Text("\(Int(range.upperBound))\(unit)")
                    .font(.caption.monospacedDigit())
                    .frame(width: 50)
            }
        }
    }
}

struct HSLQualifierPreview: View {
    let hueRange: ClosedRange<Double>
    let saturationRange: ClosedRange<Double>
    let luminanceRange: ClosedRange<Double>
    
    var body: some View {
        // Simplified preview showing selected HSL range
        Rectangle()
            .fill(
                LinearGradient(
                    colors: [.clear, .white, .clear],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .overlay(
                Text("HSL Selection Preview")
                    .foregroundColor(.secondary)
            )
    }
}

// MARK: - LUT Section

struct LUTSection: View {
    @ObservedObject var effectsProcessor: VideoEffectsProcessor
    @State private var selectedLUT: String = "None"
    @State private var lutIntensity: Double = 1.0
    @State private var showLUTBrowser = false
    
    let builtInLUTs = [
        "None", "Cinematic", "Warm", "Cool", "Vintage", "Black & White", "Sepia"
    ]
    
    var body: some View {
        VStack(spacing: 16) {
            Text("Lookup Tables (LUT)")
                .font(.headline)
            
            // LUT selection
            VStack(alignment: .leading, spacing: 8) {
                Text("Current LUT")
                    .font(.subheadline.bold())
                
                Picker("LUT", selection: $selectedLUT) {
                    ForEach(builtInLUTs, id: \.self) { lut in
                        Text(lut).tag(lut)
                    }
                }
                .pickerStyle(.menu)
                
                Button("Load Custom LUT...") {
                    showLUTBrowser = true
                }
                .buttonStyle(.bordered)
            }
            
            // Intensity
            VStack(alignment: .leading, spacing: 8) {
                Text("Intensity")
                    .font(.subheadline.bold())
                
                HStack {
                    Slider(value: $lutIntensity, in: 0...1)
                    Text("\(Int(lutIntensity * 100))%")
                        .monospacedDigit()
                        .frame(width: 40)
                }
            }
            
            // LUT preview grid
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3)) {
                ForEach(builtInLUTs.dropFirst(), id: \.self) { lutName in
                    LUTPreviewCard(
                        name: lutName,
                        isSelected: selectedLUT == lutName
                    ) {
                        selectedLUT = lutName
                    }
                }
            }
        }
        .fileImporter(
            isPresented: $showLUTBrowser,
            allowedContentTypes: [.data],
            allowsMultipleSelection: false
        ) { result in
            // Handle LUT file import
        }
    }
}

struct LUTPreviewCard: View {
    let name: String
    let isSelected: Bool
    let onSelect: () -> Void
    
    var body: some View {
        VStack(spacing: 4) {
            Rectangle()
                .fill(lutGradient)
                .frame(height: 60)
                .cornerRadius(4)
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
                )
            
            Text(name)
                .font(.caption)
                .lineLimit(1)
        }
        .onTapGesture {
            onSelect()
        }
    }
    
    private var lutGradient: LinearGradient {
        // Different gradient for each LUT type
        switch name {
        case "Cinematic":
            return LinearGradient(colors: [.orange, .blue], startPoint: .topLeading, endPoint: .bottomTrailing)
        case "Warm":
            return LinearGradient(colors: [.yellow, .red], startPoint: .topLeading, endPoint: .bottomTrailing)
        case "Cool":
            return LinearGradient(colors: [.blue, .cyan], startPoint: .topLeading, endPoint: .bottomTrailing)
        case "Vintage":
            return LinearGradient(colors: [.brown, .yellow], startPoint: .topLeading, endPoint: .bottomTrailing)
        case "Black & White":
            return LinearGradient(colors: [.black, .white], startPoint: .topLeading, endPoint: .bottomTrailing)
        case "Sepia":
            return LinearGradient(colors: [.brown, .yellow], startPoint: .topLeading, endPoint: .bottomTrailing)
        default:
            return LinearGradient(colors: [.gray], startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }
}

// MARK: - Scopes Section

struct ScopesSection: View {
    @Binding var showVectorscope: Bool
    @Binding var showWaveform: Bool
    @StateObject private var scopesManager = VideoScopesManager()
    
    var body: some View {
        VStack(spacing: 16) {
            Text("Video Scopes")
                .font(.headline)
            
            // Scope toggles
            HStack {
                Toggle("Vectorscope", isOn: $showVectorscope)
                Toggle("Waveform", isOn: $showWaveform)
            }
            .toggleStyle(.button)
            
            // Vectorscope
            if showVectorscope {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Vectorscope")
                        .font(.subheadline.bold())
                    
                    VectorscopeView(data: scopesManager.vectorscopeData)
                        .frame(height: 200)
                        .background(Color.black)
                        .cornerRadius(8)
                }
            }
            
            // Waveform
            if showWaveform {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Waveform Monitor")
                        .font(.subheadline.bold())
                    
                    WaveformScope(data: scopesManager.waveformData)
                        .frame(height: 150)
                        .background(Color.black)
                        .cornerRadius(8)
                }
            }
            
            // Scope settings
            GroupBox("Scope Settings") {
                VStack(spacing: 8) {
                    Toggle("Show IRE Scale", isOn: $scopesManager.showIRE)
                    Toggle("Show 75% Line", isOn: $scopesManager.show75Percent)
                    
                    HStack {
                        Text("Opacity")
                        Slider(value: $scopesManager.scopeOpacity, in: 0.3...1.0)
                        Text("\(Int(scopesManager.scopeOpacity * 100))%")
                            .monospacedDigit()
                            .frame(width: 40)
                    }
                }
            }
        }
        .onAppear {
            // Start scope analysis
        }
        .onDisappear {
            scopesManager.stopAnalyzing()
        }
    }
}