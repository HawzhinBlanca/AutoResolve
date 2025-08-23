// AUTORESOLVE V3.2 - PROFESSIONAL COLOR GRADING
// DaVinci Resolve Quality Color Correction with Wheels, Curves, and Node Graph

import SwiftUI
import CoreImage

// MARK: - PROFESSIONAL COLOR GRADING PANEL
struct ProfessionalColorGrading: View {
    @EnvironmentObject var store: UnifiedStore
    @StateObject private var colorEngine = ColorGradingEngine()
    @State private var selectedMode: ColorMode = .wheels
    @State private var selectedNode: ColorNode? = nil
    @State private var selectedLUT: String? = nil
    
    var body: some View {
        VStack(spacing: 0) {
            // COLOR GRADING TOOLBAR
            ColorGradingToolbar(
                selectedMode: $selectedMode,
                selectedLUT: $selectedLUT,
                onReset: { colorEngine.resetAll() },
                onAddNode: { colorEngine.addNode() }
            )
            
            Divider()
            
            HStack(spacing: 0) {
                // COLOR GRADING CONTROLS
                VStack(spacing: 0) {
                    switch selectedMode {
                    case .wheels:
                        ColorWheelsPanel(engine: colorEngine)
                    case .curves:
                        ColorCurvesPanel(engine: colorEngine)
                    case .hsl:
                        HSLPanel(engine: colorEngine)
                    case .nodes:
                        NodeGraphPanel(engine: colorEngine, selectedNode: $selectedNode)
                    }
                }
                .frame(maxWidth: .infinity)
                
                Divider()
                
                // COLOR INFORMATION PANEL
                VStack(spacing: 12) {
                    Text("COLOR INFO")
                        .font(.system(.caption, weight: .bold))
                        .foregroundColor(.secondary)
                    
                    ColorInformationDisplay(engine: colorEngine)
                    
                    Spacer()
                    
                    // LUT BROWSER
                    LUTBrowser(selectedLUT: $selectedLUT, onApplyLUT: colorEngine.applyLUT)
                }
                .frame(width: 200)
                .padding()
                .background(.ultraThinMaterial)
            }
        }
        .background(.black.opacity(0.05))
    }
}

// MARK: - COLOR GRADING TOOLBAR
struct ColorGradingToolbar: View {
    @Binding var selectedMode: ColorMode
    @Binding var selectedLUT: String?
    let onReset: () -> Void
    let onAddNode: () -> Void
    
    var body: some View {
        HStack(spacing: 16) {
            // COLOR MODE SELECTOR
            HStack(spacing: 2) {
                ForEach(ColorMode.allCases, id: \.self) { mode in
                    Button(mode.displayName) {
                        selectedMode = mode
                    }
                    .font(.system(.caption, weight: .medium))
                    .foregroundColor(selectedMode == mode ? .white : .secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        selectedMode == mode ?
                            .blue : .clear,
                        in: RoundedRectangle(cornerRadius: 6)
                    )
                }
            }
            .padding(.horizontal, 8)
            .background(.thickMaterial, in: RoundedRectangle(cornerRadius: 8))
            
            Spacer()
            
            // ACTION BUTTONS
            HStack(spacing: 8) {
                if selectedMode == .nodes {
                    Button(action: onAddNode) {
                        HStack(spacing: 4) {
                            Image(systemName: "plus.circle")
                            Text("Add Node")
                        }
                        .font(.caption)
                    }
                    .buttonStyle(.bordered)
                }
                
                Button("Reset All") {
                    onReset()
                }
                .font(.caption)
                .buttonStyle(.bordered)
                
                if selectedLUT != nil {
                    Button("Remove LUT") {
                        selectedLUT = nil
                    }
                    .font(.caption)
                    .buttonStyle(.bordered)
                }
            }
        }
        .padding()
        .background(.thickMaterial)
    }
}

// MARK: - COLOR WHEELS PANEL
struct ColorWheelsPanel: View {
    @ObservedObject var engine: ColorGradingEngine
    
    var body: some View {
        VStack(spacing: 20) {
            Text("LIFT • GAMMA • GAIN")
                .font(.system(.headline, weight: .bold))
                .foregroundColor(.primary)
            
            HStack(spacing: 40) {
                // LIFT (Shadows)
                VStack(spacing: 12) {
                    Text("LIFT")
                        .font(.system(.caption, weight: .bold))
                        .foregroundColor(.secondary)
                    
                    ColorWheel(
                        value: $engine.lift,
                        wheelSize: 120,
                        title: "Shadows"
                    )
                    
                    // Luminance control
                    VStack(spacing: 4) {
                        Text("LUMA")
                            .font(.system(.caption2, weight: .bold))
                            .foregroundColor(.secondary)
                        
                        Slider(value: $engine.liftLuma, in: -1...1)
                            .frame(width: 100)
                        
                        Text(String(format: "%.2f", engine.liftLuma))
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                }
                
                // GAMMA (Midtones)
                VStack(spacing: 12) {
                    Text("GAMMA")
                        .font(.system(.caption, weight: .bold))
                        .foregroundColor(.secondary)
                    
                    ColorWheel(
                        value: $engine.gamma,
                        wheelSize: 120,
                        title: "Midtones"
                    )
                    
                    VStack(spacing: 4) {
                        Text("LUMA")
                            .font(.system(.caption2, weight: .bold))
                            .foregroundColor(.secondary)
                        
                        Slider(value: $engine.gammaLuma, in: -1...1)
                            .frame(width: 100)
                        
                        Text(String(format: "%.2f", engine.gammaLuma))
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                }
                
                // GAIN (Highlights)
                VStack(spacing: 12) {
                    Text("GAIN")
                        .font(.system(.caption, weight: .bold))
                        .foregroundColor(.secondary)
                    
                    ColorWheel(
                        value: $engine.gain,
                        wheelSize: 120,
                        title: "Highlights"
                    )
                    
                    VStack(spacing: 4) {
                        Text("LUMA")
                            .font(.system(.caption2, weight: .bold))
                            .foregroundColor(.secondary)
                        
                        Slider(value: $engine.gainLuma, in: -1...1)
                            .frame(width: 100)
                        
                        Text(String(format: "%.2f", engine.gainLuma))
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Divider()
            
            // GLOBAL CONTROLS
            HStack(spacing: 40) {
                VStack {
                    Text("EXPOSURE")
                        .font(.system(.caption, weight: .bold))
                        .foregroundColor(.secondary)
                    
                    Slider(value: $engine.exposure, in: -3...3)
                        .frame(width: 150)
                    
                    Text(String(format: "%.2f", engine.exposure))
                        .font(.system(.caption2, design: .monospaced))
                }
                
                VStack {
                    Text("CONTRAST")
                        .font(.system(.caption, weight: .bold))
                        .foregroundColor(.secondary)
                    
                    Slider(value: $engine.contrast, in: -1...1)
                        .frame(width: 150)
                    
                    Text(String(format: "%.2f", engine.contrast))
                        .font(.system(.caption2, design: .monospaced))
                }
                
                VStack {
                    Text("SATURATION")
                        .font(.system(.caption, weight: .bold))
                        .foregroundColor(.secondary)
                    
                    Slider(value: $engine.saturation, in: -1...1)
                        .frame(width: 150)
                    
                    Text(String(format: "%.2f", engine.saturation))
                        .font(.system(.caption2, design: .monospaced))
                }
            }
        }
        .padding()
    }
}

// MARK: - COLOR WHEEL
struct ColorWheel: View {
    @Binding var value: CGPoint
    let wheelSize: CGFloat
    let title: String
    
    @State private var isDragging = false
    
    var body: some View {
        ZStack {
            // Color wheel background
            Circle()
                .fill(
                    AngularGradient(
                        colors: [
                            .red, .yellow, .green, .cyan, .blue, .purple, .red
                        ],
                        center: .center
                    )
                )
                .frame(width: wheelSize, height: wheelSize)
                .overlay(
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [.white.opacity(0.8), .clear],
                                center: .center,
                                startRadius: 0,
                                endRadius: wheelSize / 2
                            )
                        )
                )
                .overlay(
                    Circle()
                        .stroke(.black.opacity(0.2), lineWidth: 1)
                )
            
            // Center point
            Circle()
                .fill(.white)
                .frame(width: 4, height: 4)
            
            // Control point
            Circle()
                .fill(.white)
                .frame(width: 12, height: 12)
                .shadow(radius: 2)
                .overlay(
                    Circle()
                        .stroke(.black.opacity(0.3), lineWidth: 1)
                )
                .offset(
                    x: value.x * wheelSize / 2,
                    y: value.y * wheelSize / 2
                )
                .scaleEffect(isDragging ? 1.2 : 1.0)
                .animation(.easeInOut(duration: 0.1), value: isDragging)
        }
        .gesture(
            DragGesture(coordinateSpace: .local)
                .onChanged { drag in
                    isDragging = true
                    let center = CGPoint(x: wheelSize / 2, y: wheelSize / 2)
                    let offset = CGPoint(
                        x: drag.location.x - center.x,
                        y: drag.location.y - center.y
                    )
                    let distance = sqrt(offset.x * offset.x + offset.y * offset.y)
                    let maxDistance = wheelSize / 2
                    
                    if distance <= maxDistance {
                        value = CGPoint(
                            x: offset.x / maxDistance,
                            y: offset.y / maxDistance
                        )
                    } else {
                        // Constrain to circle edge
                        let angle = atan2(offset.y, offset.x)
                        value = CGPoint(
                            x: cos(angle),
                            y: sin(angle)
                        )
                    }
                }
                .onEnded { _ in
                    isDragging = false
                }
        )
    }
}

// MARK: - COLOR CURVES PANEL
struct ColorCurvesPanel: View {
    @ObservedObject var engine: ColorGradingEngine
    @State private var selectedCurve: CurveType = .master
    
    var body: some View {
        VStack(spacing: 16) {
            // Curve selector
            HStack {
                ForEach(CurveType.allCases, id: \.self) { curve in
                    Button(curve.displayName) {
                        selectedCurve = curve
                    }
                    .foregroundColor(selectedCurve == curve ? .white : .secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        selectedCurve == curve ? curve.color : .clear,
                        in: RoundedRectangle(cornerRadius: 6)
                    )
                }
            }
            
            // Curve display
            ColorCurveEditor(
                points: bindingForCurve(selectedCurve),
                curveColor: selectedCurve.color,
                backgroundColor: .black.opacity(0.1)
            )
            .frame(height: 300)
            .background(.black.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
        }
        .padding()
    }
    
    private func bindingForCurve(_ curve: CurveType) -> Binding<[CGPoint]> {
        switch curve {
        case .master: return $engine.masterCurve
        case .red: return $engine.redCurve
        case .green: return $engine.greenCurve
        case .blue: return $engine.blueCurve
        }
    }
}

// MARK: - COLOR CURVE EDITOR
struct ColorCurveEditor: View {
    @Binding var points: [CGPoint]
    let curveColor: Color
    let backgroundColor: Color
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background grid
                Path { path in
                    let gridSize: CGFloat = 20
                    for x in stride(from: 0, to: geometry.size.width, by: gridSize) {
                        path.move(to: CGPoint(x: x, y: 0))
                        path.addLine(to: CGPoint(x: x, y: geometry.size.height))
                    }
                    for y in stride(from: 0, to: geometry.size.height, by: gridSize) {
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: geometry.size.width, y: y))
                    }
                }
                .stroke(.gray.opacity(0.2), lineWidth: 0.5)
                
                // Curve path
                Path { path in
                    if points.count >= 2 {
                        let first = points[0]
                        path.move(to: CGPoint(
                            x: first.x * geometry.size.width,
                            y: (1 - first.y) * geometry.size.height
                        ))
                        
                        for point in points.dropFirst() {
                            path.addLine(to: CGPoint(
                                x: point.x * geometry.size.width,
                                y: (1 - point.y) * geometry.size.height
                            ))
                        }
                    }
                }
                .stroke(curveColor, lineWidth: 2)
                
                // Control points
                ForEach(Array(points.enumerated()), id: \.offset) { index, point in
                    Circle()
                        .fill(curveColor)
                        .frame(width: 8, height: 8)
                        .position(
                            x: point.x * geometry.size.width,
                            y: (1 - point.y) * geometry.size.height
                        )
                        .gesture(
                            DragGesture()
                                .onChanged { drag in
                                    let newX = max(0, min(1, drag.location.x / geometry.size.width))
                                    let newY = max(0, min(1, 1 - drag.location.y / geometry.size.height))
                                    points[index] = CGPoint(x: newX, y: newY)
                                }
                        )
                }
            }
        }
        .onAppear {
            if points.isEmpty {
                points = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 1)]
            }
        }
    }
}

// MARK: - ENUMS AND TYPES
enum ColorMode: CaseIterable {
    case wheels, curves, hsl, nodes
    
    var displayName: String {
        switch self {
        case .wheels: return "Wheels"
        case .curves: return "Curves"
        case .hsl: return "HSL"
        case .nodes: return "Nodes"
        }
    }
}

enum CurveType: CaseIterable {
    case master, red, green, blue
    
    var displayName: String {
        switch self {
        case .master: return "Master"
        case .red: return "Red"
        case .green: return "Green"
        case .blue: return "Blue"
        }
    }
    
    var color: Color {
        switch self {
        case .master: return .white
        case .red: return .red
        case .green: return .green
        case .blue: return .blue
        }
    }
}

// MARK: - COLOR GRADING ENGINE
class ColorGradingEngine: ObservableObject {
    // Color wheels
    @Published var lift = CGPoint.zero
    @Published var gamma = CGPoint.zero
    @Published var gain = CGPoint.zero
    @Published var liftLuma: Double = 0
    @Published var gammaLuma: Double = 0
    @Published var gainLuma: Double = 0
    
    // Global controls
    @Published var exposure: Double = 0
    @Published var contrast: Double = 0
    @Published var saturation: Double = 0
    
    // Curves
    @Published var masterCurve: [CGPoint] = []
    @Published var redCurve: [CGPoint] = []
    @Published var greenCurve: [CGPoint] = []
    @Published var blueCurve: [CGPoint] = []
    
    // Nodes
    @Published var nodes: [ColorNode] = []
    
    func resetAll() {
        lift = .zero
        gamma = .zero
        gain = .zero
        liftLuma = 0
        gammaLuma = 0
        gainLuma = 0
        exposure = 0
        contrast = 0
        saturation = 0
        
        masterCurve = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 1)]
        redCurve = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 1)]
        greenCurve = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 1)]
        blueCurve = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 1)]
    }
    
    func addNode() {
        let newNode = ColorNode(
            id: UUID(),
            name: "Node \(nodes.count + 1)",
            position: CGPoint(x: 100, y: 100),
            isEnabled: true
        )
        nodes.append(newNode)
    }
    
    func applyLUT(_ lutName: String) {
        // LUT application logic
        print("Applying LUT: \(lutName)")
    }
}

// MARK: - COLOR NODE
struct ColorNode: Identifiable, Equatable {
    let id: UUID
    var name: String
    var position: CGPoint
    var isEnabled: Bool
}

// MARK: - HSL PANEL
struct HSLPanel: View {
    @ObservedObject var engine: ColorGradingEngine
    
    var body: some View {
        VStack {
            Text("HSL (Hue, Saturation, Lightness)")
                .font(.headline)
                .padding()
            
            // HSL controls would go here
            Rectangle()
                .fill(.gray.opacity(0.1))
                .frame(height: 400)
                .overlay(
                    Text("HSL Qualifiers & Controls\n(Would contain color range selectors)")
                        .multilineTextAlignment(.center)
                        .foregroundColor(.secondary)
                )
        }
        .padding()
    }
}

// MARK: - NODE GRAPH PANEL
struct NodeGraphPanel: View {
    @ObservedObject var engine: ColorGradingEngine
    @Binding var selectedNode: ColorNode?
    
    var body: some View {
        VStack {
            Text("Node Graph")
                .font(.headline)
                .padding()
            
            // Node graph would go here
            Rectangle()
                .fill(.gray.opacity(0.1))
                .frame(height: 400)
                .overlay(
                    VStack {
                        Text("Node-Based Color Grading")
                            .font(.title3)
                        Text("(Would contain node graph interface)")
                            .foregroundColor(.secondary)
                        
                        Button("Add Node") {
                            engine.addNode()
                        }
                        .padding()
                    }
                )
        }
        .padding()
    }
}

// MARK: - COLOR INFORMATION DISPLAY
struct ColorInformationDisplay: View {
    @ObservedObject var engine: ColorGradingEngine
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("RGB")
                .font(.system(.caption, weight: .bold))
                .foregroundColor(.secondary)
            
            HStack {
                Text("R: 127")
                    .font(.system(.caption2, design: .monospaced))
                Spacer()
            }
            HStack {
                Text("G: 127")
                    .font(.system(.caption2, design: .monospaced))
                Spacer()
            }
            HStack {
                Text("B: 127")
                    .font(.system(.caption2, design: .monospaced))
                Spacer()
            }
            
            Divider()
            
            Text("HSV")
                .font(.system(.caption, weight: .bold))
                .foregroundColor(.secondary)
            
            HStack {
                Text("H: 180°")
                    .font(.system(.caption2, design: .monospaced))
                Spacer()
            }
            HStack {
                Text("S: 50%")
                    .font(.system(.caption2, design: .monospaced))
                Spacer()
            }
            HStack {
                Text("V: 50%")
                    .font(.system(.caption2, design: .monospaced))
                Spacer()
            }
        }
        .padding()
        .background(.thickMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - LUT BROWSER
struct LUTBrowser: View {
    @Binding var selectedLUT: String?
    let onApplyLUT: (String) -> Void
    
    private let availableLUTs = [
        "Rec709 to DCI-P3",
        "Film Emulation",
        "Vintage Look",
        "High Contrast",
        "Desaturated",
        "Warm Tone",
        "Cool Tone"
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("LUT BROWSER")
                .font(.system(.caption, weight: .bold))
                .foregroundColor(.secondary)
            
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(availableLUTs, id: \.self) { lut in
                        Button(lut) {
                            selectedLUT = lut
                            onApplyLUT(lut)
                        }
                        .font(.system(.caption2))
                        .foregroundColor(selectedLUT == lut ? .blue : .primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 2)
                    }
                }
            }
            .frame(maxHeight: 200)
        }
        .padding()
        .background(.thickMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}