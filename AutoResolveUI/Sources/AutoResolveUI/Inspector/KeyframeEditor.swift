import SwiftUI
import Combine

// MARK: - Professional Keyframe Editor

public struct KeyframeEditor: View {
    @ObservedObject var timeline: TimelineModel
    @State private var selectedProperty: UIAnimatableProperty?
    @State private var selectedKeyframes: Set<UUID> = []
    @State private var zoomLevel: Double = 1.0
    @State private var timelineOffset: Double = 0
    @State private var showOnlyAnimated = false
    @State private var interpolationMode: InterpolationMode = .bezier
    
    private let keyframeHeight: CGFloat = 40
    private let headerWidth: CGFloat = 200
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            KeyframeEditorHeader(
                showOnlyAnimated: $showOnlyAnimated,
                interpolationMode: $interpolationMode,
                selectedKeyframes: selectedKeyframes
            )
            
            Divider()
            
            // Main content
            HStack(spacing: 0) {
                // Property list
                PropertyListView(
                    properties: animatableProperties,
                    selectedProperty: $selectedProperty,
                    showOnlyAnimated: showOnlyAnimated
                )
                .frame(width: headerWidth)
                
                Divider()
                
                // Keyframe timeline
                KeyframeTimelineView(
                    properties: animatableProperties,
                    selectedProperty: $selectedProperty,
                    selectedKeyframes: $selectedKeyframes,
                    zoomLevel: $zoomLevel,
                    timelineOffset: $timelineOffset,
                    interpolationMode: interpolationMode,
                    timeline: timeline
                )
            }
            
            Divider()
            
            // Controls
            KeyframeControls(
                selectedProperty: selectedProperty,
                selectedKeyframes: selectedKeyframes,
                interpolationMode: $interpolationMode,
                timeline: timeline
            )
        }
        .frame(width: 800, height: 500)
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private var animatableProperties: [UIAnimatableProperty] {
        // Get all animatable properties from selected clips
        guard let clipId = timeline.selectedClips.first,
              let clip = timeline.tracks.flatMap({ $0.clips }).first(where: { $0.id == clipId }) else {
            return []
        }
        
        return UIAnimatableProperty.propertiesForClip(clip)
    }
}

// MARK: - Keyframe Editor Header

struct KeyframeEditorHeader: View {
    @Binding var showOnlyAnimated: Bool
    @Binding var interpolationMode: InterpolationMode
    let selectedKeyframes: Set<UUID>
    
    public var body: some View {
        HStack {
            Text("Keyframe Editor")
                .font(.headline)
            
            Spacer()
            
            // Filter toggle
            Toggle("Animated Only", isOn: $showOnlyAnimated)
                .toggleStyle(.button)
                .controlSize(.small)
            
            // Interpolation mode
            Picker("Interpolation", selection: $interpolationMode) {
                ForEach(InterpolationMode.allCases, id: \.self) { mode in
                    Label(mode.rawValue, systemImage: mode.icon)
                        .tag(mode)
                }
            }
            .pickerStyle(.menu)
            .disabled(selectedKeyframes.isEmpty)
            
            // Keyframe actions
            HStack(spacing: 4) {
                Button(action: addKeyframe) {
                    Image(systemName: "plus")
                }
                .help("Add Keyframe")
                
                Button(action: deleteSelectedKeyframes) {
                    Image(systemName: "trash")
                }
                .disabled(selectedKeyframes.isEmpty)
                .help("Delete Selected Keyframes")
                
                Button(action: copyKeyframes) {
                    Image(systemName: "doc.on.doc")
                }
                .disabled(selectedKeyframes.isEmpty)
                .help("Copy Keyframes")
                
                Button(action: pasteKeyframes) {
                    Image(systemName: "doc.on.clipboard")
                }
                .help("Paste Keyframes")
            }
            .buttonStyle(.plain)
        }
        .padding()
    }
    
    private func addKeyframe() {
        // Add keyframe at current playhead position
    }
    
    private func deleteSelectedKeyframes() {
        // Delete selected keyframes
    }
    
    private func copyKeyframes() {
        // Copy selected keyframes to clipboard
    }
    
    private func pasteKeyframes() {
        // Paste keyframes from clipboard
    }
}

// MARK: - Property List View

struct PropertyListView: View {
    let properties: [UIAnimatableProperty]
    @Binding var selectedProperty: UIAnimatableProperty?
    let showOnlyAnimated: Bool
    
    public var body: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                ForEach(filteredProperties) { property in
                    PropertyRow(
                        property: property,
                        isSelected: selectedProperty?.id == property.id,
                        onSelect: { selectedProperty = property }
                    )
                }
            }
        }
        .background(Color(NSColor.windowBackgroundColor))
    }
    
    private var filteredProperties: [UIAnimatableProperty] {
        if showOnlyAnimated {
            return properties.filter { !$0.keyframes.isEmpty }
        }
        return properties
    }
}

// MARK: - Property Row

struct PropertyRow: View {
    let property: UIAnimatableProperty
    let isSelected: Bool
    let onSelect: () -> Void
    
    @State private var isExpanded = false
    @State private var currentValue: Double = 0
    
    public var body: some View {
        VStack(spacing: 0) {
            // Main property row
            HStack {
                // Expand/collapse button
                if property.hasSubProperties {
                    Button(action: { isExpanded.toggle() }) {
                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                } else {
                    Spacer()
                        .frame(width: 16)
                }
                
                // Property icon
                Image(systemName: property.icon)
                    .foregroundColor(property.hasKeyframes ? .accentColor : .secondary)
                    .frame(width: 20)
                
                // Property name
                Text(property.name)
                    .font(.caption)
                    .lineLimit(1)
                
                Spacer()
                
                // Current value
                Text(property.formattedValue(currentValue))
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .frame(height: 28)
            .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
            .onTapGesture {
                onSelect()
            }
            
            // Sub-properties
            if isExpanded && property.hasSubProperties {
                ForEach(property.subProperties) { subProperty in
                    PropertyRow(
                        property: subProperty,
                        isSelected: false,
                        onSelect: {}
                    )
                    .padding(.leading, 20)
                }
            }
        }
    }
}

// MARK: - Keyframe Timeline View

struct KeyframeTimelineView: View {
    let properties: [UIAnimatableProperty]
    @Binding var selectedProperty: UIAnimatableProperty?
    @Binding var selectedKeyframes: Set<UUID>
    @Binding var zoomLevel: Double
    @Binding var timelineOffset: Double
    let interpolationMode: InterpolationMode
    @ObservedObject var timeline: TimelineModel
    
    @State private var dragStartPosition: CGPoint?
    @State private var isDragging = false
    
    private let keyframeHeight: CGFloat = 40
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background
                Color(NSColor.textBackgroundColor)
                
                // Time grid
                TimeGrid(
                    width: geometry.size.width,
                    height: geometry.size.height,
                    zoomLevel: zoomLevel,
                    offset: timelineOffset,
                    timeline: timeline
                )
                
                // Property keyframe tracks
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(Array(properties.enumerated()), id: \.element.id) { index, property in
                            KeyframeTrackView(
                                property: property,
                                width: geometry.size.width,
                                isSelected: selectedProperty?.id == property.id,
                                selectedKeyframes: $selectedKeyframes,
                                zoomLevel: zoomLevel,
                                offset: timelineOffset,
                                interpolationMode: interpolationMode,
                                timeline: timeline
                            )
                            .frame(height: keyframeHeight)
                        }
                    }
                }
                
                // Playhead
                PlayheadIndicator(
                    position: timeline.playheadPosition,
                    width: geometry.size.width,
                    height: geometry.size.height,
                    zoomLevel: zoomLevel,
                    offset: timelineOffset
                )
                
                // Selection rectangle
                if isDragging, let startPos = dragStartPosition {
                    SelectionRectangle(
                        start: startPos,
                        end: CGPoint(x: 0, y: 0) // Updated during drag
                    )
                }
            }
            .gesture(
                DragGesture()
                    .onChanged { drag in
                        if !isDragging {
                            isDragging = true
                            dragStartPosition = drag.startLocation
                        }
                        // Update selection
                    }
                    .onEnded { _ in
                        isDragging = false
                        dragStartPosition = nil
                    }
            )
        }
    }
}

// MARK: - Time Grid

struct TimeGrid: View {
    let width: CGFloat
    let height: CGFloat
    let zoomLevel: Double
    let offset: Double
    @ObservedObject var timeline: TimelineModel
    
    public var body: some View {
        Canvas { context, size in
            let pixelsPerSecond = zoomLevel * 50 // Base: 50 pixels per second
            let secondsVisible = size.width / pixelsPerSecond
            let startTime = offset
            let endTime = startTime + secondsVisible
            
            // Determine grid interval
            let gridInterval = calculateGridInterval(secondsVisible: secondsVisible)
            
            // Draw vertical grid lines
            var time = ceil(startTime / gridInterval) * gridInterval
            while time <= endTime {
                let x = CGFloat((time - startTime) * pixelsPerSecond)
                
                let isMajor = Int(time) % Int(gridInterval * 4) == 0
                let lineWidth: CGFloat = isMajor ? 1.0 : 0.5
                let opacity: Double = isMajor ? 0.3 : 0.15
                
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: x, y: 0))
                        path.addLine(to: CGPoint(x: x, y: size.height))
                    },
                    with: .color(.secondary.opacity(opacity)),
                    lineWidth: lineWidth
                )
                
                // Time labels for major lines
                if isMajor {
                    let timecode = timeline.timecode(for: time)
                    context.draw(
                        Text(timecode)
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary),
                        at: CGPoint(x: x + 4, y: 12)
                    )
                }
                
                time += gridInterval
            }
        }
    }
    
    private func calculateGridInterval(secondsVisible: Double) -> Double {
        if secondsVisible > 600 { return 60 }      // 1 minute
        if secondsVisible > 120 { return 30 }      // 30 seconds
        if secondsVisible > 60 { return 10 }       // 10 seconds
        if secondsVisible > 20 { return 5 }        // 5 seconds
        if secondsVisible > 10 { return 1 }        // 1 second
        return 0.1  // 100ms
    }
}

// MARK: - Keyframe Track View

struct KeyframeTrackView: View {
    let property: UIAnimatableProperty
    let width: CGFloat
    let isSelected: Bool
    @Binding var selectedKeyframes: Set<UUID>
    let zoomLevel: Double
    let offset: Double
    let interpolationMode: InterpolationMode
    @ObservedObject var timeline: TimelineModel
    
    private let keyframeHeight: CGFloat = 60
    
    public var body: some View {
        ZStack(alignment: .leading) {
            // Track background
            Rectangle()
                .fill(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
            
            // Keyframes
            ForEach(property.keyframes) { keyframe in
                KeyframeView(
                    keyframe: keyframe,
                    property: property,
                    isSelected: selectedKeyframes.contains(keyframe.id),
                    interpolationMode: interpolationMode,
                    onSelect: { toggleKeyframeSelection(keyframe) },
                    onMove: { newTime in moveKeyframe(keyframe, to: newTime) }
                )
                .position(
                    x: timeToX(keyframe.time),
                    y: keyframeHeight / 2
                )
            }
            
            // Animation curves between keyframes
            if property.keyframes.count > 1 {
                AnimationCurveView(
                    keyframes: property.keyframes,
                    property: property,
                    width: width,
                    zoomLevel: zoomLevel,
                    offset: offset
                )
            }
        }
        .frame(height: keyframeHeight)
        .clipped()
    }
    
    private func timeToX(_ time: TimeInterval) -> CGFloat {
        let pixelsPerSecond = zoomLevel * 50
        return CGFloat((time - offset) * pixelsPerSecond)
    }
    
    private func toggleKeyframeSelection(_ keyframe: UIKeyframe) {
        if selectedKeyframes.contains(keyframe.id) {
            selectedKeyframes.remove(keyframe.id)
        } else {
            selectedKeyframes.insert(keyframe.id)
        }
    }
    
    private func moveKeyframe(_ keyframe: UIKeyframe, to newTime: TimeInterval) {
        // Update keyframe time
    }
}

// MARK: - Keyframe View

struct KeyframeView: View {
    let keyframe: UIKeyframe
    let property: UIAnimatableProperty
    let isSelected: Bool
    let interpolationMode: InterpolationMode
    let onSelect: () -> Void
    let onMove: (TimeInterval) -> Void
    
    @State private var isDragging = false
    
    public var body: some View {
        ZStack {
            // Keyframe diamond shape
            KeyframeDiamond(
                interpolationMode: keyframe.interpolation,
                isSelected: isSelected
            )
            .frame(width: 12, height: 12)
            
            // Value indicator
            if isSelected {
                Text(property.formattedValue(keyframe.value))
                    .font(.caption2)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .background(Color.black.opacity(0.8))
                    .foregroundColor(.white)
                    .cornerRadius(4)
                    .offset(y: -20)
            }
        }
        .onTapGesture {
            onSelect()
        }
        .gesture(
            DragGesture()
                .onChanged { drag in
                    isDragging = true
                    // Convert drag distance to time and call onMove
                }
                .onEnded { _ in
                    isDragging = false
                }
        )
        .scaleEffect(isDragging ? 1.2 : 1.0)
        .animation(.easeInOut(duration: 0.1), value: isDragging)
    }
}

// MARK: - Keyframe Diamond

struct KeyframeDiamond: View {
    let interpolationMode: InterpolationMode
    let isSelected: Bool
    
    public var body: some View {
        Path { path in
            path.move(to: CGPoint(x: 6, y: 0))
            path.addLine(to: CGPoint(x: 12, y: 6))
            path.addLine(to: CGPoint(x: 6, y: 12))
            path.addLine(to: CGPoint(x: 0, y: 6))
            path.closeSubpath()
        }
        .fill(fillColor)
        .stroke(strokeColor, lineWidth: isSelected ? 2 : 1)
    }
    
    private var fillColor: Color {
        switch interpolationMode {
        case .linear: return .blue
        case .bezier: return .green
        case .step: return .orange
        case .hold: return .red
        }
    }
    
    private var strokeColor: Color {
        isSelected ? .white : .black
    }
}

// MARK: - Animation Curve View

struct AnimationCurveView: View {
    let keyframes: [UIKeyframe]
    let property: UIAnimatableProperty
    let width: CGFloat
    let zoomLevel: Double
    let offset: Double
    
    public var body: some View {
        Canvas { context, size in
            guard keyframes.count > 1 else { return }
            
            let pixelsPerSecond = zoomLevel * 50
            let sortedKeyframes = keyframes.sorted { (a, b) in a.time < b.time }
            
            // Draw curves between keyframes
            for i in 0..<sortedKeyframes.count - 1 {
                let startKeyframe = sortedKeyframes[i]
                let endKeyframe = sortedKeyframes[i + 1]
                
                let startX = CGFloat((startKeyframe.time - offset) * pixelsPerSecond)
                let endX = CGFloat((endKeyframe.time - offset) * pixelsPerSecond)
                
                // Skip if both points are outside visible area
                guard endX >= 0 && startX <= width else { continue }
                
                let startY = valueToY(startKeyframe.value, in: size.height)
                let endY = valueToY(endKeyframe.value, in: size.height)
                
                var path = Path()
                path.move(to: CGPoint(x: startX, y: startY))
                
                switch startKeyframe.interpolation {
                case .linear:
                    path.addLine(to: CGPoint(x: endX, y: endY))
                    
                case .bezier:
                    // Smooth bezier curve
                    let controlPoint1 = CGPoint(x: startX + (endX - startX) * 0.33, y: startY)
                    let controlPoint2 = CGPoint(x: endX - (endX - startX) * 0.33, y: endY)
                    path.addCurve(to: CGPoint(x: endX, y: endY), control1: controlPoint1, control2: controlPoint2)
                    
                case .step:
                    // Step function
                    path.addLine(to: CGPoint(x: endX, y: startY))
                    path.addLine(to: CGPoint(x: endX, y: endY))
                    
                case .hold:
                    // Hold until next keyframe
                    path.addLine(to: CGPoint(x: endX, y: startY))
                }
                
                context.stroke(path, with: .color(.accentColor), lineWidth: 2)
            }
        }
    }
    
    private func valueToY(_ value: Double, in height: CGFloat) -> CGFloat {
        // Normalize value to 0-1 range and convert to Y position
        let normalizedValue = (value - property.minValue) / (property.maxValue - property.minValue)
        return height - (CGFloat(normalizedValue) * height * 0.8) - height * 0.1
    }
}

// MARK: - Playhead Indicator

struct PlayheadIndicator: View {
    let position: TimeInterval
    let width: CGFloat
    let height: CGFloat
    let zoomLevel: Double
    let offset: Double
    
    public var body: some View {
        let pixelsPerSecond = zoomLevel * 50
        let x = CGFloat((position - offset) * pixelsPerSecond)
        
        if x >= 0 && x <= width {
            Rectangle()
                .fill(Color.red)
                .frame(width: 2, height: height)
                .position(x: x, y: height / 2)
        }
    }
}

// MARK: - Keyframe Controls

struct KeyframeControls: View {
    let selectedProperty: UIAnimatableProperty?
    let selectedKeyframes: Set<UUID>
    @Binding var interpolationMode: InterpolationMode
    @ObservedObject var timeline: TimelineModel
    
    public var body: some View {
        HStack {
            // Property info
            if let property = selectedProperty {
                VStack(alignment: .leading, spacing: 2) {
                    Text(property.name)
                        .font(.caption.bold())
                    Text("\(property.keyframes.count) keyframes")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            } else {
                Text("No property selected")
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Zoom controls
            HStack {
                Button(action: { /* zoom out */ }) {
                    Image(systemName: "minus.magnifyingglass")
                }
                
                Text("100%")
                    .monospacedDigit()
                    .frame(width: 40)
                
                Button(action: { /* zoom in */ }) {
                    Image(systemName: "plus.magnifyingglass")
                }
            }
            .buttonStyle(.plain)
            
            Divider()
            
            // Keyframe tools
            HStack {
                Button("Ease In") {
                    applyEasing(.easeIn)
                }
                .disabled(selectedKeyframes.isEmpty)
                
                Button("Ease Out") {
                    applyEasing(.easeOut)
                }
                .disabled(selectedKeyframes.isEmpty)
                
                Button("Linear") {
                    applyEasing(.linear)
                }
                .disabled(selectedKeyframes.isEmpty)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding()
    }
    
    private func applyEasing(_ easing: EasingType) {
        // Apply easing to selected keyframes
    }
}

// MARK: - Selection Rectangle

struct SelectionRectangle: View {
    let start: CGPoint
    let end: CGPoint
    
    public var body: some View {
        Rectangle()
            .stroke(Color.accentColor, lineWidth: 1)
            .background(Color.accentColor.opacity(0.1))
            .frame(
                width: abs(end.x - start.x),
                height: abs(end.y - start.y)
            )
            .position(
                x: (start.x + end.x) / 2,
                y: (start.y + end.y) / 2
            )
    }
}

// MARK: - Models

public struct UIAnimatableProperty: Identifiable {
    public let id = UUID()
    public let name: String
    public let icon: String
    public let minValue: Double
    public let maxValue: Double
    public let unit: String
    public var keyframes: [UIKeyframe] = []
    public var subProperties: [UIAnimatableProperty] = []
    
    public var hasKeyframes: Bool {
        !keyframes.isEmpty || subProperties.contains { $0.hasKeyframes }
    }
    
    public var hasSubProperties: Bool {
        !subProperties.isEmpty
    }
    
    public func formattedValue(_ value: Double) -> String {
        if unit.isEmpty {
            return String(format: "%.2f", value)
        } else {
            return String(format: "%.2f%@", value, unit)
        }
    }
    
    public static func propertiesForClip(_ clip: TimelineClip) -> [UIAnimatableProperty] {
        [
            // Transform properties
            UIAnimatableProperty(
                name: "Transform",
                icon: "move.3d",
                minValue: 0,
                maxValue: 1,
                unit: "",
                subProperties: [
                    UIAnimatableProperty(name: "Position X", icon: "arrow.left.and.right", minValue: -1920, maxValue: 1920, unit: "px"),
                    UIAnimatableProperty(name: "Position Y", icon: "arrow.up.and.down", minValue: -1080, maxValue: 1080, unit: "px"),
                    UIAnimatableProperty(name: "Scale X", icon: "arrow.left.and.right.square", minValue: 0, maxValue: 500, unit: "%"),
                    UIAnimatableProperty(name: "Scale Y", icon: "arrow.up.and.down.square", minValue: 0, maxValue: 500, unit: "%"),
                    UIAnimatableProperty(name: "Rotation", icon: "rotate.right", minValue: -360, maxValue: 360, unit: "°")
                ]
            ),
            
            // Opacity
            UIAnimatableProperty(name: "Opacity", icon: "eye", minValue: 0, maxValue: 100, unit: "%"),
            
            // Audio properties
            UIAnimatableProperty(
                name: "Audio",
                icon: "waveform",
                minValue: 0,
                maxValue: 1,
                unit: "",
                subProperties: [
                    UIAnimatableProperty(name: "Volume", icon: "speaker.wave.2", minValue: -60, maxValue: 12, unit: "dB"),
                    UIAnimatableProperty(name: "Pan", icon: "speaker.wave.2.bubble.left.and.bubble.right", minValue: -100, maxValue: 100, unit: "")
                ]
            ),
            
            // Color properties
            UIAnimatableProperty(
                name: "Color",
                icon: "paintpalette",
                minValue: 0,
                maxValue: 1,
                unit: "",
                subProperties: [
                    UIAnimatableProperty(name: "Brightness", icon: "sun.max", minValue: -100, maxValue: 100, unit: "%"),
                    UIAnimatableProperty(name: "Contrast", icon: "circle.lefthalf.filled", minValue: 0, maxValue: 200, unit: "%"),
                    UIAnimatableProperty(name: "Saturation", icon: "drop.fill", minValue: 0, maxValue: 200, unit: "%"),
                    UIAnimatableProperty(name: "Hue", icon: "paintpalette", minValue: -180, maxValue: 180, unit: "°")
                ]
            )
        ]
    }
}

public struct UIKeyframe: Identifiable {
    public let id = UUID()
    public var time: TimeInterval
    public var value: Double
    public var interpolation: InterpolationMode = .bezier
    public var easeIn: Double = 0.5
    public var easeOut: Double = 0.5
}

public enum InterpolationMode: String, CaseIterable {
    case linear = "Linear"
    case bezier = "Bezier"
    case step = "Step"
    case hold = "Hold"
    
    public var icon: String {
        switch self {
        case .linear: return "line.diagonal"
        case .bezier: return "scribble.variable"
        case .step: return "stairstep"
        case .hold: return "rectangle"
        }
    }
}

public enum EasingType {
    case linear
    case easeIn
    case easeOut
    case easeInOut
}
