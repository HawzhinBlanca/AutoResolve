// AUTORESOLVE V3.0 - TIMELINE ZOOM CONTROLS
import Combine
// Professional zoom controls with keyboard shortcuts and smooth animations

import SwiftUI
import AppKit

// MARK: - Timeline Zoom Controller
struct TimelineZoomController: View {
    @ObservedObject var timeline: TimelineModel
    @State private var zoomSliderValue: Double = 50
    @State private var isZoomPopoverVisible = false
    @State private var customZoomText = ""
    @State private var zoomFocusPoint: CGFloat = 0.5
    
    private let zoomPresets: [(String, Double)] = [
        ("5 sec", 200),
        ("10 sec", 100),
        ("30 sec", 50),
        ("1 min", 25),
        ("5 min", 10),
        ("10 min", 5),
        ("Fit", 0) // Special case for fit
    ]
    
    public var body: some View {
        HStack(spacing: 12) {
            // Zoom out button
            Button(action: zoomOut) {
                Image(systemName: "minus.magnifyingglass")
                    .font(.system(size: 14))
            }
            .buttonStyle(PlainButtonStyle())
            .help("Zoom Out (⌘-)")
            .keyboardShortcut("-", modifiers: .command)
            
            // Zoom slider
            ZoomSlider(
                value: $timeline.zoomLevel,
                range: 5...200,
                onChanged: { newZoom in
                    applyZoom(newZoom, focusPoint: zoomFocusPoint)
                }
            )
            .frame(width: 120)
            
            // Zoom in button
            Button(action: zoomIn) {
                Image(systemName: "plus.magnifyingglass")
                    .font(.system(size: 14))
            }
            .buttonStyle(PlainButtonStyle())
            .help("Zoom In (⌘+)")
            .keyboardShortcut("+", modifiers: .command)
            
            // Zoom percentage display
            Button(action: { isZoomPopoverVisible.toggle() }) {
                Text("\(Int(timeline.zoomLevel * 2))%")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .frame(width: 50)
            }
            .buttonStyle(PlainButtonStyle())
            .popover(isPresented: $isZoomPopoverVisible) {
                ZoomPopover(timeline: timeline)
            }
            
            Divider()
                .frame(height: 20)
            
            // Zoom presets
            Menu {
                ForEach(zoomPresets, id: \.0) { preset in
                    Button(preset.0) {
                        if preset.1 == 0 {
                            fitTimelineToView()
                        } else {
                            applyZoom(preset.1, animated: true)
                        }
                    }
                }
                
                Divider()
                
                Button("Zoom to Selection") {
                    zoomToSelection()
                }
                .disabled(timeline.selectedClips.isEmpty)
                
                Button("Zoom to Playhead") {
                    zoomToPlayhead()
                }
            } label: {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.system(size: 14))
            }
            .menuStyle(BorderlessButtonMenuStyle())
            .frame(width: 30)
            .help("Zoom Presets")
        }
    }
    
    // MARK: - Zoom Actions
    private func zoomIn() {
        let newZoom = min(200, timeline.zoomLevel * 1.2)
        applyZoom(newZoom, animated: true)
    }
    
    private func zoomOut() {
        let newZoom = max(5, timeline.zoomLevel / 1.2)
        applyZoom(newZoom, animated: true)
    }
    
    private func applyZoom(_ zoom: Double, focusPoint: CGFloat = 0.5, animated: Bool = false) {
        if animated {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                timeline.zoomLevel = zoom
                adjustScrollForZoom(focusPoint: focusPoint)
            }
        } else {
            timeline.zoomLevel = zoom
            adjustScrollForZoom(focusPoint: focusPoint)
        }
    }
    
    private func adjustScrollForZoom(focusPoint: CGFloat) {
        // Maintain focus point during zoom
        let timeAtFocus = timeline.timeFromX(focusPoint)
        let newX = timeline.xFromTime(timeAtFocus)
        let scrollAdjustment = newX - focusPoint
        
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            timeline.scrollOffset -= scrollAdjustment
        }
    }
    
    private func fitTimelineToView() {
        guard timeline.duration > 0 else { return }
        
        // Calculate zoom to fit entire timeline
        let viewWidth = NSScreen.main?.frame.width ?? 1920
        let availableWidth = viewWidth - 380 - 380 - 120 // Minus panels and track header
        let requiredZoom = Double(availableWidth) / timeline.duration
        
        withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
            timeline.zoomLevel = min(200, max(5, requiredZoom))
            timeline.scrollOffset = 0
        }
    }
    
    private func zoomToSelection() {
        guard !timeline.selectedClips.isEmpty else { return }
        
        // Find bounds of selected clips
        var minTime = Double.infinity
        var maxTime = -Double.infinity
        
        for clipId in timeline.selectedClips {
            if let clip = timeline.findClip(id: clipId) {
                minTime = min(minTime, clip.startTime)
                maxTime = max(maxTime, clip.startTime + clip.duration ?? 0)
            }
        }
        
        guard minTime < Double.infinity else { return }
        
        let duration = maxTime - minTime
        let viewWidth = NSScreen.main?.frame.width ?? 1920
        let availableWidth = viewWidth - 380 - 380 - 120
        let requiredZoom = Double(availableWidth) / duration * 0.8 // 80% to add padding
        
        withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
            timeline.zoomLevel = min(200, max(5, requiredZoom))
            timeline.scrollOffset = -timeline.xFromTime(minTime) + 50 // Add left padding
        }
    }
    
    private func zoomToPlayhead() {
        let playheadX = timeline.xFromTime(timeline.playheadPosition)
        let viewCenter = (NSScreen.main?.frame.width ?? 1920) / 2
        
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            timeline.zoomLevel = min(200, timeline.zoomLevel * 1.5)
            timeline.scrollOffset = viewCenter - playheadX
        }
    }
}

// MARK: - Custom Zoom Slider
struct ZoomSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    let onChanged: (Double) -> Void
    
    @State private var isDragging = false
    @State private var hoverLocation: CGFloat?
    
    private var normalizedValue: Double {
        (value - range.lowerBound) / (range.upperBound - range.lowerBound)
    }
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Track
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.white.opacity(0.1))
                    .frame(height: 4)
                
                // Fill
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.accentColor)
                    .frame(width: geometry.size.width * normalizedValue, height: 4)
                
                // Thumb
                Circle()
                    .fill(Color.white)
                    .frame(width: 12, height: 12)
                    .shadow(radius: 2)
                    .offset(x: geometry.size.width * normalizedValue - 6)
                    .scaleEffect(isDragging ? 1.2 : 1.0)
                    .animation(.spring(response: 0.2, dampingFraction: 0.8), value: isDragging)
                
                // Hover indicator
                if let hover = hoverLocation {
                    Circle()
                        .fill(Color.white.opacity(0.3))
                        .frame(width: 8, height: 8)
                        .position(x: hover, y: geometry.size.height / 2)
                }
            }
            .frame(height: 20)
            .contentShape(Rectangle())
            .onHover { hovering in
                if !hovering {
                    hoverLocation = nil
                }
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { drag in
                        isDragging = true
                        let newValue = (drag.location.x / geometry.size.width) * (range.upperBound - range.lowerBound) + range.lowerBound
                        value = min(range.upperBound, max(range.lowerBound, newValue))
                        onChanged(value)
                    }
                    .onEnded { _ in
                        isDragging = false
                    }
            )
        }
        .frame(height: 20)
    }
}

// MARK: - Zoom Popover
struct ZoomPopover: View {
    @ObservedObject var timeline: TimelineModel
    @State private var customZoomText = ""
    @FocusState private var isTextFieldFocused: Bool
    
    public var body: some View {
        VStack(spacing: 12) {
            Text("Timeline Zoom")
                .font(.system(size: 12, weight: .semibold))
            
            VStack(alignment: .leading, spacing: 8) {
                // Current zoom info
                HStack {
                    Text("Current:")
                    Spacer()
                    Text("\(Int(timeline.zoomLevel * 2))%")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                }
                
                HStack {
                    Text("Scale:")
                    Spacer()
                    Text("\(String(format: "%.1f", timeline.zoomLevel)) px/sec")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                }
                
                Divider()
                
                // Custom zoom input
                HStack {
                    Text("Custom:")
                    TextField("50", text: $customZoomText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .frame(width: 60)
                        .focused($isTextFieldFocused)
                        .onSubmit {
                            applyCustomZoom()
                        }
                    Text("%")
                    
                    Button("Apply") {
                        applyCustomZoom()
                    }
                    .buttonStyle(BorderedButtonStyle())
                    .controlSize(.small)
                }
            }
            .font(.system(size: 11))
            
            Divider()
            
            // Keyboard shortcuts
            VStack(alignment: .leading, spacing: 4) {
                Text("Keyboard Shortcuts")
                    .font(.system(size: 10, weight: .semibold))
                
                HStack {
                    Text("⌘+")
                    Spacer()
                    Text("Zoom In")
                }
                
                HStack {
                    Text("⌘-")
                    Spacer()
                    Text("Zoom Out")
                }
                
                HStack {
                    Text("⌘0")
                    Spacer()
                    Text("Fit Timeline")
                }
                
                HStack {
                    Text("⌥Z")
                    Spacer()
                    Text("Zoom to Selection")
                }
            }
            .font(.system(size: 10))
            .foregroundColor(.secondary)
        }
        .padding()
        .frame(width: 250)
        .onAppear {
            customZoomText = "\(Int(timeline.zoomLevel * 2))"
            isTextFieldFocused = true
        }
    }
    
    private func applyCustomZoom() {
        guard let zoomPercent = Double(customZoomText) else { return }
        let zoomLevel = zoomPercent / 2.0
        
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            timeline.zoomLevel = min(200, max(5, zoomLevel))
        }
    }
}

// MARK: - Zoom Gesture Modifier
struct TimelineZoomGesture: ViewModifier {
    @ObservedObject var timeline: TimelineModel
    @State private var magnificationStart: Double = 1.0
    
    func body(content: Content) -> some View {
        content
            .onAppear {
                // Enable trackpad pinch zoom
                NSEvent.addLocalMonitorForEvents(matching: .magnify) { event in
                    handleMagnify(event)
                    return event
                }
            }
            .gesture(
                MagnificationGesture()
                    .onChanged { value in
                        let delta = value / magnificationStart
                        let newZoom = timeline.zoomLevel * delta
                        timeline.zoomLevel = min(200, max(5, newZoom))
                    }
                    .onEnded { value in
                        magnificationStart = 1.0
                    }
            )
    }
    
    private func handleMagnify(_ event: NSEvent) {
        if event.phase == .began {
            magnificationStart = timeline.zoomLevel
        } else if event.phase == .changed {
            let newZoom = magnificationStart * (1 + event.magnification)
            timeline.zoomLevel = min(200, max(5, newZoom))
        }
    }
}

// MARK: - Zoom Focus Tracker
class ZoomFocusTracker: ObservableObject {
    @Published var focusPoint: CGPoint = .zero
    @Published var lastZoomLevel: Double = 50
    
    func updateFocus(at point: CGPoint, zoom: Double) {
        focusPoint = point
        lastZoomLevel = zoom
    }
    
    func calculateScrollAdjustment(newZoom: Double, viewWidth: CGFloat) -> CGFloat {
        guard lastZoomLevel != newZoom else { return 0 }
        
        let zoomRatio = newZoom / lastZoomLevel
        let focusOffset = focusPoint.x - viewWidth / 2
        let adjustment = focusOffset * (zoomRatio - 1)
        
        lastZoomLevel = newZoom
        return adjustment
    }
}

// Extension removed - methods already exist in TimelineModel.swift
