import SwiftUI
import AVFoundation
import AppKit

// MARK: - Clip View

struct ClipView: View {
    let clip: TimelineClip
    @ObservedObject var timeline: TimelineModel
    let isSelected: Bool
    let isHovered: Bool
    
    @State private var isDragging = false
    @State private var dragOffset: CGSize = .zero
    @State private var isResizingStart = false
    @State private var isResizingEnd = false
    @State private var resizeStartX: CGFloat = 0
    
    private let handleWidth: CGFloat = 8
    private let cornerRadius: CGFloat = 4
    
    public var body: some View {
        ZStack(alignment: .leading) {
            // Main clip body
            RoundedRectangle(cornerRadius: cornerRadius)
                .fill(clipColor)
                .overlay(
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .stroke(borderColor, lineWidth: borderWidth)
                )
            
            // Clip content
            HStack(spacing: 4) {
                // Thumbnail (if available)
                if let thumbnailData = clip.thumbnailData,
                   let image = NSImage(data: thumbnailData) {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 40, height: 40)
                        .clipped()
                        .cornerRadius(2)
                        .padding(.leading, 4)
                }
                
                // Clip info
                VStack(alignment: .leading, spacing: 2) {
                    Text(clip.name)
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(1)
                    
                    Text("\(formatDuration(clip.duration ?? 0))")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 4)
                
                Spacer()
            }
            .padding(.vertical, 4)
            
            // Resize handles
            if isSelected {
                // Start handle
                ResizeHandle(edge: .leading)
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                handleStartResize(value: value)
                            }
                            .onEnded { _ in
                                isResizingStart = false
                            }
                    )
                
                // End handle
                ResizeHandle(edge: .trailing)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                handleEndResize(value: value)
                            }
                            .onEnded { _ in
                                isResizingEnd = false
                            }
                    )
            }
        }
        .frame(width: clipWidth, height: clipHeight)
        .offset(dragOffset)
        .opacity(isDragging ? 0.7 : 1.0)
        .animation(.easeInOut(duration: 0.1), value: isHovered)
        .gesture(
            DragGesture()
                .onChanged { value in
                    handleDrag(value: value)
                }
                .onEnded { value in
                    handleDragEnd(value: value)
                }
        )
        .contextMenu {
            clipContextMenu
        }
    }
    
    // MARK: - Computed Properties
    
    private var clipWidth: CGFloat {
        CGFloat(clip.duration ?? 0 * timeline.zoomLevel)
    }
    
    private var clipHeight: CGFloat {
        if let track = timeline.tracks.first(where: { $0.clips.contains { $0.id == clip.id } }) {
            return track.height - 4
        }
        return 56
    }
    
    private var clipColor: Color {
        if isSelected {
            return clip.color.opacity(0.9)
        } else if isHovered {
            return clip.color.opacity(0.7)
        } else {
            return clip.color.opacity(0.6)
        }
    }
    
    private var borderColor: Color {
        if isSelected {
            return Color.accentColor
        } else if isHovered {
            return clip.color
        } else {
            return Color.clear
        }
    }
    
    private var borderWidth: CGFloat {
        isSelected ? 2 : 1
    }
    
    // MARK: - Context Menu
    
    @ViewBuilder
    private var clipContextMenu: some View {
        Button("Cut") {
            timeline.cutAtPlayhead()
        }
        
        Button("Copy") {
            // Copy to clipboard
        }
        
        Button("Delete") {
            timeline.removeClip(id: clip.id)
        }
        
        Divider()
        
        Button("Duplicate") {
            timeline.duplicateSelected()
        }
        
        Menu("Color") {
            ForEach([Color.blue, Color.green, Color.orange, Color.purple, Color.red], id: \.self) { color in
                Button(action: { changeClipColor(to: color) }) {
                    HStack {
                        Circle()
                            .fill(color)
                            .frame(width: 12, height: 12)
                        Text(colorName(for: color))
                    }
                }
            }
        }
        
        Divider()
        
        Button("Properties...") {
            // Show properties panel
        }
    }
    
    // MARK: - Drag Handling
    
    private func handleDrag(value: DragGesture.Value) {
        if !isDragging {
            isDragging = true
            if !isSelected {
                timeline.selectClip(id: clip.id)
            }
        }
        
        dragOffset = value.translation
    }
    
    private func handleDragEnd(value: DragGesture.Value) {
        isDragging = false
        
        // Calculate new time position
        let deltaTime = value.translation.width / CGFloat(timeline.zoomLevel)
        let newStartTime = clip.startTime + deltaTime
        
        // Snap to nearby points
        let snappedTime = timeline.snap(time: newStartTime, threshold: 0.5)
        
        // Move selected clips
        timeline.moveSelectedClips(by: snappedTime - clip.startTime)
        
        dragOffset = .zero
    }
    
    // MARK: - Resize Handling
    
    private func handleStartResize(value: DragGesture.Value) {
        if !isResizingStart {
            isResizingStart = true
            resizeStartX = value.startLocation.x
        }
        
        let deltaTime = value.translation.width / CGFloat(timeline.zoomLevel)
        let newStartTime = max(0, clip.startTime + deltaTime)
        let newDuration = clip.duration ?? 0 - deltaTime
        
        if newDuration > 0.1 {  // Minimum clip duration
            updateClip { clip in
                clip.startTime = newStartTime
                clip.duration = newDuration
                clip.inPoint += deltaTime
            }
        }
    }
    
    private func handleEndResize(value: DragGesture.Value) {
        if !isResizingEnd {
            isResizingEnd = true
        }
        
        let deltaTime = value.translation.width / CGFloat(timeline.zoomLevel)
        let newDuration = max(0.1, clip.duration ?? 0 + deltaTime)
        
        updateClip { clip in
            clip.duration = newDuration
            clip.outPoint = clip.inPoint + newDuration
        }
    }
    
    // MARK: - Helper Methods
    
    private func updateClip(_ transform: (inout TimelineClip) -> Void) {
        for trackIndex in timeline.tracks.indices {
            if let clipIndex = timeline.tracks[trackIndex].clips.firstIndex(where: { $0.id == clip.id }) {
                transform(&timeline.tracks[trackIndex].clips[clipIndex])
            }
        }
    }
    
    private func changeClipColor(to color: Color) {
        updateClip { clip in
            clip.color = color
        }
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.unitsStyle = .abbreviated
        return formatter.string(from: duration) ?? "\(Int(duration))s"
    }
    
    private func colorName(for color: Color) -> String {
        switch color {
        case .blue: return "Blue"
        case .green: return "Green"
        case .orange: return "Orange"
        case .purple: return "Purple"
        case .red: return "Red"
        default: return "Custom"
        }
    }
}

// MARK: - Resize Handle

struct ResizeHandle: View {
    let edge: Edge
    
    public var body: some View {
        Rectangle()
            .fill(Color.white.opacity(0.01))  // Nearly invisible but clickable
            .frame(width: 8)
            .overlay(
                Rectangle()
                    .fill(Color.accentColor)
                    .frame(width: 2)
                    .padding(.horizontal, 3),
                alignment: edge == .leading ? .leading : .trailing
            )
            .onHover { hovering in
                if hovering {
                    NSCursor.resizeLeftRight.push()
                } else {
                    NSCursor.pop()
                }
            }
    }
}

// MARK: - Clip Thumbnail Generator

class ClipThumbnailGenerator {
    static func generateThumbnail(for url: URL, at time: TimeInterval = 0) -> Data? {
        // This would use AVAsset to generate actual thumbnails
        // For now, return nil (placeholder)
        return nil
    }
}

// MARK: - Waveform View for Audio Clips

struct AudioWaveformView: View {
    let samples: [Float]
    let color: Color
    
    public var body: some View {
        GeometryReader { geometry in
            Path { path in
                let width = geometry.size.width
                let height = geometry.size.height
                let midY = height / 2
                
                guard !samples.isEmpty else { return }
                
                let samplesPerPixel = max(1, samples.count / Int(width))
                
                for x in 0..<Int(width) {
                    let sampleIndex = x * samplesPerPixel
                    if sampleIndex < samples.count {
                        let sample = CGFloat(samples[sampleIndex])
                        let y = midY - (sample * midY)
                        
                        if x == 0 {
                            path.move(to: CGPoint(x: CGFloat(x), y: y))
                        } else {
                            path.addLine(to: CGPoint(x: CGFloat(x), y: y))
                        }
                    }
                }
                
                // Mirror for bottom half
                for x in stride(from: Int(width) - 1, through: 0, by: -1) {
                    let sampleIndex = x * samplesPerPixel
                    if sampleIndex < samples.count {
                        let sample = CGFloat(samples[sampleIndex])
                        let y = midY + (sample * midY)
                        path.addLine(to: CGPoint(x: CGFloat(x), y: y))
                    }
                }
                
                path.closeSubpath()
            }
            .fill(color.opacity(0.3))
            .background(color.opacity(0.1))
        }
    }
}
