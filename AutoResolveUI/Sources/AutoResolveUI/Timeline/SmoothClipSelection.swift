// AUTORESOLVE V3.0 - SMOOTH CLIP SELECTION
import Combine
// Professional clip selection with smooth animations and multi-select

import SwiftUI
import AppKit
import AVFoundation

// MARK: - Enhanced Clip View with Selection
struct EnhancedClipView: View {
    let clip: TimelineClip
    @ObservedObject var timeline: TimelineModel
    @State private var isHovered = false
    @State private var isDragging = false
    @State private var dragOffset: CGSize = .zero
    @State private var resizeHandle: ResizeHandle = .none
    @State private var originalStartTime: TimeInterval = 0
    @State private var originalDuration: TimeInterval = 0
    
    enum ResizeHandle {
        case none, leading, trailing
    }
    
    private var isSelected: Bool {
        timeline.selectedClips.contains(clip.id)
    }
    
    private var clipWidth: CGFloat {
        CGFloat(clip.duration ?? 0.seconds * timeline.zoomLevel)
    }
    
    public var body: some View {
        ZStack {
            // Main clip body
            RoundedRectangle(cornerRadius: 4)
                .fill(clipBackgroundColor)
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .strokeBorder(clipBorderColor, lineWidth: isSelected ? 2 : 1)
                )
                .overlay(
                    // Selection highlight
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.accentColor.opacity(isSelected ? 0.2 : 0))
                        .animation(.easeInOut(duration: 0.15), value: isSelected)
                )
            
            // Clip content
            HStack {
                // Thumbnail if available
                if let thumbnail = clip.thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 60, height: clip.type == .video ? 40 : 30)
                        .clipped()
                        .cornerRadius(2)
                        .padding(.leading, 4)
                }
                
                // Clip name and info
                VStack(alignment: .leading, spacing: 2) {
                    Text(clip.name)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.white)
                        .lineLimit(1)
                    
                    if clip.type == .video {
                        HStack(spacing: 4) {
                            Image(systemName: "video")
                                .font(.system(size: 9))
                            Text(formatDuration(clip.duration ?? 0.seconds))
                                .font(.system(size: 9))
                        }
                        .foregroundColor(.white.opacity(0.7))
                    }
                }
                .padding(.horizontal, 8)
                
                Spacer()
            }
            
            // Hover overlay
            if isHovered {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.white.opacity(0.05))
                    .allowsHitTesting(false)
            }
            
            // Resize handles
            if isSelected && !timeline.isPlaying {
                // Leading handle
                ResizeHandleView(edge: .leading)
                    .frame(width: 8)
                    .position(x: 4, y: clip.type == .video ? 30 : 20)
                    .onHover { hovering in
                        if hovering {
                            NSCursor.resizeLeftRight.push()
                        } else {
                            NSCursor.pop()
                        }
                    }
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                handleResize(.leading, translation: value.translation)
                            }
                            .onEnded { _ in
                                commitResize()
                            }
                    )
                
                // Trailing handle
                ResizeHandleView(edge: .trailing)
                    .frame(width: 8)
                    .position(x: clipWidth - 4, y: clip.type == .video ? 30 : 20)
                    .onHover { hovering in
                        if hovering {
                            NSCursor.resizeLeftRight.push()
                        } else {
                            NSCursor.pop()
                        }
                    }
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                handleResize(.trailing, translation: value.translation)
                            }
                            .onEnded { _ in
                                commitResize()
                            }
                    )
            }
        }
        .frame(width: clipWidth, height: clip.type == .video ? 60 : 40)
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isSelected)
        .animation(.spring(response: 0.4, dampingFraction: 0.7), value: clipWidth)
        .offset(dragOffset)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.1)) {
                isHovered = hovering
            }
        }
        .onTapGesture {
            handleSelection()
        }
        .gesture(
            DragGesture()
                .onChanged { value in
                    if !isDragging {
                        isDragging = true
                        originalStartTime = clip.startTime.seconds
                        if !isSelected {
                            handleSelection()
                        }
                    }
                    handleDrag(translation: value.translation)
                }
                .onEnded { _ in
                    commitDrag()
                    isDragging = false
                }
        )
        .contextMenu {
            ClipContextMenu(clip: clip, timeline: timeline)
        }
    }
    
    // MARK: - Colors
    private var clipBackgroundColor: Color {
        switch clip.type {
        case .video:
            return Color(red: 0.2, green: 0.3, blue: 0.5)
        case .audio:
            return Color(red: 0.3, green: 0.5, blue: 0.3)
        case .title:
            return Color(red: 0.5, green: 0.3, blue: 0.5)
        case .transition:
            return Color(red: 0.5, green: 0.4, blue: 0.3)
        case .effect:
            return Color(red: 0.4, green: 0.3, blue: 0.4)
        }
    }
    
    private var clipBorderColor: Color {
        if isSelected {
            return Color.accentColor
        } else if isHovered {
            return Color.white.opacity(0.5)
        } else {
            return Color.white.opacity(0.2)
        }
    }
    
    // MARK: - Interaction Handlers
    private func handleSelection() {
        let isMultiSelect = NSEvent.modifierFlags.contains(.shift) || NSEvent.modifierFlags.contains(.command)
        
        withAnimation(.easeInOut(duration: 0.15)) {
            if isMultiSelect {
                if isSelected {
                    timeline.deselectClip(id: clip.id)
                } else {
                    timeline.selectClip(id: clip.id, multi: true)
                }
            } else {
                timeline.selectClip(id: clip.id, multi: false)
            }
        }
    }
    
    private func handleDrag(translation: CGSize) {
        let timeDelta = translation.width / timeline.zoomLevel
        dragOffset = CGSize(width: translation.width, height: 0)
        
        // Show snapping indicators
        let proposedTime = originalStartTime + timeDelta
        if let snapTime = timeline.findSnapPoint(near: proposedTime) {
            let snapX = CGFloat((snapTime - originalStartTime) * timeline.zoomLevel)
            dragOffset = CGSize(width: snapX, height: 0)
            
            // Haptic feedback for snapping
            NSHapticFeedbackManager.defaultPerformer.perform(
                .alignment,
                performanceTime: .now
            )
        }
    }
    
    private func commitDrag() {
        let finalTime = originalStartTime + (dragOffset.width / timeline.zoomLevel)
        
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            timeline.moveClip(id: clip.id, to: finalTime)
            dragOffset = .zero
        }
    }
    
    private func handleResize(_ handle: ResizeHandle, translation: CGSize) {
        if resizeHandle == .none {
            resizeHandle = handle
            originalStartTime = clip.startTime.seconds
            originalDuration = clip.duration ?? 0.seconds
        }
        
        let timeDelta = translation.width / timeline.zoomLevel
        
        switch handle {
        case .leading:
            let newStartTime = max(0, originalStartTime + timeDelta)
            let newDuration = originalDuration - (newStartTime - originalStartTime)
            if newDuration > 0.1 { // Minimum clip duration
                timeline.updateClip(id: clip.id, startTime: newStartTime, duration: newDuration)
            }
        case .trailing:
            let newDuration = max(0.1, originalDuration + timeDelta)
            timeline.updateClip(id: clip.id, duration: newDuration)
        case .none:
            break
        }
    }
    
    private func commitResize() {
        resizeHandle = .none
        // Changes already committed to the model
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.unitsStyle = .positional
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: duration) ?? "0:00"
    }
}

// MARK: - Resize Handle View
struct ResizeHandleView: View {
    let edge: Edge
    @State private var isHovered = false
    
    enum Edge {
        case leading, trailing
    }
    
    public var body: some View {
        RoundedRectangle(cornerRadius: 2)
            .fill(Color.white.opacity(isHovered ? 0.8 : 0.4))
            .frame(width: 4)
            .scaleEffect(x: isHovered ? 1.5 : 1)
            .animation(.easeInOut(duration: 0.1), value: isHovered)
            .onHover { hovering in
                isHovered = hovering
            }
    }
}

// MARK: - Clip Context Menu
struct ClipContextMenu: View {
    let clip: TimelineClip
    @ObservedObject var timeline: TimelineModel
    
    public var body: some View {
        VStack {
            Button("Cut") {
                timeline.cutClip(id: clip.id, at: clip.duration ?? 0.seconds / 2)
            }
            
            Button("Copy") {
                // timeline.copyClip(id: clip.id)
            }
            
            Button("Paste") {
                // timeline.paste()
            }
            
            Divider()
            
            Button("Duplicate") {
                // timeline.duplicateClip(id: clip.id)
            }
            
            Button("Delete") {
                // timeline.deleteClip(id: clip.id)
            }
            
            Divider()
            
            Menu("Speed") {
                Button("50%") { /* timeline.setClipSpeed(id: clip.id, speed: 0.5) */ }
                Button("75%") { /* timeline.setClipSpeed(id: clip.id, speed: 0.75) */ }
                Button("100%") { /* timeline.setClipSpeed(id: clip.id, speed: 1.0) */ }
                Button("150%") { /* timeline.setClipSpeed(id: clip.id, speed: 1.5) */ }
                Button("200%") { /* timeline.setClipSpeed(id: clip.id, speed: 2.0) */ }
            }
            
            Menu("Audio") {
                Button("Detach Audio") { /* timeline.detachAudio(id: clip.id) */ }
                Button("Mute") { /* timeline.muteClip(id: clip.id) */ }
                Button("Audio Levels...") { /* Show audio levels */ }
            }
            
            Divider()
            
            Button("Properties...") {
                // Show clip properties
            }
        }
    }
}

// MARK: - Selection Rectangle
struct TimelineSelectionRectangleView: View {
    @Binding var startPoint: CGPoint
    @Binding var currentPoint: CGPoint
    @Binding var isSelecting: Bool
    
    private var rect: CGRect {
        let x = min(startPoint.x, currentPoint.x)
        let y = min(startPoint.y, currentPoint.y)
        let width = abs(currentPoint.x - startPoint.x)
        let height = abs(currentPoint.y - startPoint.y)
        return CGRect(x: x, y: y, width: width, height: height)
    }
    
    public var body: some View {
        if isSelecting {
            Rectangle()
                .fill(Color.accentColor.opacity(0.1))
                .overlay(
                    Rectangle()
                        .stroke(Color.accentColor, lineWidth: 1)
                )
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)
                .allowsHitTesting(false)
                .animation(.none, value: rect)
        }
    }
}

// MARK: - Multi-Select Manager
class MultiSelectManager: ObservableObject {
    @Published var selectedClips: Set<UUID> = []
    @Published var lastSelectedClip: UUID?
    @Published var selectionAnchor: UUID?
    
    func selectClip(_ id: UUID, withModifiers: NSEvent.ModifierFlags) {
        if withModifiers.contains(.shift), let anchor = selectionAnchor {
            // Range selection
            selectRange(from: anchor, to: id)
        } else if withModifiers.contains(.command) {
            // Toggle selection
            if selectedClips.contains(id) {
                selectedClips.remove(id)
            } else {
                selectedClips.insert(id)
            }
        } else {
            // Single selection
            selectedClips = [id]
            selectionAnchor = id
        }
        lastSelectedClip = id
    }
    
    func selectAll(_ clips: [TimelineClip]) {
        selectedClips = Set(clips.map { $0.id })
    }
    
    func deselectAll() {
        selectedClips.removeAll()
        lastSelectedClip = nil
        selectionAnchor = nil
    }
    
    private func selectRange(from: UUID, to: UUID) {
        // Implementation depends on clip ordering
        // This is a simplified version
        selectedClips.insert(from)
        selectedClips.insert(to)
    }
}
