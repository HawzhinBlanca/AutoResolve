// AUTORESOLVE V3.0 - RESIZABLE PANEL SYSTEM
// Professional panel management with drag handles

import SwiftUI
import AppKit

// MARK: - Resizable Panel View
public struct ResizablePanelView<Content: View>: View {
    let content: Content
    let edge: Edge
    @Binding var width: CGFloat
    let minWidth: CGFloat
    let maxWidth: CGFloat
    
    @State private var isDragging = false
    @State private var dragOffset: CGFloat = 0
    
    public init(
        edge: Edge,
        width: Binding<CGFloat>,
        minWidth: CGFloat = 200,
        maxWidth: CGFloat = 600,
        @ViewBuilder content: () -> Content
    ) {
        self.edge = edge
        self._width = width
        self.minWidth = minWidth
        self.maxWidth = maxWidth
        self.content = content()
    }
    
    public var body: some View {
        HStack(spacing: 0) {
            if edge == .trailing {
                PanelResizeHandle(
                    isDragging: $isDragging,
                    onDrag: handleDrag
                )
            }
            
            content
                .frame(width: width + dragOffset)
            
            if edge == .leading {
                PanelResizeHandle(
                    isDragging: $isDragging,
                    onDrag: handleDrag
                )
            }
        }
        .onChange(of: isDragging) { dragging in
            if !dragging {
                // Commit the drag
                width = min(maxWidth, max(minWidth, width + dragOffset))
                dragOffset = 0
            }
        }
    }
    
    private func handleDrag(_ translation: CGFloat) {
        let proposedWidth = width + translation
        if proposedWidth >= minWidth && proposedWidth <= maxWidth {
            dragOffset = translation
        }
    }
}

// MARK: - Resize Handle
struct PanelResizeHandle: View {
    @Binding var isDragging: Bool
    let onDrag: (CGFloat) -> Void
    
    @State private var hovered = false
    
    public var body: some View {
        Rectangle()
            .fill(Color.clear)
            .frame(width: 8)
            .overlay(
                Rectangle()
                    .fill(isDragging ? Color.accentColor : (hovered ? Color.gray.opacity(0.5) : Color.gray.opacity(0.2)))
                    .frame(width: 2)
            )
            .contentShape(Rectangle())
            .onHover { hovering in
                hovered = hovering
                if hovering {
                    NSCursor.resizeLeftRight.push()
                } else {
                    NSCursor.pop()
                }
            }
            .gesture(
                DragGesture()
                    .onChanged { value in
                        if !isDragging {
                            isDragging = true
                        }
                        onDrag(value.translation.width)
                    }
                    .onEnded { _ in
                        isDragging = false
                    }
            )
    }
}

// MARK: - Collapsible Panel
public struct CollapsiblePanel<Header: View, Content: View>: View {
    let header: Header
    let content: Content
    @Binding var isExpanded: Bool
    let animation: Animation
    
    public init(
        isExpanded: Binding<Bool>,
        animation: Animation = .easeInOut(duration: 0.2),
        @ViewBuilder header: () -> Header,
        @ViewBuilder content: () -> Content
    ) {
        self._isExpanded = isExpanded
        self.animation = animation
        self.header = header()
        self.content = content()
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "chevron.right")
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    .animation(animation, value: isExpanded)
                
                header
                
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(white: 0.15))
            .contentShape(Rectangle())
            .onTapGesture {
                withAnimation(animation) {
                    isExpanded.toggle()
                }
            }
            
            if isExpanded {
                content
                    .transition(.asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .move(edge: .top).combined(with: .opacity)
                    ))
            }
        }
    }
}

// MARK: - Panel Layout Manager
public class PanelLayoutManager: ObservableObject {
    @Published public var leftPanelWidth: CGFloat = 380
    @Published public var rightPanelWidth: CGFloat = 380
    @Published public var bottomPanelHeight: CGFloat = 300
    
    @Published public var leftPanelVisible = true
    @Published public var rightPanelVisible = true
    @Published public var bottomPanelVisible = true
    
    public enum Workspace: String, CaseIterable {
        case edit = "Edit"
        case color = "Color"
        case audio = "Fairlight"
        case effects = "Fusion"
        case deliver = "Deliver"
    }
    
    @Published public var currentWorkspace: Workspace = .edit
    
    public init() {}
    
    public func applyWorkspace(_ workspace: Workspace) {
        withAnimation(.easeInOut(duration: 0.3)) {
            currentWorkspace = workspace
            
            switch workspace {
            case .edit:
                leftPanelVisible = true
                rightPanelVisible = true
                bottomPanelVisible = true
                leftPanelWidth = 380
                rightPanelWidth = 380
                
            case .color:
                leftPanelVisible = false
                rightPanelVisible = true
                bottomPanelVisible = true
                rightPanelWidth = 500
                
            case .audio:
                leftPanelVisible = true
                rightPanelVisible = false
                bottomPanelVisible = true
                leftPanelWidth = 300
                bottomPanelHeight = 400
                
            case .effects:
                leftPanelVisible = true
                rightPanelVisible = true
                bottomPanelVisible = false
                leftPanelWidth = 400
                rightPanelWidth = 400
                
            case .deliver:
                leftPanelVisible = true
                rightPanelVisible = true
                bottomPanelVisible = false
                leftPanelWidth = 450
                rightPanelWidth = 450
            }
        }
    }
    
    public func saveLayout() {
        UserDefaults.standard.set(leftPanelWidth, forKey: "leftPanelWidth")
        UserDefaults.standard.set(rightPanelWidth, forKey: "rightPanelWidth")
        UserDefaults.standard.set(bottomPanelHeight, forKey: "bottomPanelHeight")
        UserDefaults.standard.set(currentWorkspace.rawValue, forKey: "currentWorkspace")
    }
    
    public func loadLayout() {
        if let width = UserDefaults.standard.object(forKey: "leftPanelWidth") as? CGFloat {
            leftPanelWidth = width
        }
        if let width = UserDefaults.standard.object(forKey: "rightPanelWidth") as? CGFloat {
            rightPanelWidth = width
        }
        if let height = UserDefaults.standard.object(forKey: "bottomPanelHeight") as? CGFloat {
            bottomPanelHeight = height
        }
        if let workspace = UserDefaults.standard.string(forKey: "currentWorkspace"),
           let ws = Workspace(rawValue: workspace) {
            currentWorkspace = ws
        }
    }
}

// MARK: - Floating Panel
public struct FloatingPanel<Content: View>: View {
    let title: String
    let content: Content
    @Binding var isPresented: Bool
    @State private var offset = CGSize.zero
    @State private var lastOffset = CGSize.zero
    
    public init(
        title: String,
        isPresented: Binding<Bool>,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self._isPresented = isPresented
        self.content = content()
    }
    
    public var body: some View {
        if isPresented {
            VStack(spacing: 0) {
                // Title bar
                HStack {
                    Text(title)
                        .font(.system(size: 13, weight: .medium))
                    
                    Spacer()
                    
                    Button(action: { isPresented = false }) {
                        Image(systemName: "xmark")
                            .font(.system(size: 10))
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(white: 0.18))
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            offset = CGSize(
                                width: lastOffset.width + value.translation.width,
                                height: lastOffset.height + value.translation.height
                            )
                        }
                        .onEnded { _ in
                            lastOffset = offset
                        }
                )
                
                content
                    .frame(minWidth: 300, minHeight: 200)
            }
            .background(Color(white: 0.15))
            .cornerRadius(8)
            .shadow(color: .black.opacity(0.5), radius: 10)
            .offset(offset)
        }
    }
}
