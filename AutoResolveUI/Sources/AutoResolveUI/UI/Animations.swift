// AUTORESOLVE V3.0 - POLISHED ANIMATIONS
// Smooth transitions and professional animations

import SwiftUI

// MARK: - Animation Extensions
public extension Animation {
    /// Professional spring animation
    static var smooth: Animation {
        .spring(response: 0.5, dampingFraction: 0.85, blendDuration: 0)
    }
    
    /// Quick responsive animation
    static var quick: Animation {
        .spring(response: 0.3, dampingFraction: 0.75, blendDuration: 0)
    }
    
    /// Elastic bounce animation
    static var bounce: Animation {
        .spring(response: 0.6, dampingFraction: 0.6, blendDuration: 0)
    }
    
    /// Smooth ease for UI elements
    static var smoothEase: Animation {
        .easeInOut(duration: 0.35)
    }
}

// MARK: - View Modifiers

/// Adds a smooth scale effect on tap
struct ScaleOnTap: ViewModifier {
    @State private var isPressed = false
    let scale: CGFloat
    
    init(scale: CGFloat = 0.95) {
        self.scale = scale
    }
    
    func body(content: Content) -> some View {
        content
            .scaleEffect(isPressed ? scale : 1.0)
            .animation(.quick, value: isPressed)
            .onLongPressGesture(
                minimumDuration: .infinity,
                maximumDistance: .infinity,
                pressing: { pressing in
                    isPressed = pressing
                },
                perform: {}
            )
    }
}

/// Adds a smooth hover effect
struct HoverEffect: ViewModifier {
    @State private var isHovered = false
    let scale: CGFloat
    let brightness: Double
    
    init(scale: CGFloat = 1.02, brightness: Double = 0.05) {
        self.scale = scale
        self.brightness = brightness
    }
    
    func body(content: Content) -> some View {
        content
            .scaleEffect(isHovered ? scale : 1.0)
            .brightness(isHovered ? brightness : 0)
            .animation(.smooth, value: isHovered)
            .onHover { hovering in
                isHovered = hovering
            }
    }
}

/// Adds a smooth fade-in animation
struct FadeIn: ViewModifier {
    @State private var opacity: Double = 0
    let duration: Double
    let delay: Double
    
    init(duration: Double = 0.5, delay: Double = 0) {
        self.duration = duration
        self.delay = delay
    }
    
    func body(content: Content) -> some View {
        content
            .opacity(opacity)
            .onAppear {
                withAnimation(.easeIn(duration: duration).delay(delay)) {
                    opacity = 1
                }
            }
    }
}

/// Adds a slide and fade transition
struct SlideAndFade: ViewModifier {
    @State private var offset: CGFloat = 20
    @State private var opacity: Double = 0
    let duration: Double
    let delay: Double
    
    init(duration: Double = 0.5, delay: Double = 0) {
        self.duration = duration
        self.delay = delay
    }
    
    func body(content: Content) -> some View {
        content
            .offset(y: offset)
            .opacity(opacity)
            .onAppear {
                withAnimation(.smooth.delay(delay)) {
                    offset = 0
                    opacity = 1
                }
            }
    }
}

/// Adds a pulsing glow effect
struct PulsingGlow: ViewModifier {
    @State private var glowAmount: Double = 0
    let color: Color
    let radius: CGFloat
    
    init(color: Color = .blue, radius: CGFloat = 10) {
        self.color = color
        self.radius = radius
    }
    
    func body(content: Content) -> some View {
        content
            .shadow(color: color.opacity(glowAmount), radius: radius)
            .onAppear {
                withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
                    glowAmount = 0.5
                }
            }
    }
}

// MARK: - View Extensions
public extension View {
    /// Apply scale on tap
    func scaleOnTap(_ scale: CGFloat = 0.95) -> some View {
        modifier(ScaleOnTap(scale: scale))
    }
    
    /// Apply hover effect
    func hoverEffect(scale: CGFloat = 1.02, brightness: Double = 0.05) -> some View {
        modifier(HoverEffect(scale: scale, brightness: brightness))
    }
    
    /// Apply fade in animation
    func fadeIn(duration: Double = 0.5, delay: Double = 0) -> some View {
        modifier(FadeIn(duration: duration, delay: delay))
    }
    
    /// Apply slide and fade animation
    func slideAndFade(duration: Double = 0.5, delay: Double = 0) -> some View {
        modifier(SlideAndFade(duration: duration, delay: delay))
    }
    
    /// Apply pulsing glow effect
    func pulsingGlow(color: Color = .blue, radius: CGFloat = 10) -> some View {
        modifier(PulsingGlow(color: color, radius: radius))
    }
}

// MARK: - Animated Components

/// Animated progress bar
public struct AnimatedProgressBar: View {
    let progress: Double
    let height: CGFloat
    let color: Color
    @State private var animatedProgress: Double = 0
    
    public init(
        progress: Double,
        height: CGFloat = 4,
        color: Color = .blue
    ) {
        self.progress = min(max(progress, 0), 1)
        self.height = height
        self.color = color
    }
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background
                RoundedRectangle(cornerRadius: height / 2)
                    .fill(Color.gray.opacity(0.2))
                    .frame(height: height)
                
                // Progress
                RoundedRectangle(cornerRadius: height / 2)
                    .fill(
                        LinearGradient(
                            colors: [color, color.opacity(0.8)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: geometry.size.width * animatedProgress, height: height)
                    .animation(.smooth, value: animatedProgress)
            }
        }
        .frame(height: height)
        .onAppear {
            animatedProgress = progress
        }
        .onChange(of: progress) { _, newValue in
            animatedProgress = newValue
        }
    }
}

/// Animated activity indicator
public struct AnimatedActivityIndicator: View {
    @State private var rotation: Double = 0
    @State private var trimEnd: Double = 0.6
    let size: CGFloat
    let lineWidth: CGFloat
    let color: Color
    
    public init(
        size: CGFloat = 40,
        lineWidth: CGFloat = 3,
        color: Color = .blue
    ) {
        self.size = size
        self.lineWidth = lineWidth
        self.color = color
    }
    
    public var body: some View {
        Circle()
            .trim(from: 0, to: trimEnd)
            .stroke(
                LinearGradient(
                    colors: [color, color.opacity(0.5)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
            )
            .frame(width: size, height: size)
            .rotationEffect(.degrees(rotation))
            .onAppear {
                withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
                    rotation = 360
                }
                withAnimation(.easeInOut(duration: 1).repeatForever(autoreverses: true)) {
                    trimEnd = 0.9
                }
            }
    }
}

/// Smooth transition container
public struct SmoothTransition<Content: View>: View {
    let content: Content
    @State private var appeared = false
    
    public init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }
    
    public var body: some View {
        content
            .opacity(appeared ? 1 : 0)
            .scaleEffect(appeared ? 1 : 0.95)
            .onAppear {
                withAnimation(.smooth) {
                    appeared = true
                }
            }
    }
}

/// Ripple effect button
public struct RippleButton<Label: View>: View {
    let action: () -> Void
    let label: () -> Label
    @State private var ripples: [Ripple] = []
    
    public init(action: @escaping () -> Void, @ViewBuilder label: @escaping () -> Label) {
        self.action = action
        self.label = label
    }
    
    struct Ripple: Identifiable {
        public let id = UUID()
        let position: CGPoint
        let startTime = Date()
    }
    
    public var body: some View {
        Button(action: action) {
            label()
                .overlay(
                    GeometryReader { geometry in
                        ZStack {
                            ForEach(ripples) { ripple in
                                Circle()
                                    .fill(Color.white.opacity(0.3))
                                    .frame(width: 1, height: 1)
                                    .position(ripple.position)
                                    .modifier(RippleModifier(ripple: ripple))
                            }
                        }
                    }
                    .clipped()
                )
        }
        .buttonStyle(.plain)
        .onTapGesture { location in
            ripples.append(Ripple(position: location))
            
            // Clean up old ripples
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                ripples.removeAll { ripple in
                    Date().timeIntervalSince(ripple.startTime) > 0.6
                }
            }
        }
    }
    
    struct RippleModifier: ViewModifier {
        let ripple: Ripple
        @State private var scale: CGFloat = 0
        @State private var opacity: Double = 1
        
        func body(content: Content) -> some View {
            content
                .scaleEffect(scale)
                .opacity(opacity)
                .onAppear {
                    withAnimation(.easeOut(duration: 0.6)) {
                        scale = 200
                        opacity = 0
                    }
                }
        }
    }
}
