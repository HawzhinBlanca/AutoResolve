// AUTORESOLVE V3.0 - POLISHED LOADING STATES
// Professional loading indicators and error states

import SwiftUI

// MARK: - Loading Overlay
public struct LoadingOverlay: View {
    let message: String
    let progress: Double?
    @State private var rotation: Double = 0
    @State private var pulseScale: CGFloat = 1.0
    
    public init(message: String = "Loading...", progress: Double? = nil) {
        self.message = message
        self.progress = progress
    }
    
    public var body: some View {
        ZStack {
            // Semi-transparent background
            Color.black.opacity(0.7)
                .ignoresSafeArea()
                .transition(.opacity)
            
            // Loading card
            VStack(spacing: 24) {
                // Animated logo or spinner
                ZStack {
                    // Outer ring
                    Circle()
                        .stroke(
                            LinearGradient(
                                colors: [.blue, .purple],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 3
                        )
                        .frame(width: 60, height: 60)
                        .rotationEffect(.degrees(rotation))
                        .scaleEffect(pulseScale)
                    
                    // Inner progress
                    if let progress = progress {
                        Circle()
                            .trim(from: 0, to: progress)
                            .stroke(
                                LinearGradient(
                                    colors: [.green, .blue],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                ),
                                style: StrokeStyle(lineWidth: 4, lineCap: .round)
                            )
                            .frame(width: 50, height: 50)
                            .rotationEffect(.degrees(-90))
                            .animation(.easeInOut(duration: 0.5), value: progress)
                    } else {
                        // Indeterminate spinner
                        Circle()
                            .trim(from: 0, to: 0.7)
                            .stroke(
                                Color.white.opacity(0.8),
                                style: StrokeStyle(lineWidth: 4, lineCap: .round)
                            )
                            .frame(width: 50, height: 50)
                            .rotationEffect(.degrees(rotation))
                    }
                }
                
                // Message
                VStack(spacing: 8) {
                    Text(message)
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.white)
                    
                    if let progress = progress {
                        Text("\(Int(progress * 100))%")
                            .font(.system(size: 14, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
            }
            .padding(32)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.white.opacity(0.1), lineWidth: 1)
                    )
            )
            .shadow(color: .black.opacity(0.3), radius: 20, y: 10)
            .scaleEffect(pulseScale)
            .transition(.scale.combined(with: .opacity))
        }
        .onAppear {
            withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
                rotation = 360
            }
            
            if progress == nil {
                withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
                    pulseScale = 1.05
                }
            }
        }
    }
}

// MARK: - Connection Status Banner
public struct ConnectionStatusBanner: View {
    @EnvironmentObject private var backendClient: BackendClient
    @State private var showBanner = false
    @State private var opacity: Double = 1
    
    public var body: some View {
        VStack {
            if showBanner {
                HStack(spacing: 12) {
                    // Status icon
                    Group {
                        if backendClient.isConnected {
                            Image(systemName: "wifi")
                                .foregroundColor(.green)
                        } else {
                            Image(systemName: "wifi.slash")
                                .foregroundColor(.red)
                        }
                    }
                    .font(.system(size: 14))
                    
                    // Status text
                    Text(backendClient.isConnected ? "Connected" : "Disconnected")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.white)
                    
                    Spacer()
                    
                    // Retry button for errors
                    if !backendClient.isConnected {
                        Button(action: { 
                            // TODO: Implement reconnect functionality
                        }) {
                            Text("Retry")
                                .font(.system(size: 12, weight: .semibold))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 4)
                                .background(Color.blue)
                                .cornerRadius(4)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(backgroundForState)
                .cornerRadius(8)
                .shadow(color: .black.opacity(0.2), radius: 4, y: 2)
                .opacity(opacity)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.5, dampingFraction: 0.8), value: showBanner)
        .onReceive(backendClient.$isConnected) { connected in
            updateBanner(for: connected)
        }
    }
    
    private var backgroundForState: some View {
        Group {
            if backendClient.isConnected {
                Color.green.opacity(0.9)
            } else {
                Color.red.opacity(0.9)
            }
        }
    }
    
    private func updateBanner(for connected: Bool) {
        if connected {
            // Show briefly then hide
            showBanner = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                withAnimation {
                    opacity = 0
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    showBanner = false
                    opacity = 1
                }
            }
        } else {
            // Keep visible when not connected
            showBanner = true
            opacity = 1
        }
    }
}

// MARK: - Error View
public struct ErrorView: View {
    let error: Error
    let retryAction: (() -> Void)?
    @State private var shake = false
    
    public init(error: Error, retryAction: (() -> Void)? = nil) {
        self.error = error
        self.retryAction = retryAction
    }
    
    public var body: some View {
        VStack(spacing: 24) {
            // Error icon
            Image(systemName: "exclamationmark.octagon")
                .font(.system(size: 64))
                .foregroundColor(.red)
                .rotationEffect(.degrees(shake ? -5 : 5))
                .animation(.easeInOut(duration: 0.15).repeatCount(3, autoreverses: true), value: shake)
            
            // Error message
            VStack(spacing: 8) {
                Text("Something went wrong")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(.primary)
                
                Text(error.localizedDescription)
                    .font(.system(size: 14))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
            }
            
            // Retry button
            if let retryAction = retryAction {
                Button(action: {
                    retryAction()
                    shake = true
                }) {
                    HStack {
                        Image(systemName: "arrow.clockwise")
                        Text("Try Again")
                    }
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                    .background(
                        LinearGradient(
                            colors: [.blue, .blue.opacity(0.8)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(40)
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                shake = true
            }
        }
    }
}

// MARK: - Empty State View
public struct EmptyStateView: View {
    let icon: String
    let title: String
    let message: String
    let actionTitle: String?
    let action: (() -> Void)?
    
    @State private var bounceAmount: CGFloat = 0
    
    public init(
        icon: String = "doc.text.magnifyingglass",
        title: String = "No Content",
        message: String = "There's nothing here yet",
        actionTitle: String? = nil,
        action: (() -> Void)? = nil
    ) {
        self.icon = icon
        self.title = title
        self.message = message
        self.actionTitle = actionTitle
        self.action = action
    }
    
    public var body: some View {
        VStack(spacing: 24) {
            // Icon with subtle animation
            Image(systemName: icon)
                .font(.system(size: 72))
                .foregroundColor(.gray.opacity(0.5))
                .offset(y: bounceAmount)
            
            // Text content
            VStack(spacing: 8) {
                Text(title)
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(.primary)
                
                Text(message)
                    .font(.system(size: 14))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            
            // Optional action button
            if let actionTitle = actionTitle, let action = action {
                Button(action: action) {
                    Text(actionTitle)
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.accentColor)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.accentColor, lineWidth: 1.5)
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(40)
        .onAppear {
            withAnimation(.easeInOut(duration: 2).repeatForever(autoreverses: true)) {
                bounceAmount = -10
            }
        }
    }
}

// MARK: - Skeleton Loading
public struct SkeletonView: View {
    @State private var shimmerOffset: CGFloat = -1
    let height: CGFloat
    
    public init(height: CGFloat = 20) {
        self.height = height
    }
    
    public var body: some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(Color.gray.opacity(0.2))
            .frame(height: height)
            .overlay(
                GeometryReader { geometry in
                    RoundedRectangle(cornerRadius: 4)
                        .fill(
                            LinearGradient(
                                colors: [
                                    Color.gray.opacity(0.2),
                                    Color.gray.opacity(0.35),
                                    Color.gray.opacity(0.2)
                                ],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * 0.3)
                        .offset(x: geometry.size.width * shimmerOffset)
                }
            )
            .onAppear {
                withAnimation(.linear(duration: 1.5).repeatForever(autoreverses: false)) {
                    shimmerOffset = 2
                }
            }
    }
}
