// AUTORESOLVE V3.0 - COMPREHENSIVE SYSTEM TELEMETRY
// Real-time monitoring and performance dashboard

import SwiftUI
import Combine

/// Comprehensive telemetry dashboard
public struct SystemTelemetryDashboard: View {
    @StateObject private var telemetry = SystemTelemetryManager.shared
    @State private var selectedTab = TelemetryTab.overview
    @State private var isMinimized = false
    
    enum TelemetryTab: String, CaseIterable {
        case overview = "Overview"
        case network = "Network"
        case performance = "Performance"
        case errors = "Errors"
        case backend = "Backend"
        
        var icon: String {
            switch self {
            case .overview: return "gauge.with.dots.needle.33percent"
            case .network: return "network"
            case .performance: return "speedometer"
            case .errors: return "exclamationmark.triangle"
            case .backend: return "server.rack"
            }
        }
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "antenna.radiowaves.left.and.right")
                    .font(.system(size: 14))
                    .foregroundColor(.blue)
                
                Text("System Telemetry")
                    .font(.system(size: 14, weight: .semibold))
                
                Spacer()
                
                // System status
                HStack(spacing: 4) {
                    Circle()
                        .fill(telemetry.systemHealth.color)
                        .frame(width: 8, height: 8)
                    Text(telemetry.systemHealth.rawValue)
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
                
                Button(action: { isMinimized.toggle() }) {
                    Image(systemName: isMinimized ? "chevron.down" : "chevron.up")
                        .font(.system(size: 12))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(white: 0.12))
            
            if !isMinimized {
                // Tab selector
                HStack(spacing: 1) {
                    ForEach(TelemetryTab.allCases, id: \.self) { tab in
                        Button(action: { selectedTab = tab }) {
                            VStack(spacing: 4) {
                                Image(systemName: tab.icon)
                                    .font(.system(size: 14))
                                Text(tab.rawValue)
                                    .font(.system(size: 10))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(selectedTab == tab ? Color.blue.opacity(0.2) : Color.clear)
                            .foregroundColor(selectedTab == tab ? .blue : .secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .background(Color(white: 0.1))
                
                // Content
                Group {
                    switch selectedTab {
                    case .overview:
                        TelemetryOverviewTab(telemetry: telemetry)
                    case .network:
                        NetworkTab(telemetry: telemetry)
                    case .performance:
                        TelemetryPerformanceTab(telemetry: telemetry)
                    case .errors:
                        ErrorsTab(telemetry: telemetry)
                    case .backend:
                        BackendTab(telemetry: telemetry)
                    }
                }
                .frame(height: 300)
                .background(Color(white: 0.08))
            }
        }
        .background(Color(white: 0.06))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.white.opacity(0.1), lineWidth: 1)
        )
        .animation(.smooth, value: isMinimized)
        .animation(.quick, value: selectedTab)
    }
}

// MARK: - Overview Tab

struct TelemetryOverviewTab: View {
    @ObservedObject var telemetry: SystemTelemetryManager
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Key metrics
                HStack(spacing: 16) {
                    MetricCard(
                        title: "CPU",
                        value: "\(Int(telemetry.cpuUsage))%",
                        trend: telemetry.cpuTrend,
                        color: telemetry.cpuUsage > 80 ? .red : .green
                    )
                    
                    MetricCard(
                        title: "Memory",
                        value: formatBytes(telemetry.memoryUsage),
                        trend: telemetry.memoryTrend,
                        color: telemetry.memoryPressure > 0.8 ? .red : .green
                    )
                    
                    MetricCard(
                        title: "Network",
                        value: "\(telemetry.activeConnections)",
                        trend: .stable,
                        color: .blue
                    )
                    
                    MetricCard(
                        title: "Errors",
                        value: "\(telemetry.errorCount)",
                        trend: telemetry.errorTrend,
                        color: telemetry.errorCount > 0 ? .orange : .green
                    )
                }
                
                // System components
                VStack(alignment: .leading, spacing: 8) {
                    Text("System Components")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.secondary)
                    
                    ForEach(telemetry.components, id: \.name) { component in
                        ComponentStatusRow(component: component)
                    }
                }
                
                // Recent activity
                VStack(alignment: .leading, spacing: 8) {
                    Text("Recent Activity")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.secondary)
                    
                    ForEach(telemetry.recentActivity.prefix(5), id: \.id) { activity in
                        ActivityRow(activity: activity)
                    }
                }
            }
            .padding()
        }
    }
}

// MARK: - Network Tab

struct NetworkTab: View {
    @ObservedObject var telemetry: SystemTelemetryManager
    
    public var body: some View {
        VStack(spacing: 0) {
            // Connection status
            ConnectionStatusBanner()
                .padding()
            
            // WebSocket status
            WebSocketStatusView()
                .padding(.horizontal)
            
            // Rate limiters
            VStack(alignment: .leading, spacing: 8) {
                Text("Rate Limiters")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
                
                RateLimitMonitor(name: "API", limiter: RateLimitManager.shared.apiLimiter)
                    .padding(.horizontal)
                
                RateLimitMonitor(name: "Media", limiter: RateLimitManager.shared.mediaLimiter)
                    .padding(.horizontal)
                
                RateLimitMonitor(name: "Analysis", limiter: RateLimitManager.shared.analysisLimiter)
                    .padding(.horizontal)
            }
            
            Spacer()
        }
    }
}

// MARK: - Performance Tab

struct TelemetryPerformanceTab: View {
    @ObservedObject var telemetry: SystemTelemetryManager
    
    public var body: some View {
        VStack(spacing: 16) {
            // Performance graph (simplified without Charts framework)
            VStack(alignment: .leading, spacing: 8) {
                Text("Performance Metrics")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.secondary)
                
                HStack {
                    VStack(alignment: .leading) {
                        Text("CPU Usage")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                        Text("\(Int(telemetry.cpuUsage))%")
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .foregroundColor(.blue)
                    }
                    
                    Spacer()
                    
                    VStack(alignment: .leading) {
                        Text("Memory")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                        Text(formatBytes(telemetry.memoryUsage))
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .foregroundColor(.green)
                    }
                    
                    Spacer()
                }
                
                // Simple performance bars
                ForEach(telemetry.performanceHistory.suffix(10), id: \.id) { dataPoint in
                    HStack(spacing: 4) {
                        Text(dataPoint.timestamp, style: .time)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(.secondary)
                            .frame(width: 60)
                        
                        // CPU bar
                        GeometryReader { geometry in
                            Rectangle()
                                .fill(Color.blue.opacity(0.6))
                                .frame(width: geometry.size.width * (dataPoint.cpu / 100))
                        }
                        .frame(height: 4)
                        
                        // Memory bar
                        GeometryReader { geometry in
                            Rectangle()
                                .fill(Color.green.opacity(0.6))
                                .frame(width: geometry.size.width * (dataPoint.memory / 1000000000)) // Scale to GB
                        }
                        .frame(height: 4)
                    }
                    .frame(height: 8)
                }
            }
            .padding()
            
            // Cache statistics
            CacheMonitorView()
                .padding(.horizontal)
            
            // Circuit breakers
            HStack {
                CircuitBreakerMonitor(breaker: CircuitBreaker<Any>(configuration: .default))
                Spacer()
            }
            .padding(.horizontal)
            
            Spacer()
        }
    }
}

// MARK: - Errors Tab

struct ErrorsTab: View {
    @ObservedObject var telemetry: SystemTelemetryManager
    
    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                if telemetry.errors.isEmpty {
                    EmptyStateView(
                        icon: "checkmark.circle",
                        title: "No Errors",
                        message: "System is running smoothly"
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ForEach(telemetry.errors.reversed(), id: \.id) { error in
                        ErrorItemRow(error: error)
                    }
                }
            }
            .padding()
        }
    }
}

// MARK: - Backend Tab

struct BackendTab: View {
    @ObservedObject var telemetry: SystemTelemetryManager
    
    public var body: some View {
        VStack(spacing: 16) {
            // Backend health
            HStack {
                VStack(alignment: .leading) {
                    Text("Backend Status")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.secondary)
                    
                    HStack {
                        Circle()
                            .fill(telemetry.backendHealthy ? Color.green : Color.red)
                            .frame(width: 10, height: 10)
                        Text(telemetry.backendHealthy ? "Connected" : "Disconnected")
                            .font(.system(size: 14))
                    }
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("Memory")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                    Text("\(telemetry.backendMemoryMB) MB")
                        .font(.system(size: 14, weight: .medium, design: .monospaced))
                }
                
                VStack(alignment: .trailing) {
                    Text("Active Tasks")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                    Text("\(telemetry.activeTasks)")
                        .font(.system(size: 14, weight: .medium, design: .monospaced))
                }
            }
            .padding()
            .background(Color(white: 0.1))
            .cornerRadius(8)
            .padding(.horizontal)
            
            // Pipeline status
            VStack(alignment: .leading, spacing: 8) {
                Text("Pipeline Status")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.secondary)
                
                ForEach(telemetry.pipelineTasks, id: \.id) { task in
                    PipelineTaskRow(task: task)
                }
            }
            .padding(.horizontal)
            
            Spacer()
        }
    }
}

// MARK: - Supporting Views

struct MetricCard: View {
    let title: String
    let value: String
    let trend: SystemTelemetryManager.Trend
    let color: Color
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 10))
                .foregroundColor(.secondary)
            
            HStack(alignment: .bottom, spacing: 4) {
                Text(value)
                    .font(.system(size: 16, weight: .semibold, design: .monospaced))
                    .foregroundColor(color)
                
                Image(systemName: trend.icon)
                    .font(.system(size: 10))
                    .foregroundColor(trend.color)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(Color(white: 0.1))
        .cornerRadius(6)
    }
}

struct ComponentStatusRow: View {
    let component: SystemTelemetryManager.Component
    
    public var body: some View {
        HStack {
            Circle()
                .fill(component.status.color)
                .frame(width: 8, height: 8)
            
            Text(component.name)
                .font(.system(size: 11))
            
            Spacer()
            
            Text(component.status.rawValue)
                .font(.system(size: 10))
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 2)
    }
}

struct ActivityRow: View {
    let activity: SystemTelemetryManager.Activity
    
    public var body: some View {
        HStack {
            Text(activity.timestamp, style: .time)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.secondary)
            
            Text(activity.message)
                .font(.system(size: 11))
                .lineLimit(1)
            
            Spacer()
        }
    }
}

struct ErrorItemRow: View {
    let error: SystemTelemetryManager.ErrorItem
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "exclamationmark.circle")
                    .foregroundColor(.red)
                    .font(.system(size: 12))
                
                Text(error.message)
                    .font(.system(size: 11))
                    .lineLimit(2)
                
                Spacer()
                
                Text(error.timestamp, style: .time)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            
            if let context = error.context {
                Text(context)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(8)
        .background(Color.red.opacity(0.1))
        .cornerRadius(4)
    }
}

struct PipelineTaskRow: View {
    let task: SystemTelemetryManager.PipelineTask
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(task.name)
                    .font(.system(size: 11))
                
                Spacer()
                
                Text(task.status)
                    .font(.system(size: 10))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(task.statusColor.opacity(0.2))
                    .foregroundColor(task.statusColor)
                    .cornerRadius(4)
            }
            
            if task.progress > 0 {
                ProgressView(value: task.progress)
                    .tint(task.statusColor)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Telemetry Manager

public class SystemTelemetryManager: ObservableObject {
    public static let shared = SystemTelemetryManager()
    
    // System metrics
    @Published public var cpuUsage: Double = 0
    @Published public var memoryUsage: Int = 0
    @Published public var memoryPressure: Double = 0
    @Published public var activeConnections = 0
    @Published public var errorCount = 0
    
    // Trends
    @Published public var cpuTrend: Trend = .stable
    @Published public var memoryTrend: Trend = .stable
    @Published public var errorTrend: Trend = .stable
    
    // System health
    @Published public var systemHealth: SystemHealth = .healthy
    
    // Backend metrics
    @Published public var backendHealthy = false
    @Published public var backendMemoryMB = 0
    @Published public var activeTasks = 0
    
    // Collections
    @Published public var components: [Component] = []
    @Published public var recentActivity: [Activity] = []
    @Published public var errors: [ErrorItem] = []
    @Published public var pipelineTasks: [PipelineTask] = []
    @Published public var performanceHistory: [PerformanceDataPoint] = []
    
    // Types
    public enum SystemHealth: String {
        case healthy = "Healthy"
        case degraded = "Degraded"
        case critical = "Critical"
        
        var color: Color {
            switch self {
            case .healthy: return .green
            case .degraded: return .orange
            case .critical: return .red
            }
        }
    }
    
    public enum Trend {
        case rising, falling, stable
        
        var icon: String {
            switch self {
            case .rising: return "arrow.up"
            case .falling: return "arrow.down"
            case .stable: return "minus"
            }
        }
        
        var color: Color {
            switch self {
            case .rising: return .orange
            case .falling: return .green
            case .stable: return .gray
            }
        }
    }
    
    public struct Component {
        let name: String
        let status: ComponentStatus
        
        enum ComponentStatus: String {
            case online = "Online"
            case offline = "Offline"
            case degraded = "Degraded"
            
            var color: Color {
                switch self {
                case .online: return .green
                case .offline: return .red
                case .degraded: return .orange
                }
            }
        }
    }
    
    public struct Activity: Identifiable {
        public let id = UUID()
        let timestamp: Date
        let message: String
    }
    
    public struct ErrorItem: Identifiable {
        public let id = UUID()
        let timestamp: Date
        let message: String
        let context: String?
    }
    
    public struct PipelineTask: Identifiable {
        public let id: String
        let name: String
        let status: String
        let progress: Double
        
        var statusColor: Color {
            switch status {
            case "running": return .blue
            case "completed": return .green
            case "failed": return .red
            default: return .gray
            }
        }
    }
    
    public struct PerformanceDataPoint: Identifiable {
        public let id = UUID()
        let timestamp: Date
        let cpu: Double
        let memory: Double
    }
    
    private var cancellables = Set<AnyCancellable>()
    private var updateTimer: Timer?
    
    private init() {
        setupMonitoring()
        initializeComponents()
    }
    
    private func setupMonitoring() {
        // Update metrics periodically
        updateTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { _ in
            self.updateMetrics()
        }
        
        // Monitor connection
        ConnectionManager.shared.$healthStatus
            .sink { [weak self] health in
                self?.backendHealthy = health?.isHealthy ?? false
                self?.backendMemoryMB = health?.memory_mb ?? 0
                self?.activeTasks = health?.active_tasks ?? 0
            }
            .store(in: &cancellables)
    }
    
    private func initializeComponents() {
        components = [
            Component(name: "Backend Service", status: .online),
            Component(name: "WebSocket", status: .online),
            Component(name: "Cache System", status: .online),
            Component(name: "Rate Limiter", status: .online),
            Component(name: "Circuit Breaker", status: .online)
        ]
    }
    
    private func updateMetrics() {
        // Simulate metrics (in production, get real values)
        cpuUsage = Double.random(in: 10...40)
        memoryUsage = Int.random(in: 200...400) * 1024 * 1024
        memoryPressure = Double.random(in: 0.3...0.6)
        
        // Update trends
        cpuTrend = cpuUsage > 30 ? .rising : .stable
        memoryTrend = memoryPressure > 0.5 ? .rising : .stable
        
        // Update health
        if cpuUsage > 80 || memoryPressure > 0.8 {
            systemHealth = .critical
        } else if cpuUsage > 60 || memoryPressure > 0.6 {
            systemHealth = .degraded
        } else {
            systemHealth = .healthy
        }
        
        // Add performance data point
        performanceHistory.append(
            PerformanceDataPoint(
                timestamp: Date(),
                cpu: cpuUsage,
                memory: Double(memoryUsage)
            )
        )
        
        // Keep only recent history
        if performanceHistory.count > 60 {
            performanceHistory.removeFirst()
        }
    }
    
    public func logActivity(_ message: String) {
        recentActivity.append(Activity(timestamp: Date(), message: message))
        if recentActivity.count > 100 {
            recentActivity.removeFirst()
        }
    }
    
    public func logError(_ message: String, context: String? = nil) {
        errors.append(ErrorItem(timestamp: Date(), message: message, context: context))
        errorCount = errors.count
        errorTrend = .rising
        
        if errors.count > 100 {
            errors.removeFirst()
        }
    }
}

// MARK: - Helpers

private func formatBytes(_ bytes: Int) -> String {
    let formatter = ByteCountFormatter()
    formatter.countStyle = .binary
    return formatter.string(fromByteCount: Int64(bytes))
}
