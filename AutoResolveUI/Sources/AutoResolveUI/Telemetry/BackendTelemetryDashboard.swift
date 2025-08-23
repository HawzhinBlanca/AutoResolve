import SwiftUI
import Charts
import Combine

// MARK: - Supporting Types

enum MetricType: String, CaseIterable {
        case cpu = "CPU Usage"
        case memory = "Memory"
        case gpu = "GPU"
        case disk = "Disk I/O"
        case network = "Network"
        case pipeline = "Pipeline"
        
        var icon: String {
            switch self {
            case .cpu: return "cpu"
            case .memory: return "memorychip"
            case .gpu: return "rectangle.3.group"
            case .disk: return "internaldrive"
            case .network: return "network"
            case .pipeline: return "gearshape.2"
            }
        }
    }
    
enum ChartType: String, CaseIterable {
    case line = "Line"
    case bar = "Bar"
    case area = "Area"
}

enum TelemetryTimeRange: String, CaseIterable {
        case last1Minute = "1 min"
        case last5Minutes = "5 min"
        case last15Minutes = "15 min"
        case last1Hour = "1 hour"
        case last24Hours = "24 hours"
        
        var seconds: TimeInterval {
            switch self {
            case .last1Minute: return 60
            case .last5Minutes: return 300
            case .last15Minutes: return 900
            case .last1Hour: return 3600
            case .last24Hours: return 86400
            }
        }
    }

// MARK: - Backend Telemetry Dashboard
public struct BackendTelemetryDashboard: View {
    @StateObject private var telemetry = TelemetryManager()
    @State private var selectedMetric = MetricType.cpu
    @State private var timeRange = TelemetryTimeRange.last5Minutes
    @State private var autoRefresh = true
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            dashboardHeader
            
            Divider()
            
            // Main Content
            HSplitView {
                // Left: Metrics Overview
                metricsOverview
                    .frame(minWidth: 250, maxWidth: 350)
                
                // Center: Charts
                chartsPanel
                    .frame(minWidth: 400)
                
                // Right: Alerts & Logs
                alertsPanel
                    .frame(width: 300)
            }
            
            // Bottom: Status Bar
            statusBar
        }
        .onAppear {
            telemetry.startMonitoring()
        }
        .onDisappear {
            telemetry.stopMonitoring()
        }
    }
    
    // MARK: - Header
    private var dashboardHeader: some View {
        HStack {
            Text("Backend Telemetry")
                .font(.headline)
            
            Spacer()
            
            // Time Range Selector
            Picker("Time Range", selection: $timeRange) {
                ForEach(TelemetryTimeRange.allCases, id: \.self) { range in
                    Text(range.rawValue).tag(range)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .frame(width: 250)
            
            Toggle("Auto Refresh", isOn: $autoRefresh)
                .toggleStyle(SwitchToggleStyle())
            
            Button(action: { telemetry.refresh() }) {
                Image(systemName: "arrow.clockwise")
            }
            .disabled(!telemetry.isConnected)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    // MARK: - Metrics Overview
    private var metricsOverview: some View {
        VStack(spacing: 0) {
            Text("System Metrics")
                .font(.subheadline)
                .fontWeight(.medium)
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            ScrollView {
                VStack(spacing: 12) {
                    // CPU Metric
                    MetricCard(
                        title: "CPU Usage",
                        value: "\(Int(telemetry.cpuUsage))%",
                        trend: telemetry.cpuTrend,
                        status: telemetry.cpuStatus,
                        icon: "cpu",
                        isSelected: selectedMetric == .cpu,
                        onTap: { selectedMetric = .cpu }
                    )
                    
                    // Memory Metric
                    MetricCard(
                        title: "Memory Usage",
                        value: formatMemory(telemetry.memoryUsed),
                        subtitle: "/ \(formatMemory(telemetry.memoryTotal))",
                        trend: telemetry.memoryTrend,
                        status: telemetry.memoryStatus,
                        icon: "memorychip",
                        isSelected: selectedMetric == .memory,
                        onTap: { selectedMetric = .memory }
                    )
                    
                    // GPU Metric
                    MetricCard(
                        title: "GPU Usage",
                        value: "\(Int(telemetry.gpuUsage))%",
                        subtitle: "VRAM: \(formatMemory(telemetry.vramUsed))",
                        trend: telemetry.gpuTrend,
                        status: telemetry.gpuStatus,
                        icon: "rectangle.3.group",
                        isSelected: selectedMetric == .gpu,
                        onTap: { selectedMetric = .gpu }
                    )
                    
                    // Disk I/O
                    MetricCard(
                        title: "Disk I/O",
                        value: "\(formatBandwidth(telemetry.diskReadRate))",
                        subtitle: "R: \(formatBandwidth(telemetry.diskReadRate)) W: \(formatBandwidth(telemetry.diskWriteRate))",
                        trend: .stable,
                        status: telemetry.diskStatus,
                        icon: "internaldrive",
                        isSelected: selectedMetric == .disk,
                        onTap: { selectedMetric = .disk }
                    )
                    
                    // Network
                    MetricCard(
                        title: "Network",
                        value: "\(formatBandwidth(telemetry.networkRate))",
                        subtitle: "↓ \(formatBandwidth(telemetry.downloadRate)) ↑ \(formatBandwidth(telemetry.uploadRate))",
                        trend: .stable,
                        status: telemetry.networkStatus,
                        icon: "network",
                        isSelected: selectedMetric == .network,
                        onTap: { selectedMetric = .network }
                    )
                    
                    // Pipeline Status
                    MetricCard(
                        title: "Pipeline",
                        value: telemetry.pipelineStatus.capitalized,
                        subtitle: telemetry.currentOperation,
                        trend: .stable,
                        status: telemetry.pipelineHealthStatus,
                        icon: "gearshape.2",
                        isSelected: selectedMetric == .pipeline,
                        onTap: { selectedMetric = .pipeline }
                    )
                    
                    Divider()
                        .padding(.vertical, 8)
                    
                    // Quick Stats
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Quick Stats")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        StatRow(label: "Uptime", value: formatUptime(telemetry.uptime))
                        StatRow(label: "Jobs Processed", value: "\(telemetry.jobsProcessed)")
                        StatRow(label: "Errors (24h)", value: "\(telemetry.errorCount24h)")
                        StatRow(label: "Avg Response", value: "\(Int(telemetry.avgResponseTime))ms")
                    }
                    .padding()
                    .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
                    .cornerRadius(8)
                }
                .padding()
            }
        }
    }
    
    // MARK: - Charts Panel
    private var chartsPanel: some View {
        VStack(spacing: 0) {
            // Chart Header
            HStack {
                Image(systemName: selectedMetric.icon)
                    .foregroundColor(.accentColor)
                
                Text(selectedMetric.rawValue)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                // Chart Type Selector
                Picker("Chart Type", selection: $telemetry.chartType) {
                    Image(systemName: "chart.line.uptrend.xyaxis").tag(ChartType.line)
                    Image(systemName: "chart.bar").tag(ChartType.bar)
                    Image(systemName: "rectangle.split.3x1").tag(ChartType.area)
                }
                .pickerStyle(SegmentedPickerStyle())
                .frame(width: 100)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Main Chart
            if #available(macOS 14.0, *) {
                Chart(telemetry.getDataPoints(for: selectedMetric, range: timeRange)) { point in
                    switch telemetry.chartType {
                    case .line:
                        LineMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Value", point.value)
                        )
                        .foregroundStyle(colorForMetric(selectedMetric))
                        
                    case .bar:
                        BarMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Value", point.value)
                        )
                        .foregroundStyle(colorForMetric(selectedMetric).opacity(0.7))
                        
                    case .area:
                        AreaMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Value", point.value)
                        )
                        .foregroundStyle(colorForMetric(selectedMetric).opacity(0.3))
                    }
                }
                .frame(maxHeight: .infinity)
                .padding()
            } else {
                // Fallback for older macOS versions
                MetricChartView(
                    data: telemetry.getDataPoints(for: selectedMetric, range: timeRange),
                    color: colorForMetric(selectedMetric)
                )
                .frame(maxHeight: .infinity)
                .padding()
            }
            
            // Chart Statistics
            HStack(spacing: 20) {
                ChartStat(label: "Current", value: formatValue(telemetry.currentValue(for: selectedMetric), metric: selectedMetric))
                ChartStat(label: "Average", value: formatValue(telemetry.averageValue(for: selectedMetric), metric: selectedMetric))
                ChartStat(label: "Peak", value: formatValue(telemetry.peakValue(for: selectedMetric), metric: selectedMetric))
                ChartStat(label: "Min", value: formatValue(telemetry.minValue(for: selectedMetric), metric: selectedMetric))
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
        }
    }
    
    // MARK: - Alerts Panel
    private var alertsPanel: some View {
        VStack(spacing: 0) {
            // Alerts Header
            HStack {
                Text("Alerts & Events")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                if telemetry.activeAlerts.count > 0 {
                    Text("\(telemetry.activeAlerts.count)")
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color.red)
                        .cornerRadius(10)
                }
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Alerts List
            List {
                Section("Active Alerts") {
                    if telemetry.activeAlerts.isEmpty {
                        Text("No active alerts")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    } else {
                        ForEach(telemetry.activeAlerts) { alert in
                            AlertRow(alert: alert)
                        }
                    }
                }
                
                Section("Recent Events") {
                    ForEach(telemetry.recentEvents.prefix(10)) { event in
                        EventRow(event: event)
                    }
                }
            }
            .listStyle(PlainListStyle())
        }
    }
    
    // MARK: - Status Bar
    private var statusBar: some View {
        HStack {
            // Connection Status
            HStack(spacing: 4) {
                Circle()
                    .fill(telemetry.isConnected ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                
                Text(telemetry.isConnected ? "Connected" : "Disconnected")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Divider()
                .frame(height: 20)
            
            // Backend Version
            Text("Backend v\(telemetry.backendVersion)")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Divider()
                .frame(height: 20)
            
            // Python Version
            Text("Python \(telemetry.pythonVersion)")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Spacer()
            
            // Last Update
            Text("Updated: \(telemetry.lastUpdateTime, formatter: timeFormatter)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    // MARK: - Helper Methods
    
    private func colorForMetric(_ metric: MetricType) -> Color {
        switch metric {
        case .cpu: return .blue
        case .memory: return .green
        case .gpu: return .orange
        case .disk: return .purple
        case .network: return .cyan
        case .pipeline: return .pink
        }
    }
    
    private func formatMemory(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter.string(fromByteCount: bytes)
    }
    
    private func formatBandwidth(_ bytesPerSecond: Double) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return "\(formatter.string(fromByteCount: Int64(bytesPerSecond)))/s"
    }
    
    private func formatUptime(_ seconds: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.unitsStyle = .abbreviated
        formatter.allowedUnits = [.day, .hour, .minute]
        return formatter.string(from: seconds) ?? "0m"
    }
    
    private func formatValue(_ value: Double, metric: MetricType) -> String {
        switch metric {
        case .cpu, .gpu:
            return "\(Int(value))%"
        case .memory:
            return formatMemory(Int64(value))
        case .disk, .network:
            return formatBandwidth(value)
        case .pipeline:
            return "\(Int(value))"
        }
    }
    
    private var timeFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.timeStyle = .medium
        return formatter
    }
}

// MARK: - Supporting Views

struct MetricCard: View {
    let title: String
    let value: String
    var subtitle: String? = nil
    let trend: TrendDirection
    let status: HealthStatus
    let icon: String
    let isSelected: Bool
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack {
                // Icon
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(status.color)
                    .frame(width: 30)
                
                // Content
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    HStack(alignment: .lastTextBaseline, spacing: 4) {
                        Text(value)
                            .font(.headline)
                            .fontWeight(.semibold)
                        
                        if let subtitle = subtitle {
                            Text(subtitle)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        // Trend Indicator
                        Image(systemName: trend.icon)
                            .font(.caption)
                            .foregroundColor(trend.color)
                    }
                }
                
                Spacer()
                
                // Status Indicator
                Circle()
                    .fill(status.color)
                    .frame(width: 8, height: 8)
            }
            .padding()
            .background(isSelected ? Color.accentColor.opacity(0.1) : Color(NSColor.controlBackgroundColor))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct StatRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }
}

struct ChartStat: View {
    let label: String
    let value: String
    
    var body: some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption)
                .fontWeight(.semibold)
        }
    }
}

struct AlertRow: View {
    let alert: TelemetryAlert
    
    var body: some View {
        HStack {
            Image(systemName: alert.severity.icon)
                .foregroundColor(alert.severity.color)
                .font(.caption)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(alert.title)
                    .font(.caption)
                    .fontWeight(.medium)
                
                Text(alert.message)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }
            
            Spacer()
            
            Text(alert.timestamp, formatter: timeOnlyFormatter)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
    
    private var timeOnlyFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter
    }
}

struct EventRow: View {
    let event: TelemetryEvent
    
    var body: some View {
        HStack {
            Circle()
                .fill(event.type.color)
                .frame(width: 6, height: 6)
            
            Text(event.message)
                .font(.caption)
                .lineLimit(1)
            
            Spacer()
            
            Text(event.timestamp, formatter: timeOnlyFormatter)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 2)
    }
    
    private var timeOnlyFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter
    }
}

// Fallback chart for older macOS
struct MetricChartView: View {
    let data: [DataPoint]
    let color: Color
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                guard !data.isEmpty else { return }
                
                let xScale = geometry.size.width / CGFloat(data.count - 1)
                let yMax = data.map { $0.value }.max() ?? 1
                let yScale = geometry.size.height / yMax
                
                for (index, point) in data.enumerated() {
                    let x = CGFloat(index) * xScale
                    let y = geometry.size.height - (point.value * yScale)
                    
                    if index == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
            }
            .stroke(color, lineWidth: 2)
        }
    }
}

// MARK: - Telemetry Manager

@MainActor
class TelemetryManager: ObservableObject {
    // Connection
    @Published var isConnected = false
    @Published var backendVersion = "Unknown"
    @Published var pythonVersion = "Unknown"
    @Published var lastUpdateTime = Date()
    
    // System Metrics
    @Published var cpuUsage: Double = 0
    @Published var cpuTrend = TrendDirection.stable
    @Published var cpuStatus = HealthStatus.good
    
    @Published var memoryUsed: Int64 = 0
    @Published var memoryTotal: Int64 = 16_000_000_000
    @Published var memoryTrend = TrendDirection.stable
    @Published var memoryStatus = HealthStatus.good
    
    @Published var gpuUsage: Double = 0
    @Published var vramUsed: Int64 = 0
    @Published var gpuTrend = TrendDirection.stable
    @Published var gpuStatus = HealthStatus.good
    
    @Published var diskReadRate: Double = 0
    @Published var diskWriteRate: Double = 0
    @Published var diskStatus = HealthStatus.good
    
    @Published var networkRate: Double = 0
    @Published var downloadRate: Double = 0
    @Published var uploadRate: Double = 0
    @Published var networkStatus = HealthStatus.good
    
    // Pipeline Metrics
    @Published var pipelineStatus = "idle"
    @Published var currentOperation = "None"
    @Published var pipelineHealthStatus = HealthStatus.good
    @Published var jobsProcessed = 0
    @Published var errorCount24h = 0
    @Published var avgResponseTime: Double = 0
    @Published var uptime: TimeInterval = 0
    
    // Alerts & Events
    @Published var activeAlerts: [TelemetryAlert] = []
    @Published var recentEvents: [TelemetryEvent] = []
    
    // Chart Settings
    @Published var chartType = ChartType.line
    
    // Data Storage
    private var dataHistory: [MetricType: [DataPoint]] = [:]
    private var refreshTimer: Timer?
    private let backendService = AutoResolveService()
    
    init() {
        // Initialize with sample data
        generateSampleData()
    }
    
    func startMonitoring() {
        isConnected = true
        
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            Task { @MainActor in
                self.updateMetrics()
            }
        }
    }
    
    func stopMonitoring() {
        refreshTimer?.invalidate()
        refreshTimer = nil
    }
    
    func refresh() {
        updateMetrics()
    }
    
    private func updateMetrics() {
        // Simulate metric updates
        cpuUsage = Double.random(in: 20...80)
        memoryUsed = Int64.random(in: 4_000_000_000...12_000_000_000)
        gpuUsage = Double.random(in: 10...60)
        vramUsed = Int64.random(in: 1_000_000_000...4_000_000_000)
        
        diskReadRate = Double.random(in: 1_000_000...50_000_000)
        diskWriteRate = Double.random(in: 1_000_000...30_000_000)
        
        downloadRate = Double.random(in: 100_000...5_000_000)
        uploadRate = Double.random(in: 50_000...1_000_000)
        networkRate = downloadRate + uploadRate
        
        jobsProcessed += Int.random(in: 0...2)
        uptime += 1
        
        lastUpdateTime = Date()
        
        // Update trends
        updateTrends()
        
        // Update status
        updateHealthStatus()
        
        // Store data points
        storeDataPoint()
    }
    
    private func updateTrends() {
        cpuTrend = cpuUsage > 70 ? .up : cpuUsage < 30 ? .down : .stable
        memoryTrend = Double(memoryUsed) / Double(memoryTotal) > 0.8 ? .up : .stable
        gpuTrend = gpuUsage > 50 ? .up : .stable
    }
    
    private func updateHealthStatus() {
        cpuStatus = cpuUsage > 90 ? .critical : cpuUsage > 75 ? .warning : .good
        memoryStatus = Double(memoryUsed) / Double(memoryTotal) > 0.9 ? .critical : 
                      Double(memoryUsed) / Double(memoryTotal) > 0.75 ? .warning : .good
        gpuStatus = gpuUsage > 90 ? .warning : .good
        diskStatus = (diskReadRate + diskWriteRate) > 100_000_000 ? .warning : .good
        networkStatus = networkRate > 10_000_000 ? .warning : .good
    }
    
    private func storeDataPoint() {
        let timestamp = Date()
        
        // Store for each metric type
        addDataPoint(.cpu, value: cpuUsage, at: timestamp)
        addDataPoint(.memory, value: Double(memoryUsed), at: timestamp)
        addDataPoint(.gpu, value: gpuUsage, at: timestamp)
        addDataPoint(.disk, value: diskReadRate + diskWriteRate, at: timestamp)
        addDataPoint(.network, value: networkRate, at: timestamp)
        addDataPoint(.pipeline, value: Double(jobsProcessed), at: timestamp)
    }
    
    private func addDataPoint(_ metric: MetricType, value: Double, at timestamp: Date) {
        if dataHistory[metric] == nil {
            dataHistory[metric] = []
        }
        
        dataHistory[metric]?.append(DataPoint(timestamp: timestamp, value: value))
        
        // Keep only last 1000 points
        if dataHistory[metric]?.count ?? 0 > 1000 {
            dataHistory[metric]?.removeFirst()
        }
    }
    
    func getDataPoints(for metric: MetricType, range: TelemetryTimeRange) -> [DataPoint] {
        let cutoff = Date().addingTimeInterval(-range.seconds)
        return dataHistory[metric]?.filter { $0.timestamp > cutoff } ?? []
    }
    
    func currentValue(for metric: MetricType) -> Double {
        switch metric {
        case .cpu: return cpuUsage
        case .memory: return Double(memoryUsed)
        case .gpu: return gpuUsage
        case .disk: return diskReadRate + diskWriteRate
        case .network: return networkRate
        case .pipeline: return Double(jobsProcessed)
        }
    }
    
    func averageValue(for metric: MetricType) -> Double {
        let points = dataHistory[metric] ?? []
        guard !points.isEmpty else { return 0 }
        return points.reduce(0) { $0 + $1.value } / Double(points.count)
    }
    
    func peakValue(for metric: MetricType) -> Double {
        dataHistory[metric]?.map { $0.value }.max() ?? 0
    }
    
    func minValue(for metric: MetricType) -> Double {
        dataHistory[metric]?.map { $0.value }.min() ?? 0
    }
    
    private func generateSampleData() {
        // Generate historical data for charts
        let now = Date()
        for i in 0..<100 {
            let timestamp = now.addingTimeInterval(TimeInterval(-i * 60))
            
            addDataPoint(.cpu, value: Double.random(in: 20...80), at: timestamp)
            addDataPoint(.memory, value: Double.random(in: 4_000_000_000...12_000_000_000), at: timestamp)
            addDataPoint(.gpu, value: Double.random(in: 10...60), at: timestamp)
            addDataPoint(.disk, value: Double.random(in: 2_000_000...80_000_000), at: timestamp)
            addDataPoint(.network, value: Double.random(in: 150_000...6_000_000), at: timestamp)
            addDataPoint(.pipeline, value: Double(i), at: timestamp)
        }
        
        // Generate sample alerts
        activeAlerts = [
            TelemetryAlert(
                title: "High Memory Usage",
                message: "Memory usage exceeds 80% threshold",
                severity: .warning,
                timestamp: Date()
            )
        ]
        
        // Generate sample events
        recentEvents = [
            TelemetryEvent(message: "Pipeline started", type: .info, timestamp: Date().addingTimeInterval(-300)),
            TelemetryEvent(message: "B-roll analysis completed", type: .success, timestamp: Date().addingTimeInterval(-240)),
            TelemetryEvent(message: "Silence detection finished", type: .success, timestamp: Date().addingTimeInterval(-180)),
            TelemetryEvent(message: "Export to Resolve initiated", type: .info, timestamp: Date().addingTimeInterval(-120)),
            TelemetryEvent(message: "Memory cleanup performed", type: .info, timestamp: Date().addingTimeInterval(-60))
        ]
    }
}

// MARK: - Data Models

struct DataPoint: Identifiable {
    let id = UUID()
    let timestamp: Date
    let value: Double
}

enum TrendDirection {
    case up, down, stable
    
    var icon: String {
        switch self {
        case .up: return "arrow.up.right"
        case .down: return "arrow.down.right"
        case .stable: return "arrow.right"
        }
    }
    
    var color: Color {
        switch self {
        case .up: return .red
        case .down: return .green
        case .stable: return .gray
        }
    }
}

enum HealthStatus {
    case good, warning, critical
    
    var color: Color {
        switch self {
        case .good: return .green
        case .warning: return .orange
        case .critical: return .red
        }
    }
}

struct TelemetryAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
    let severity: AlertSeverity
    let timestamp: Date
    
    enum AlertSeverity {
        case info, warning, critical
        
        var icon: String {
            switch self {
            case .info: return "info.circle"
            case .warning: return "exclamationmark.triangle"
            case .critical: return "exclamationmark.octagon"
            }
        }
        
        var color: Color {
            switch self {
            case .info: return .blue
            case .warning: return .orange
            case .critical: return .red
            }
        }
    }
}

struct TelemetryEvent: Identifiable {
    let id = UUID()
    let message: String
    let type: EventType
    let timestamp: Date
    
    enum EventType {
        case info, success, warning, error
        
        var color: Color {
            switch self {
            case .info: return .blue
            case .success: return .green
            case .warning: return .orange
            case .error: return .red
            }
        }
    }
}