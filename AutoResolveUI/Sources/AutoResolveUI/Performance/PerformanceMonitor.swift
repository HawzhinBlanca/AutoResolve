// AUTORESOLVE V3.0 - PERFORMANCE MONITOR
// Real-time performance monitoring and metrics collection

import SwiftUI
import Combine
import QuartzCore
import os.log

// MARK: - Performance Monitor
@MainActor
final class PerformanceMonitor: ObservableObject {
    static let shared = PerformanceMonitor()
    
    // Performance Metrics
    @Published var currentFPS: Double = 60
    @Published var averageFPS: Double = 60
    @Published var memoryUsage: Double = 0 // MB
    @Published var cpuUsage: Double = 0 // Percentage
    @Published var diskUsage: Double = 0 // MB
    @Published var networkLatency: Double = 0 // ms
    @Published var cacheHitRate: Double = 0 // Percentage
    
    // Frame timing
    @Published var frameTime: Double = 0 // ms
    @Published var renderTime: Double = 0 // ms
    @Published var updateTime: Double = 0 // ms
    
    // Thresholds
    let targetFPS: Double = 60
    let maxMemoryMB: Double = 500
    let maxCPUPercent: Double = 50
    
    // Alerts
    @Published var performanceAlerts: [PerformanceAlert] = []
    
    // Internal state
    private var fpsTimer: Timer?
    private var frameTimestamps: [TimeInterval] = []
    private var lastFrameTime: TimeInterval = 0
    private var cpuTimer: Timer?
    private var memoryTimer: Timer?
    private let logger = Logger.shared
    
    // Metrics history
    private var fpsHistory: [Double] = []
    private var memoryHistory: [Double] = []
    private var cpuHistory: [Double] = []
    private let historySize = 60 // Keep 60 seconds of history
    
    // MARK: - Performance Alert
    struct PerformanceAlert: Identifiable {
        public let id = UUID()
        let type: AlertType
        let message: String
        let severity: Severity
        let timestamp: Date
        
        enum AlertType {
            case fps, memory, cpu, network, disk
        }
        
        enum Severity {
            case info, warning, critical
            
            var color: Color {
                switch self {
                case .info: return .blue
                case .warning: return .yellow
                case .critical: return .red
                }
            }
        }
    }
    
    // MARK: - Initialization
    private init() {
        startMonitoring()
    }
    
    // MARK: - Monitoring Control
    func startMonitoring() {
        startFPSMonitoring()
        startMemoryMonitoring()
        startCPUMonitoring()
        
        logger.info("Performance monitoring started")
    }
    
    func stopMonitoring() {
        fpsTimer?.invalidate()
        fpsTimer = nil
        
        cpuTimer?.invalidate()
        cpuTimer = nil
        
        memoryTimer?.invalidate()
        memoryTimer = nil
        
        logger.info("Performance monitoring stopped")
    }
    
    // MARK: - FPS Monitoring
    private func startFPSMonitoring() {
        // Use a timer for FPS monitoring on macOS
        fpsTimer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.updateFPS()
            }
        }
    }
    
    private func updateFPS() {
        let currentTime = CACurrentMediaTime()
        
        if lastFrameTime > 0 {
            let deltaTime = currentTime - lastFrameTime
            let fps = 1.0 / deltaTime
            
            // Update current FPS
            currentFPS = fps
            frameTime = deltaTime * 1000 // Convert to ms
            
            // Track history
            frameTimestamps.append(currentTime)
            frameTimestamps = Array(frameTimestamps.suffix(60)) // Keep last 60 frames
            
            // Calculate average FPS
            if frameTimestamps.count > 1 {
                let timeSpan = frameTimestamps.last! - frameTimestamps.first!
                averageFPS = Double(frameTimestamps.count - 1) / timeSpan
            }
            
            // Check for performance issues
            if fps < targetFPS * 0.9 { // Below 90% of target
                checkFPSPerformance(fps)
            }
            
            // Update history
            fpsHistory.append(fps)
            if fpsHistory.count > historySize {
                fpsHistory.removeFirst()
            }
        }
        
        lastFrameTime = currentTime
    }
    
    private func checkFPSPerformance(_ fps: Double) {
        if fps < 30 {
            addAlert(
                type: .fps,
                message: "Critical: FPS dropped to \(Int(fps))",
                severity: .critical
            )
        } else if fps < 45 {
            addAlert(
                type: .fps,
                message: "Warning: FPS at \(Int(fps))",
                severity: .warning
            )
        }
    }
    
    // MARK: - Memory Monitoring
    private func startMemoryMonitoring() {
        memoryTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            Task { @MainActor in
                self.updateMemoryUsage()
            }
        }
    }
    
    private func updateMemoryUsage() {
        let info = ProcessInfo.processInfo
        _ = info.physicalMemory
        
        var vmInfo = vm_statistics_data_t()
        var vmInfoSize = mach_msg_type_number_t(MemoryLayout<vm_statistics_data_t>.size / MemoryLayout<natural_t>.size)
        
        let result = withUnsafeMutablePointer(to: &vmInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmInfoSize)) {
                host_statistics(mach_host_self(), HOST_VM_INFO, $0, &vmInfoSize)
            }
        }
        
        if result == KERN_SUCCESS {
            let pageSize = vm_kernel_page_size
            let memoryUsedPages = vmInfo.active_count + vmInfo.wire_count
            let memoryUsedBytes = UInt64(memoryUsedPages) * UInt64(pageSize)
            
            memoryUsage = Double(memoryUsedBytes) / 1_048_576 // Convert to MB
            
            // Check memory threshold
            if memoryUsage > maxMemoryMB {
                addAlert(
                    type: .memory,
                    message: "Memory usage exceeds \(Int(maxMemoryMB))MB",
                    severity: .warning
                )
            }
            
            // Update history
            memoryHistory.append(memoryUsage)
            if memoryHistory.count > historySize {
                memoryHistory.removeFirst()
            }
        }
    }
    
    // MARK: - CPU Monitoring
    private func startCPUMonitoring() {
        cpuTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            Task { @MainActor in
                self.updateCPUUsage()
            }
        }
    }
    
    private func updateCPUUsage() {
        var cpuInfo: processor_info_array_t!
        var numCpuInfo: mach_msg_type_number_t = 0
        var numCpus: natural_t = 0
        
        let result = host_processor_info(
            mach_host_self(),
            PROCESSOR_CPU_LOAD_INFO,
            &numCpus,
            &cpuInfo,
            &numCpuInfo
        )
        
        guard result == KERN_SUCCESS else { return }
        
        var totalUser = 0.0
        var totalSystem = 0.0
        var totalIdle = 0.0
        
        for i in 0..<Int32(numCpus) {
            let offset = Int(i) * Int(CPU_STATE_MAX)
            let user = Double(cpuInfo[offset + Int(CPU_STATE_USER)])
            let system = Double(cpuInfo[offset + Int(CPU_STATE_SYSTEM)])
            let idle = Double(cpuInfo[offset + Int(CPU_STATE_IDLE)])
            
            totalUser += user
            totalSystem += system
            totalIdle += idle
        }
        
        let total = totalUser + totalSystem + totalIdle
        if total > 0 {
            cpuUsage = ((totalUser + totalSystem) / total) * 100
            
            // Check CPU threshold
            if cpuUsage > maxCPUPercent {
                addAlert(
                    type: .cpu,
                    message: "CPU usage at \(Int(cpuUsage))%",
                    severity: cpuUsage > 80 ? .critical : .warning
                )
            }
            
            // Update history
            cpuHistory.append(cpuUsage)
            if cpuHistory.count > historySize {
                cpuHistory.removeFirst()
            }
        }
        
        // Deallocate memory
        let cpuInfoSize = Int(numCpuInfo)
        let cpuInfoPtr = UnsafeMutablePointer<integer_t>(cpuInfo)
        vm_deallocate(mach_task_self_, vm_address_t(bitPattern: cpuInfoPtr), vm_size_t(cpuInfoSize))
    }
    
    // MARK: - Network Monitoring
    func recordNetworkLatency(_ latency: TimeInterval) {
        networkLatency = latency * 1000 // Convert to ms
        
        if networkLatency > 500 {
            addAlert(
                type: .network,
                message: "High network latency: \(Int(networkLatency))ms",
                severity: networkLatency > 1000 ? .critical : .warning
            )
        }
    }
    
    // MARK: - Cache Monitoring
    func updateCacheHitRate(_ hits: Int, misses: Int) {
        let total = hits + misses
        cacheHitRate = total > 0 ? (Double(hits) / Double(total)) * 100 : 0
        
        if cacheHitRate < 50 && total > 100 {
            addAlert(
                type: .memory,
                message: "Low cache hit rate: \(Int(cacheHitRate))%",
                severity: .info
            )
        }
    }
    
    // MARK: - Alerts
    private func addAlert(type: PerformanceAlert.AlertType, message: String, severity: PerformanceAlert.Severity) {
        let alert = PerformanceAlert(
            type: type,
            message: message,
            severity: severity,
            timestamp: Date()
        )
        
        performanceAlerts.append(alert)
        
        // Keep only recent alerts
        performanceAlerts = Array(performanceAlerts.suffix(20))
        
        // Log based on severity
        switch severity {
        case .info:
            logger.info("\(message)")
        case .warning:
            logger.warning("\(message)")
        case .critical:
            logger.error("\(message)")
        }
    }
    
    func clearAlerts() {
        performanceAlerts.removeAll()
    }
    
    // MARK: - Performance Score
    var performanceScore: Double {
        var score = 100.0
        
        // FPS impact (40% weight)
        let fpsRatio = averageFPS / targetFPS
        score -= (1 - min(1, fpsRatio)) * 40
        
        // Memory impact (30% weight)
        let memoryRatio = memoryUsage / maxMemoryMB
        score -= max(0, memoryRatio - 1) * 30
        
        // CPU impact (20% weight)
        let cpuRatio = cpuUsage / maxCPUPercent
        score -= max(0, cpuRatio - 1) * 20
        
        // Cache impact (10% weight)
        let cacheRatio = cacheHitRate / 100
        score -= (1 - cacheRatio) * 10
        
        return max(0, min(100, score))
    }
    
    var performanceGrade: String {
        switch performanceScore {
        case 90...100: return "A"
        case 80..<90: return "B"
        case 70..<80: return "C"
        case 60..<70: return "D"
        default: return "F"
        }
    }
    
    // MARK: - Statistics
    func generateReport() -> PerformanceReport {
        PerformanceReport(
            averageFPS: averageFPS,
            minFPS: fpsHistory.min() ?? 0,
            maxFPS: fpsHistory.max() ?? 0,
            averageMemory: memoryHistory.reduce(0, +) / Double(max(1, memoryHistory.count)),
            peakMemory: memoryHistory.max() ?? 0,
            averageCPU: cpuHistory.reduce(0, +) / Double(max(1, cpuHistory.count)),
            peakCPU: cpuHistory.max() ?? 0,
            performanceScore: performanceScore,
            alertCount: performanceAlerts.count,
            timestamp: Date()
        )
    }
    
    struct PerformanceReport {
        let averageFPS: Double
        let minFPS: Double
        let maxFPS: Double
        let averageMemory: Double
        let peakMemory: Double
        let averageCPU: Double
        let peakCPU: Double
        let performanceScore: Double
        let alertCount: Int
        let timestamp: Date
    }
}

// MARK: - Performance Dashboard View
struct PerformanceDashboard: View {
    @ObservedObject private var monitor = PerformanceMonitor.shared
    @State private var showDetails = false
    @State private var selectedMetric: MetricType = .fps
    
    enum MetricType: String, CaseIterable {
        case fps = "FPS"
        case memory = "Memory"
        case cpu = "CPU"
        case network = "Network"
        
        var icon: String {
            switch self {
            case .fps: return "speedometer"
            case .memory: return "memorychip"
            case .cpu: return "cpu"
            case .network: return "network"
            }
        }
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Label("Performance Monitor", systemImage: "chart.line.uptrend.xyaxis")
                    .font(.system(size: 14, weight: .semibold))
                
                Spacer()
                
                // Performance Score
                PerformanceScoreBadge(score: monitor.performanceScore, grade: monitor.performanceGrade)
                
                Button(action: { showDetails.toggle() }) {
                    Image(systemName: showDetails ? "chevron.up" : "chevron.down")
                }
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Quick Metrics
            HStack(spacing: 20) {
                PerformanceMetricCard(
                    title: "FPS",
                    value: String(format: "%.0f", monitor.currentFPS),
                    target: String(format: "%.0f", monitor.targetFPS),
                    icon: "speedometer",
                    color: monitor.currentFPS >= monitor.targetFPS * 0.9 ? .green : .orange
                )
                
                PerformanceMetricCard(
                    title: "Memory",
                    value: String(format: "%.0f MB", monitor.memoryUsage),
                    target: String(format: "%.0f MB", monitor.maxMemoryMB),
                    icon: "memorychip",
                    color: monitor.memoryUsage <= monitor.maxMemoryMB ? .green : .orange
                )
                
                PerformanceMetricCard(
                    title: "CPU",
                    value: String(format: "%.0f%%", monitor.cpuUsage),
                    target: String(format: "%.0f%%", monitor.maxCPUPercent),
                    icon: "cpu",
                    color: monitor.cpuUsage <= monitor.maxCPUPercent ? .green : .orange
                )
                
                PerformanceMetricCard(
                    title: "Cache",
                    value: String(format: "%.0f%%", monitor.cacheHitRate),
                    target: ">80%",
                    icon: "archivebox",
                    color: monitor.cacheHitRate >= 80 ? .green : .orange
                )
            }
            .padding()
            
            if showDetails {
                Divider()
                
                // Detailed Metrics
                VStack(spacing: 12) {
                    // Metric selector
                    Picker("Metric", selection: $selectedMetric) {
                        ForEach(MetricType.allCases, id: \.self) { metric in
                            Label(metric.rawValue, systemImage: metric.icon)
                                .tag(metric)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding(.horizontal)
                    
                    // Metric details
                    MetricDetailView(type: selectedMetric, monitor: monitor)
                    
                    // Alerts
                    if !monitor.performanceAlerts.isEmpty {
                        AlertsView(alerts: monitor.performanceAlerts)
                    }
                }
                .padding()
            }
        }
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(8)
        .shadow(radius: 2)
    }
}

// MARK: - Supporting Views
struct PerformanceScoreBadge: View {
    let score: Double
    let grade: String
    
    private var color: Color {
        switch grade {
        case "A": return .green
        case "B": return .blue
        case "C": return .yellow
        case "D": return .orange
        default: return .red
        }
    }
    
    public var body: some View {
        HStack(spacing: 4) {
            Text(grade)
                .font(.system(size: 16, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .frame(width: 24, height: 24)
                .background(Circle().fill(color))
            
            Text(String(format: "%.0f%%", score))
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.secondary)
        }
    }
}

struct PerformanceMetricCard: View {
    let title: String
    let value: String
    let target: String
    let icon: String
    let color: Color
    
    public var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundColor(color)
            
            Text(value)
                .font(.system(size: 14, weight: .semibold))
            
            Text(title)
                .font(.system(size: 10))
                .foregroundColor(.secondary)
            
            Text("Target: \(target)")
                .font(.system(size: 9))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct MetricDetailView: View {
    let type: PerformanceDashboard.MetricType
    @ObservedObject var monitor: PerformanceMonitor
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch type {
            case .fps:
                PerformanceDetailRow(label: "Current", value: String(format: "%.1f fps", monitor.currentFPS))
                PerformanceDetailRow(label: "Average", value: String(format: "%.1f fps", monitor.averageFPS))
                PerformanceDetailRow(label: "Frame Time", value: String(format: "%.2f ms", monitor.frameTime))
                PerformanceDetailRow(label: "Render Time", value: String(format: "%.2f ms", monitor.renderTime))
                
            case .memory:
                PerformanceDetailRow(label: "Usage", value: String(format: "%.1f MB", monitor.memoryUsage))
                PerformanceDetailRow(label: "Frame Pool", value: FrameMemoryPool.shared.formattedMemoryUsage)
                PerformanceDetailRow(label: "Cache Hit Rate", value: String(format: "%.1f%%", monitor.cacheHitRate))
                
            case .cpu:
                PerformanceDetailRow(label: "Usage", value: String(format: "%.1f%%", monitor.cpuUsage))
                PerformanceDetailRow(label: "Cores", value: "\(ProcessInfo.processInfo.processorCount)")
                
            case .network:
                PerformanceDetailRow(label: "Latency", value: String(format: "%.0f ms", monitor.networkLatency))
                PerformanceDetailRow(label: "Batch Efficiency", value: String(format: "%.1f%%", RequestBatcher.shared.efficiency * 100))
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(6)
    }
}

struct PerformanceDetailRow: View {
    let label: String
    let value: String
    
    public var body: some View {
        HStack {
            Text(label)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 11, weight: .medium))
        }
    }
}

struct AlertsView: View {
    let alerts: [PerformanceMonitor.PerformanceAlert]
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Recent Alerts")
                .font(.system(size: 12, weight: .semibold))
            
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(alerts.reversed()) { alert in
                        HStack(spacing: 4) {
                            Circle()
                                .fill(alert.severity.color)
                                .frame(width: 6, height: 6)
                            
                            Text(alert.message)
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                            
                            Spacer()
                            
                            Text(alert.timestamp, style: .time)
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .frame(maxHeight: 100)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(6)
    }
}
