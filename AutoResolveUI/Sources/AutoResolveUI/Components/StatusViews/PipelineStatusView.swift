import SwiftUI

public struct PipelineStatusView: View {
    @StateObject private var telemetry = PipelineStatusMonitor()
    @State private var isExpanded = false
    @State private var selectedTab = 0
    
    public var body: some View {
        VStack(spacing: 0) {
            // Status Header Bar
            statusHeaderBar
            
            if isExpanded {
                // Detailed Status Panel
                TabView(selection: $selectedTab) {
                    // Overview Tab
                    OverviewTab(telemetry: telemetry)
                        .tabItem {
                            Image(systemName: "chart.line.uptrend.xyaxis")
                            Text("Overview")
                        }
                        .tag(0)
                    
                    // Jobs Tab
                    JobsTab(telemetry: telemetry)
                        .tabItem {
                            Image(systemName: "list.bullet.rectangle")
                            Text("Jobs")
                        }
                        .tag(1)
                    
                    // Performance Tab
                    PerformanceTab(telemetry: telemetry)
                        .tabItem {
                            Image(systemName: "speedometer")
                            Text("Performance")
                        }
                        .tag(2)
                    
                    // Logs Tab
                    LogsTab(telemetry: telemetry)
                        .tabItem {
                            Image(systemName: "terminal")
                            Text("Logs")
                        }
                        .tag(3)
                }
                .frame(height: 300)
            }
        }
        .background(Color(.windowBackgroundColor))
        .cornerRadius(8)
        .shadow(radius: 2)
    }
    
    private var statusHeaderBar: some View {
        HStack {
            // Status Indicator
            HStack(spacing: 8) {
                StatusIndicator(status: telemetry.currentStatus)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(telemetry.currentStatus.displayName)
                        .font(.headline)
                        .foregroundColor(telemetry.currentStatus.color)
                    
                    if !telemetry.currentOperation.isEmpty {
                        Text(telemetry.currentOperation)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Spacer()
            
            // Progress and Stats
            if telemetry.isProcessing {
                VStack(alignment: .trailing, spacing: 2) {
                    HStack(spacing: 8) {
                        Text("\(Int(telemetry.progressPercentage * 100))%")
                            .font(.caption)
                            .monospacedDigit()
                        
                        ProgressView(value: telemetry.progressPercentage)
                            .frame(width: 80)
                    }
                    
                    if telemetry.estimatedTimeRemaining > 0 {
                        Text("ETA: \(formatTimeInterval(telemetry.estimatedTimeRemaining))")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            // Active Jobs Badge
            if telemetry.activeJobCount > 0 {
                StatusBadge(count: telemetry.activeJobCount, color: .blue)
            }
            
            // Error Badge
            if telemetry.hasErrors {
                StatusBadge(count: telemetry.errorMessages.count, color: .red)
            }
            
            // Expand/Collapse Button
            Button(action: { 
                withAnimation(.easeInOut(duration: 0.3)) {
                    isExpanded.toggle()
                }
            }) {
                Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                    .foregroundColor(.secondary)
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(.controlBackgroundColor))
    }
}

// MARK: - Tab Views

private struct OverviewTab: View {
    @ObservedObject var telemetry: PipelineStatusMonitor
    
    var body: some View {
        ScrollView {
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 16) {
                // Current Status Card
                StatusCard(
                    title: "Current Status",
                    value: telemetry.currentStatus.displayName,
                    color: telemetry.currentStatus.color,
                    icon: "play.circle"
                )
                
                // Progress Card
                StatusCard(
                    title: "Progress",
                    value: "\(Int(telemetry.progressPercentage * 100))%",
                    color: .blue,
                    icon: "chart.line.uptrend.xyaxis"
                )
                
                // Active Jobs Card
                StatusCard(
                    title: "Active Jobs",
                    value: "\(telemetry.activeJobCount)",
                    color: .green,
                    icon: "gearshape.2"
                )
                
                // Memory Usage Card
                StatusCard(
                    title: "Memory",
                    value: "\(Int(telemetry.performanceMetrics.memoryUsedMB)) MB",
                    color: telemetry.performanceMetrics.memoryUsagePercent > 80 ? .red : .orange,
                    icon: "memorychip"
                )
                
                // Throughput Card
                StatusCard(
                    title: "Throughput",
                    value: "\(String(format: "%.1f", telemetry.throughputMBps)) MB/s",
                    color: .purple,
                    icon: "speedometer"
                )
                
                // Completed Jobs Card
                StatusCard(
                    title: "Completed",
                    value: "\(telemetry.totalJobsCompleted)",
                    color: .green,
                    icon: "checkmark.circle"
                )
            }
            .padding()
        }
    }
}

private struct JobsTab: View {
    @ObservedObject var telemetry: PipelineStatusMonitor
    
    public var body: some View {
        VStack {
            // Jobs Header
            HStack {
                Text("Processing Jobs")
                    .font(.headline)
                
                Spacer()
                
                Button("Clear Completed") {
                    telemetry.clearCompletedJobs()
                }
                .disabled(telemetry.totalJobsCompleted == 0)
            }
            .padding(.horizontal)
            
            // Jobs List
            if telemetry.activeJobs.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "tray")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    Text("No active jobs")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(telemetry.activeJobs) { job in
                    JobRowView(job: job)
                }
                .listStyle(PlainListStyle())
            }
        }
    }
}

private struct PerformanceTab: View {
    @ObservedObject var telemetry: PipelineStatusMonitor
    
    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Performance Charts
                HStack(spacing: 16) {
                    PerformanceChart(
                        title: "Memory Usage",
                        value: telemetry.performanceMetrics.memoryUsagePercent,
                        color: telemetry.performanceMetrics.memoryUsagePercent > 80 ? .red : .blue,
                        unit: "%"
                    )
                    
                    PerformanceChart(
                        title: "CPU Usage",
                        value: telemetry.performanceMetrics.cpuUsagePercent,
                        color: telemetry.performanceMetrics.cpuUsagePercent > 80 ? .red : .green,
                        unit: "%"
                    )
                }
                
                // Throughput Metrics
                HStack(spacing: 16) {
                    MetricView(
                        title: "Disk Read",
                        value: "\(String(format: "%.1f", telemetry.performanceMetrics.diskReadMBps)) MB/s",
                        icon: "internaldrive"
                    )
                    
                    MetricView(
                        title: "Disk Write",
                        value: "\(String(format: "%.1f", telemetry.performanceMetrics.diskWriteMBps)) MB/s",
                        icon: "externaldrive"
                    )
                    
                    MetricView(
                        title: "Frames/sec",
                        value: "\(String(format: "%.0f", telemetry.performanceMetrics.framesProcessedPerSecond))",
                        icon: "play.rectangle"
                    )
                }
            }
            .padding()
        }
    }
}

private struct LogsTab: View {
    @ObservedObject var telemetry: PipelineStatusMonitor
    @State private var selectedLogLevel: PipelineStatusMonitor.StatusMessage.LogLevel = .info
    
    public var body: some View {
        VStack {
            // Log Level Filter
            HStack {
                Text("Filter:")
                    .font(.caption)
                
                Picker("Log Level", selection: $selectedLogLevel) {
                    ForEach(PipelineStatusMonitor.StatusMessage.LogLevel.allCases, id: \.self) { level in
                        Text(level.rawValue.capitalized)
                            .tag(level)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                
                Spacer()
                
                Button("Clear") {
                    telemetry.recentMessages.removeAll()
                }
            }
            .padding(.horizontal)
            
            // Messages List
            ScrollViewReader { proxy in
                List {
                    ForEach(filteredMessages) { message in
                        LogMessageRow(message: message)
                    }
                }
                .listStyle(PlainListStyle())
                .onChange(of: telemetry.recentMessages.count) { _, _ in
                    if let lastMessage = telemetry.recentMessages.last {
                        proxy.scrollTo(lastMessage.id)
                    }
                }
            }
        }
    }
    
    private var filteredMessages: [PipelineStatusMonitor.StatusMessage] {
        telemetry.recentMessages.filter { message in
            switch selectedLogLevel {
            case .debug:
                return true
            case .info:
                return message.level != .debug
            case .warning:
                return message.level == .warning || message.level == .error
            case .error:
                return message.level == .error
            }
        }
    }
}

// MARK: - Supporting Views

private struct StatusIndicator: View {
    let status: PipelineStatusMonitor.PipelineStatus
    
    public var body: some View {
        Circle()
            .fill(status.color)
            .frame(width: 12, height: 12)
            .overlay(
                Circle()
                    .stroke(Color.white, lineWidth: 1)
            )
            .scaleEffect(status == .error ? 1.2 : 1.0)
            .animation(.easeInOut(duration: 0.3), value: status)
    }
}

private struct StatusBadge: View {
    let count: Int
    let color: Color
    
    public var body: some View {
        Text("\(count)")
            .font(.caption2)
            .fontWeight(.bold)
            .foregroundColor(.white)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color)
            .clipShape(Capsule())
    }
}

private struct StatusCard: View {
    let title: String
    let value: String
    let color: Color
    let icon: String
    
    public var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title2)
                
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(value)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(color)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.controlBackgroundColor))
        .cornerRadius(8)
    }
}

private struct JobRowView: View {
    let job: PipelineStatusMonitor.ProcessingJob
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(job.jobType)
                    .font(.headline)
                
                Spacer()
                
                Text(job.status.displayName)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(job.status.color.opacity(0.2))
                    .foregroundColor(job.status.color)
                    .cornerRadius(4)
            }
            
            if job.progress > 0 {
                ProgressView(value: job.progress)
                    .progressViewStyle(LinearProgressViewStyle())
            }
            
            HStack {
                Text(URL(fileURLWithPath: job.inputPath).lastPathComponent)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text("Duration: \(formatTimeInterval(job.duration))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            if let error = job.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }
}

private struct PerformanceChart: View {
    let title: String
    let value: Double
    let color: Color
    let unit: String
    
    public var body: some View {
        VStack {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            ZStack {
                Circle()
                    .stroke(color.opacity(0.2), lineWidth: 8)
                
                Circle()
                    .trim(from: 0, to: CGFloat(min(value / 100.0, 1.0)))
                    .stroke(color, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                
                Text("\(Int(value))\(unit)")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(color)
            }
            .frame(width: 60, height: 60)
        }
        .padding()
        .background(Color(.controlBackgroundColor))
        .cornerRadius(8)
    }
}

private struct MetricView: View {
    let title: String
    let value: String
    let icon: String
    
    public var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
            
            Text(value)
                .font(.headline)
                .fontWeight(.bold)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.controlBackgroundColor))
        .cornerRadius(8)
    }
}

private struct LogMessageRow: View {
    let message: PipelineStatusMonitor.StatusMessage
    
    public var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Circle()
                .fill(message.level.color)
                .frame(width: 6, height: 6)
                .padding(.top, 6)
            
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(message.component)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(message.level.color)
                    
                    Spacer()
                    
                    Text(DateFormatter.timeOnlyFormatter.string(from: message.timestamp))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                
                Text(message.message)
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Utilities


private extension DateFormatter {
    static let timeOnlyFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()
}
