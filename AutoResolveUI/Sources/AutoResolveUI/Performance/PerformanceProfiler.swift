// AUTORESOLVE V3.0 - PERFORMANCE PROFILER
import Combine
// Code instrumentation and performance analysis

import Foundation
import SwiftUI
import os.signpost
import os.log

// MARK: - Performance Profiler
@MainActor
final class PerformanceProfiler: ObservableObject {
    static let shared = PerformanceProfiler()
    
    // Profiling data
    @Published var traces: [PerformanceTrace] = []
    @Published var methodTimings: [String: MethodTiming] = [:]
    @Published var currentOperations: [String: Date] = [:]
    @Published var isRecording = false
    
    // Signpost logging
    private let log = OSLog(subsystem: "AutoResolve", category: .pointsOfInterest)
    private var signpostIDs: [String: OSSignpostID] = [:]
    
    // Configuration
    var maxTraces = 1000
    var autoExport = false
    
    private let logger = Logger.shared
    
    // MARK: - Performance Trace
    struct PerformanceTrace: Identifiable {
        public let id = UUID()
        let name: String
        let category: Category
        let startTime: Date
        var endTime: Date?
        let metadata: [String: Any]
        
        init(name: String, category: Category, startTime: Date, endTime: Date? = nil, metadata: [String: Any] = [:]) {
            // id is already initialized in the property declaration
            self.name = name
            self.category = category
            self.startTime = startTime
            self.endTime = endTime
            self.metadata = metadata
        }
        
        var duration: TimeInterval? {
            guard let endTime = endTime else { return nil }
            return endTime.timeIntervalSince(startTime)
        }
        
        enum Category: Hashable {
            case ui
            case network
            case disk
            case computation
            case rendering
            case cache
            case render
            case custom(String)
            
            var color: Color {
                switch self {
                case .ui: return .blue
                case .network: return .green
                case .disk: return .orange
                case .computation: return .purple
                case .rendering: return .red
                case .cache: return .cyan
                case .render: return .pink
                case .custom: return .gray
                }
            }
        }
    }
    
    // MARK: - Method Timing
    struct MethodTiming: Identifiable {
        public let id = UUID()
        let method: String
        var callCount: Int = 0
        var totalTime: TimeInterval = 0
        var minTime: TimeInterval = .infinity
        var maxTime: TimeInterval = 0
        var lastCalled: Date?
        
        var averageTime: TimeInterval {
            callCount > 0 ? totalTime / Double(callCount) : 0
        }
    }
    
    // MARK: - Recording Control
    func startRecording() {
        traces.removeAll()
        methodTimings.removeAll()
        currentOperations.removeAll()
        isRecording = true
        
        logger.info("Performance recording started")
    }
    
    func stopRecording() {
        isRecording = false
        
        // Complete any pending operations
        for (name, _) in currentOperations {
            end(name)
        }
        
        logger.info("Performance recording stopped with \(self.traces.count) traces")
        
        if autoExport {
            exportTraces()
        }
    }
    
    // MARK: - Trace Management
    @discardableResult
    func begin(_ name: String, category: PerformanceTrace.Category = .custom("General"), metadata: [String: Any] = [:]) -> String {
        guard isRecording else { return name }
        
        let trace = PerformanceTrace(
            name: name,
            category: category,
            startTime: Date(),
            metadata: metadata
        )
        
        traces.append(trace)
        currentOperations[name] = trace.startTime
        
        // Limit trace count
        if traces.count > maxTraces {
            traces.removeFirst()
        }
        
        // OS Signpost
        let signpostID = OSSignpostID(log: log)
        signpostIDs[name] = signpostID
        os_signpost(.begin, log: log, name: "Performance_Measurement", signpostID: signpostID)
        
        return name
    }
    
    func end(_ name: String) {
        guard isRecording else { return }
        guard let startTime = currentOperations.removeValue(forKey: name) else { return }
        
        let endTime = Date()
        let duration = endTime.timeIntervalSince(startTime)
        
        // Update trace
        if let index = traces.lastIndex(where: { $0.name == name && $0.endTime == nil }) {
            traces[index].endTime = endTime
        }
        
        // Update method timing
        var timing = methodTimings[name] ?? MethodTiming(method: name)
        timing.callCount += 1
        timing.totalTime += duration
        timing.minTime = min(timing.minTime, duration)
        timing.maxTime = max(timing.maxTime, duration)
        timing.lastCalled = endTime
        methodTimings[name] = timing
        
        // OS Signpost
        if let signpostID = signpostIDs.removeValue(forKey: name) {
            os_signpost(.end, log: log, name: "Performance_Measurement", signpostID: signpostID)
        }
    }
    
    // MARK: - Convenience Methods
    func measure<T>(_ name: String, category: PerformanceTrace.Category = .computation, block: () throws -> T) rethrows -> T {
        begin(name, category: category)
        defer { end(name) }
        return try block()
    }
    
    func measureAsync<T>(_ name: String, category: PerformanceTrace.Category = .computation, block: () async throws -> T) async rethrows -> T {
        begin(name, category: category)
        defer { end(name) }
        return try await block()
    }
    
    // MARK: - Analysis
    func getSlowOperations(threshold: TimeInterval = 0.1) -> [PerformanceTrace] {
        traces.filter { ($0.duration ?? 0) > threshold }
            .sorted { ($0.duration ?? 0) > ($1.duration ?? 0) }
    }
    
    func getHotMethods(minCalls: Int = 10) -> [MethodTiming] {
        methodTimings.values
            .filter { $0.callCount >= minCalls }
            .sorted { $0.totalTime > $1.totalTime }
    }
    
    func getCategoryBreakdown() -> [PerformanceTrace.Category: TimeInterval] {
        var breakdown: [PerformanceTrace.Category: TimeInterval] = [:]
        
        for trace in traces {
            if let duration = trace.duration {
                breakdown[trace.category, default: 0] += duration
            }
        }
        
        return breakdown
    }
    
    // MARK: - Export
    func exportTraces() {
        let report = generateReport()
        
        // Save to file
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsURL.appendingPathComponent("AutoResolve_Performance_\(Date().timeIntervalSince1970).json")
        
        do {
            let data = try JSONEncoder().encode(report)
            try data.write(to: fileURL)
            logger.info("Performance report exported to \(fileURL.path)")
        } catch {
            logger.error("Failed to export performance report: \(error)")
        }
    }
    
    private func generateReport() -> ProfileReport {
        ProfileReport(
            timestamp: Date(),
            traceCount: traces.count,
            totalDuration: traces.compactMap { $0.duration }.reduce(0, +),
            slowestOperation: getSlowOperations().first?.name ?? "N/A",
            hottestMethod: getHotMethods().first?.method ?? "N/A",
            categoryBreakdown: Dictionary(uniqueKeysWithValues: getCategoryBreakdown().map { (String(describing: $0.key), $0.value) }),
            traces: traces.map { ProfileReport.TraceInfo(from: $0) }
        )
    }
    
    struct ProfileReport: Codable, Sendable {
        let timestamp: Date
        let traceCount: Int
        let totalDuration: TimeInterval
        let slowestOperation: String
        let hottestMethod: String
        let categoryBreakdown: [String: TimeInterval]
        let traces: [TraceInfo]
        
        struct TraceInfo: Codable, Sendable {
            let name: String
            let category: String
            let duration: TimeInterval?
            
            init(from trace: PerformanceTrace) {
                self.name = trace.name
                self.category = String(describing: trace.category)
                self.duration = trace.duration
            }
        }
    }
    
    // MARK: - Additional Monitoring Methods
    
    func startMonitoring() {
        startRecording()
    }
    
    func stopMonitoring() {
        stopRecording()
    }
    
    func recordCacheHit(_ cacheName: String = "default") {
        begin("Cache Hit - \(cacheName)", category: .cache)
        end("Cache Hit - \(cacheName)")
    }
    
    func recordRenderTime(_ duration: TimeInterval, operation: String = "render") {
        let trace = PerformanceTrace(
            name: operation,
            category: .render,
            startTime: Date().addingTimeInterval(-duration),
            endTime: Date(),
            metadata: ["duration": duration]
        )
        traces.append(trace)
        
        // Limit trace count
        if traces.count > maxTraces {
            traces.removeFirst()
        }
    }
}

// MARK: - Performance Profiler View
struct PerformanceProfilerView: View {
    @ObservedObject private var profiler = PerformanceProfiler.shared
    @State private var selectedTab = 0
    @State private var selectedTrace: PerformanceProfiler.PerformanceTrace?
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            ProfilerHeader(profiler: profiler)
            
            Divider()
            
            // Tab selector
            Picker("View", selection: $selectedTab) {
                Text("Timeline").tag(0)
                Text("Methods").tag(1)
                Text("Categories").tag(2)
                Text("Flame Graph").tag(3)
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding()
            
            // Content
            switch selectedTab {
            case 0:
                ProfilerTimelineView(profiler: profiler, selectedTrace: $selectedTrace)
            case 1:
                MethodsView(profiler: profiler)
            case 2:
                CategoriesView(profiler: profiler)
            case 3:
                FlameGraphView(profiler: profiler)
            default:
                EmptyView()
            }
            
            // Detail panel
            if let trace = selectedTrace {
                TraceDetailPanel(trace: trace)
            }
        }
        .frame(width: 900, height: 600)
        .background(Color(NSColor.windowBackgroundColor))
    }
}

// MARK: - Profiler Header
struct ProfilerHeader: View {
    @ObservedObject var profiler: PerformanceProfiler
    
    public var body: some View {
        HStack {
            Label("Performance Profiler", systemImage: "flame")
                .font(.system(size: 14, weight: .semibold))
            
            Spacer()
            
            Text("\(profiler.traces.count) traces")
                .font(.system(size: 11))
                .foregroundColor(.secondary)
            
            Divider()
                .frame(height: 20)
            
            Button(action: {
                if profiler.isRecording {
                    profiler.stopRecording()
                } else {
                    profiler.startRecording()
                }
            }) {
                Image(systemName: profiler.isRecording ? "stop.circle.fill" : "record.circle")
                    .foregroundColor(profiler.isRecording ? .red : .primary)
            }
            
            Button(action: { profiler.exportTraces() }) {
                Image(systemName: "square.and.arrow.up")
            }
            .disabled(profiler.traces.isEmpty)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
    }
}

// MARK: - Timeline View
struct ProfilerTimelineView: View {
    @ObservedObject var profiler: PerformanceProfiler
    @Binding var selectedTrace: PerformanceProfiler.PerformanceTrace?
    
    public var body: some View {
        ScrollView {
            VStack(spacing: 2) {
                ForEach(profiler.traces.reversed()) { trace in
                    TimelineRow(trace: trace, isSelected: selectedTrace?.id == trace.id)
                        .onTapGesture {
                            selectedTrace = trace
                        }
                }
            }
            .padding()
        }
    }
}

struct TimelineRow: View {
    let trace: PerformanceProfiler.PerformanceTrace
    let isSelected: Bool
    
    private var durationText: String {
        guard let duration = trace.duration else { return "Running..." }
        if duration < 0.001 {
            return String(format: "%.3f μs", duration * 1_000_000)
        } else if duration < 1 {
            return String(format: "%.2f ms", duration * 1000)
        } else {
            return String(format: "%.2f s", duration)
        }
    }
    
    public var body: some View {
        HStack(spacing: 8) {
            Rectangle()
                .fill(trace.category.color)
                .frame(width: 4)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(trace.name)
                    .font(.system(size: 11, weight: .medium))
                    .lineLimit(1)
                
                Text(trace.startTime, style: .time)
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(durationText)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(trace.duration == nil ? .orange : .primary)
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        )
    }
}

// MARK: - Methods View
struct MethodsView: View {
    @ObservedObject var profiler: PerformanceProfiler
    
    var sortedMethods: [PerformanceProfiler.MethodTiming] {
        profiler.methodTimings.values.sorted { $0.totalTime > $1.totalTime }
    }
    
    public var body: some View {
        Table(sortedMethods) {
            TableColumn("Method") { timing in
                Text(timing.method)
                    .font(.system(size: 11))
            }
            .width(min: 200)
            
            TableColumn("Calls") { timing in
                Text("\(timing.callCount)")
                    .font(.system(size: 11, design: .monospaced))
            }
            .width(60)
            
            TableColumn("Total") { timing in
                Text(formatTime(timing.totalTime))
                    .font(.system(size: 11, design: .monospaced))
            }
            .width(80)
            
            TableColumn("Average") { timing in
                Text(formatTime(timing.averageTime))
                    .font(.system(size: 11, design: .monospaced))
            }
            .width(80)
            
            TableColumn("Min") { timing in
                Text(formatTime(timing.minTime))
                    .font(.system(size: 11, design: .monospaced))
            }
            .width(80)
            
            TableColumn("Max") { timing in
                Text(formatTime(timing.maxTime))
                    .font(.system(size: 11, design: .monospaced))
            }
            .width(80)
        }
    }
    
    private func formatTime(_ time: TimeInterval) -> String {
        if time < 0.001 {
            return String(format: "%.0fμs", time * 1_000_000)
        } else if time < 1 {
            return String(format: "%.1fms", time * 1000)
        } else {
            return String(format: "%.2fs", time)
        }
    }
}

// MARK: - Categories View
struct CategoriesView: View {
    @ObservedObject var profiler: PerformanceProfiler
    
    var categoryData: [(category: String, time: TimeInterval, percentage: Double)] {
        let breakdown = profiler.getCategoryBreakdown()
        let total = breakdown.values.reduce(0, +)
        
        return breakdown.map { category, time in
            let categoryName = String(describing: category)
            let percentage = total > 0 ? (time / total) * 100 : 0
            return (categoryName, time, percentage)
        }.sorted { $0.time > $1.time }
    }
    
    public var body: some View {
        VStack(spacing: 12) {
            ForEach(categoryData, id: \.category) { data in
                CategoryRow(
                    category: data.category,
                    time: data.time,
                    percentage: data.percentage
                )
            }
            
            Spacer()
        }
        .padding()
    }
}

struct CategoryRow: View {
    let category: String
    let time: TimeInterval
    let percentage: Double
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(category)
                    .font(.system(size: 12, weight: .medium))
                
                Spacer()
                
                Text(String(format: "%.1f%%", percentage))
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                
                Text(formatTime(time))
                    .font(.system(size: 11, design: .monospaced))
            }
            
            GeometryReader { geometry in
                Rectangle()
                    .fill(Color.accentColor)
                    .frame(width: geometry.size.width * percentage / 100, height: 20)
                    .overlay(
                        Rectangle()
                            .stroke(Color.accentColor.opacity(0.3), lineWidth: 1)
                            .frame(width: geometry.size.width, height: 20)
                    )
            }
            .frame(height: 20)
        }
    }
    
    private func formatTime(_ time: TimeInterval) -> String {
        if time < 1 {
            return String(format: "%.0fms", time * 1000)
        } else {
            return String(format: "%.2fs", time)
        }
    }
}

// MARK: - Flame Graph View
struct FlameGraphView: View {
    @ObservedObject var profiler: PerformanceProfiler
    
    public var body: some View {
        ScrollView([.horizontal, .vertical]) {
            // Simplified flame graph visualization
            VStack(spacing: 1) {
                ForEach(profiler.traces.prefix(50)) { trace in
                    if let duration = trace.duration {
                        FlameGraphBar(
                            trace: trace,
                            maxDuration: profiler.traces.compactMap { $0.duration }.max() ?? 1
                        )
                    }
                }
            }
            .padding()
        }
    }
}

struct FlameGraphBar: View {
    let trace: PerformanceProfiler.PerformanceTrace
    let maxDuration: TimeInterval
    
    private var barWidth: CGFloat {
        guard let duration = trace.duration else { return 0 }
        return CGFloat(duration / maxDuration) * 600
    }
    
    public var body: some View {
        HStack(spacing: 0) {
            Rectangle()
                .fill(trace.category.color)
                .frame(width: barWidth, height: 20)
                .overlay(
                    Text(trace.name)
                        .font(.system(size: 9))
                        .foregroundColor(.white)
                        .lineLimit(1)
                        .padding(.horizontal, 4),
                    alignment: .leading
                )
            
            Spacer()
        }
    }
}

// MARK: - Trace Detail Panel
struct TraceDetailPanel: View {
    let trace: PerformanceProfiler.PerformanceTrace
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Trace Details")
                .font(.system(size: 12, weight: .semibold))
            
            Divider()
            
            DetailField(label: "Name", value: trace.name)
            DetailField(label: "Category", value: String(describing: trace.category))
            DetailField(label: "Start", value: trace.startTime.formatted())
            
            if let endTime = trace.endTime {
                DetailField(label: "End", value: endTime.formatted())
            }
            
            if let duration = trace.duration {
                DetailField(label: "Duration", value: formatDuration(duration))
            }
            
            if !trace.metadata.isEmpty {
                Text("Metadata")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.top, 4)
                
                ForEach(Array(trace.metadata.keys), id: \.self) { key in
                    DetailField(label: key, value: String(describing: trace.metadata[key] ?? ""))
                }
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        if duration < 0.001 {
            return String(format: "%.3f microseconds", duration * 1_000_000)
        } else if duration < 1 {
            return String(format: "%.2f milliseconds", duration * 1000)
        } else {
            return String(format: "%.3f seconds", duration)
        }
    }
}

struct DetailField: View {
    let label: String
    let value: String
    
    public var body: some View {
        HStack(alignment: .top) {
            Text(label + ":")
                .font(.system(size: 10))
                .foregroundColor(.secondary)
                .frame(width: 80, alignment: .trailing)
            
            Text(value)
                .font(.system(size: 10))
                .textSelection(.enabled)
        }
    }
}
