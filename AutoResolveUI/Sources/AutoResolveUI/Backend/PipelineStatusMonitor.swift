import Foundation
import SwiftUI
import Combine

@MainActor
public class PipelineStatusMonitor: ObservableObject {
    @Published public var currentStatus: PipelineStatus = .idle
    @Published public var progressPercentage: Double = 0.0
    @Published public var currentOperation: String = ""
    @Published public var estimatedTimeRemaining: TimeInterval = 0
    @Published public var throughputMBps: Double = 0.0
    @Published public var memoryUsageMB: Double = 0.0
    @Published public var gpuUsagePercent: Double = 0.0
    @Published public var activeJobs: [ProcessingJob] = []
    @Published public var recentMessages: [StatusMessage] = []
    @Published public var performanceMetrics: SystemPerformanceMetrics = SystemPerformanceMetrics()
    
    private var webSocketTask: URLSessionWebSocketTask?
    private var statusUpdateTimer: Timer?
    private var cancellables = Set<AnyCancellable>()
    
    public enum PipelineStatus: String, CaseIterable, Codable {
        case idle = "idle"
        case initializing = "initializing"
        case loadingModels = "loading_models"
        case analyzing = "analyzing"
        case detectingSilence = "detecting_silence"
        case selectingBroll = "selecting_broll"
        case generating = "generating"
        case rendering = "rendering"
        case finalizing = "finalizing"
        case completed = "completed"
        case error = "error"
        
        var displayName: String {
            switch self {
            case .idle: return "Idle"
            case .initializing: return "Initializing"
            case .loadingModels: return "Loading AI Models"
            case .analyzing: return "Analyzing Video"
            case .detectingSilence: return "Detecting Silence"
            case .selectingBroll: return "Selecting B-roll"
            case .generating: return "Generating Timeline"
            case .rendering: return "Rendering Output"
            case .finalizing: return "Finalizing"
            case .completed: return "Completed"
            case .error: return "Error"
            }
        }
        
        var color: Color {
            switch self {
            case .idle: return .secondary
            case .initializing, .loadingModels: return .orange
            case .analyzing, .detectingSilence, .selectingBroll: return .blue
            case .generating, .rendering: return .green
            case .finalizing: return .purple
            case .completed: return .green
            case .error: return .red
            }
        }
    }
    
    public struct ProcessingJob: Identifiable, Codable {
        public let id: UUID
        public let jobType: String
        public let inputPath: String
        public let outputPath: String
        public let startTime: Date
        public var endTime: Date?
        public var status: PipelineStatus
        public var progress: Double
        public var errorMessage: String?
        
        public init(jobType: String, inputPath: String, outputPath: String) {
            self.id = UUID()
            self.jobType = jobType
            self.inputPath = inputPath
            self.outputPath = outputPath
            self.startTime = Date()
            self.status = .initializing
            self.progress = 0.0
        }
        
        public var duration: TimeInterval {
            (endTime ?? Date()).timeIntervalSince(startTime)
        }
    }
    
    public struct StatusMessage: Identifiable, Codable {
        public let id: UUID
        public let timestamp: Date
        public let level: LogLevel
        public let message: String
        public let component: String
        
        public enum LogLevel: String, Codable, CaseIterable {
            case debug, info, warning, error
            
            var color: Color {
                switch self {
                case .debug: return .secondary
                case .info: return .primary
                case .warning: return .orange
                case .error: return .red
                }
            }
        }
        
        public init(level: LogLevel, message: String, component: String = "Pipeline") {
            self.id = UUID()
            self.timestamp = Date()
            self.level = level
            self.message = message
            self.component = component
        }
    }
    
    public struct SystemPerformanceMetrics: Codable {
        public var cpuUsagePercent: Double = 0.0
        public var memoryUsedMB: Double = 0.0
        public var memoryAvailableMB: Double = 0.0
        public var diskReadMBps: Double = 0.0
        public var diskWriteMBps: Double = 0.0
        public var networkMBps: Double = 0.0
        public var framesProcessedPerSecond: Double = 0.0
        public var averageProcessingTimeMs: Double = 0.0
        
        public var memoryUsagePercent: Double {
            guard memoryAvailableMB > 0 else { return 0.0 }
            return (memoryUsedMB / (memoryUsedMB + memoryAvailableMB)) * 100.0
        }
    }
    
    public init() {
        startStatusMonitoring()
    }
    
    deinit {
        Task { @MainActor in
            stopStatusMonitoring()
        }
    }
    
    // MARK: - WebSocket Connection
    
    public func connectToStatusStream() {
        guard let url = URL(string: "ws://localhost:8000/ws/status") else {
            addMessage(.error, "Invalid WebSocket URL")
            return
        }
        
        let request = URLRequest(url: url)
        webSocketTask = URLSession.shared.webSocketTask(with: request)
        webSocketTask?.resume()
        
        receiveMessage()
        
        addMessage(.info, "Connected to status stream")
    }
    
    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let message):
                    switch message {
                    case .string(let text):
                        self?.processStatusMessage(text)
                    case .data(let data):
                        self?.processStatusData(data)
                    @unknown default:
                        break
                    }
                    self?.receiveMessage()
                    
                case .failure(let error):
                    self?.addMessage(.error, "WebSocket error: \(error.localizedDescription)")
                    self?.reconnectAfterDelay()
                }
            }
        }
    }
    
    private func reconnectAfterDelay() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
            self?.connectToStatusStream()
        }
    }
    
    // MARK: - Status Processing
    
    private func processStatusMessage(_ message: String) {
        guard let data = message.data(using: .utf8),
              let statusUpdate = try? JSONDecoder().decode(StatusUpdate.self, from: data) else {
            addMessage(.warning, "Failed to decode status message")
            return
        }
        
        updateStatus(from: statusUpdate)
    }
    
    private func processStatusData(_ data: Data) {
        guard let statusUpdate = try? JSONDecoder().decode(StatusUpdate.self, from: data) else {
            addMessage(.warning, "Failed to decode status data")
            return
        }
        
        updateStatus(from: statusUpdate)
    }
    
    private func updateStatus(from update: StatusUpdate) {
        currentStatus = PipelineStatus(rawValue: update.status) ?? .idle
        progressPercentage = update.progress ?? progressPercentage
        currentOperation = update.operation ?? currentOperation
        estimatedTimeRemaining = update.eta ?? estimatedTimeRemaining
        
        if let metrics = update.metrics {
            performanceMetrics = metrics
            throughputMBps = metrics.diskReadMBps + metrics.diskWriteMBps
            memoryUsageMB = metrics.memoryUsedMB
            gpuUsagePercent = metrics.cpuUsagePercent // Using CPU as GPU proxy
        }
        
        if let jobUpdate = update.jobUpdate {
            updateJob(jobUpdate)
        }
        
        if let message = update.message {
            addMessage(.info, message, component: update.component ?? "Pipeline")
        }
        
        // Auto-cleanup old messages (keep last 100)
        if recentMessages.count > 100 {
            recentMessages.removeFirst(recentMessages.count - 100)
        }
    }
    
    private func updateJob(_ jobUpdate: JobUpdate) {
        if let existingIndex = activeJobs.firstIndex(where: { $0.id.uuidString == jobUpdate.jobId }) {
            activeJobs[existingIndex].status = PipelineStatus(rawValue: jobUpdate.status) ?? .idle
            activeJobs[existingIndex].progress = jobUpdate.progress
            activeJobs[existingIndex].errorMessage = jobUpdate.error
            
            if jobUpdate.status == "completed" || jobUpdate.status == "error" {
                activeJobs[existingIndex].endTime = Date()
            }
        }
    }
    
    // MARK: - Status Updates
    
    public func addMessage(_ level: StatusMessage.LogLevel, _ message: String, component: String = "Pipeline") {
        let statusMessage = StatusMessage(level: level, message: message, component: component)
        recentMessages.append(statusMessage)
    }
    
    public func startJob(_ jobType: String, inputPath: String, outputPath: String) -> UUID {
        let job = ProcessingJob(jobType: jobType, inputPath: inputPath, outputPath: outputPath)
        activeJobs.append(job)
        addMessage(.info, "Started \(jobType) job", component: "JobManager")
        return job.id
    }
    
    public func completeJob(_ jobId: UUID, success: Bool = true, error: String? = nil) {
        guard let index = activeJobs.firstIndex(where: { $0.id == jobId }) else { return }
        
        activeJobs[index].endTime = Date()
        activeJobs[index].status = success ? .completed : .error
        activeJobs[index].errorMessage = error
        activeJobs[index].progress = 1.0
        
        let jobType = activeJobs[index].jobType
        let duration = activeJobs[index].duration
        
        if success {
            addMessage(.info, "Completed \(jobType) in \(String(format: "%.1f", duration))s", component: "JobManager")
        } else {
            addMessage(.error, "Failed \(jobType): \(error ?? "Unknown error")", component: "JobManager")
        }
    }
    
    public func clearCompletedJobs() {
        activeJobs.removeAll { $0.status == .completed || $0.status == .error }
    }
    
    public func resetStatus() {
        currentStatus = .idle
        progressPercentage = 0.0
        currentOperation = ""
        estimatedTimeRemaining = 0
        throughputMBps = 0.0
        memoryUsageMB = 0.0
        gpuUsagePercent = 0.0
        activeJobs.removeAll()
        performanceMetrics = SystemPerformanceMetrics()
        
        addMessage(.info, "Status reset")
    }
    
    // MARK: - Monitoring
    
    private func startStatusMonitoring() {
        connectToStatusStream()
        
        statusUpdateTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateSystemMetrics()
        }
    }
    
    private func stopStatusMonitoring() {
        webSocketTask?.cancel()
        webSocketTask = nil
        statusUpdateTimer?.invalidate()
        statusUpdateTimer = nil
    }
    
    private func updateSystemMetrics() {
        // Update system metrics (memory, CPU, etc.)
        let processInfo = ProcessInfo.processInfo
        let memoryUsage = processInfo.physicalMemory
        
        performanceMetrics.memoryUsedMB = Double(memoryUsage) / (1024 * 1024)
        
        // Update active job progress if needed
        for i in 0..<activeJobs.count {
            if activeJobs[i].status != .completed && activeJobs[i].status != .error {
                // Simulate progress for active jobs
                let elapsed = Date().timeIntervalSince(activeJobs[i].startTime)
                let estimatedDuration: TimeInterval = 60.0 // 1 minute estimate
                activeJobs[i].progress = min(0.95, elapsed / estimatedDuration)
            }
        }
    }
    
    // MARK: - Computed Properties
    
    public var isProcessing: Bool {
        switch currentStatus {
        case .idle, .completed, .error:
            return false
        default:
            return true
        }
    }
    
    public var activeJobCount: Int {
        activeJobs.filter { $0.status != .completed && $0.status != .error }.count
    }
    
    public var totalJobsCompleted: Int {
        activeJobs.filter { $0.status == .completed }.count
    }
    
    public var hasErrors: Bool {
        activeJobs.contains { $0.status == .error } || currentStatus == .error
    }
    
    public var errorMessages: [String] {
        activeJobs.compactMap { $0.errorMessage } + 
        recentMessages.filter { $0.level == .error }.map { $0.message }
    }
}

// MARK: - Supporting Data Structures

private struct StatusUpdate: Codable {
    let status: String
    let progress: Double?
    let operation: String?
    let eta: TimeInterval?
    let metrics: PipelineStatusMonitor.SystemPerformanceMetrics?
    let jobUpdate: JobUpdate?
    let message: String?
    let component: String?
}

private struct JobUpdate: Codable {
    let jobId: String
    let status: String
    let progress: Double
    let error: String?
}