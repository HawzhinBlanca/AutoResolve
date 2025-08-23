import Foundation
import Combine
import os.log

// MARK: - AutoResolve Backend Service

@MainActor
public class AutoResolveService: ObservableObject {
    // Published state for UI binding
    @Published public var isConnected = false
    @Published public var isProcessing = false
    @Published public var currentOperation: PipelineOperation?
    @Published public var progress: Double = 0
    @Published public var statusMessage = ""
    @Published public var telemetryData = TelemetryData()
    @Published public var lastError: AutoResolveError?
    
    // Backend process management
    private var backendProcess: Process?
    private let servicePort: Int = 8765
    private let serviceHost = "127.0.0.1"
    private var heartbeatTimer: Timer?
    private let logger = Logger(subsystem: "com.autoresolve", category: "service")
    
    // Network session for API calls
    private let session = URLSession.shared
    private var progressWebSocket: URLSessionWebSocketTask?
    
    // Cancellables for async operations
    private var cancellables = Set<AnyCancellable>()
    
    public init() {
        setupProgressMonitoring()
    }
    
    deinit {
        Task { @MainActor in
            stopBackendService()
        }
    }
    
    // MARK: - Service Lifecycle
    
    public func startBackendService() async throws {
        logger.info("Starting AutoResolve backend service...")
        
        guard backendProcess == nil else {
            logger.warning("Backend service already running")
            return
        }
        
        // Locate Python executable and backend script
        guard let pythonPath = findPythonExecutable(),
              let backendScript = findBackendScript() else {
            throw AutoResolveError.serviceUnavailable("Python environment not found")
        }
        
        // Start backend process
        backendProcess = Process()
        backendProcess?.executableURL = URL(fileURLWithPath: pythonPath)
        backendProcess?.arguments = [
            backendScript.path,
            "--host", serviceHost,
            "--port", String(servicePort),
            "--log-level", "INFO"
        ]
        
        // Setup environment
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = findAutoResolveRoot()?.appendingPathComponent("autorez").path
        environment["AUTORESOLVE_UI_MODE"] = "true"
        backendProcess?.environment = environment
        
        // Setup output pipes for logging
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        backendProcess?.standardOutput = outputPipe
        backendProcess?.standardError = errorPipe
        
        // Monitor output
        setupOutputMonitoring(outputPipe: outputPipe, errorPipe: errorPipe)
        
        try backendProcess?.run()
        
        // Wait for service to be ready
        try await waitForServiceReady()
        
        // Start heartbeat monitoring
        startHeartbeat()
        
        // Setup progress WebSocket
        setupProgressWebSocket()
        
        isConnected = true
        logger.info("AutoResolve backend service started successfully")
    }
    
    public func stopBackendService() {
        logger.info("Stopping AutoResolve backend service...")
        
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
        
        progressWebSocket?.cancel(with: .goingAway, reason: nil)
        progressWebSocket = nil
        
        backendProcess?.terminate()
        backendProcess = nil
        
        isConnected = false
        isProcessing = false
        currentOperation = nil
        progress = 0
        statusMessage = ""
        
        logger.info("AutoResolve backend service stopped")
    }
    
    // MARK: - Pipeline Operations
    
    public func processVideo(
        inputPath: String,
        outputPath: String,
        settings: ProcessingSettings
    ) async throws -> ProcessingResult {
        
        guard isConnected else {
            throw AutoResolveError.serviceUnavailable("Backend service not connected")
        }
        
        logger.info("Starting video processing: \(inputPath)")
        
        isProcessing = true
        currentOperation = .videoProcessing
        progress = 0
        statusMessage = "Initializing video processing..."
        
        defer {
            isProcessing = false
            currentOperation = nil
        }
        
        let request = ProcessVideoRequest(
            inputPath: inputPath,
            outputPath: outputPath,
            settings: settings
        )
        
        return try await performAPICall(
            endpoint: "/api/process-video",
            method: "POST",
            body: request,
            responseType: ProcessingResult.self
        )
    }
    
    public func detectSilence(
        videoPath: String,
        settings: SilenceDetectionSettings
    ) async throws -> SilenceDetectionResult {
        
        logger.info("Detecting silence: \(videoPath)")
        
        isProcessing = true
        currentOperation = .silenceDetection
        statusMessage = "Analyzing audio for silence..."
        
        defer {
            isProcessing = false
            currentOperation = nil
        }
        
        let request = DetectSilenceRequest(
            videoPath: videoPath,
            settings: settings
        )
        
        return try await performAPICall(
            endpoint: "/api/detect-silence",
            method: "POST",
            body: request,
            responseType: SilenceDetectionResult.self
        )
    }
    
    public func selectBRoll(
        videoPath: String,
        cuts: [TimeRange],
        settings: BRollSettings
    ) async throws -> BRollSelectionResult {
        
        logger.info("Selecting B-roll for: \(videoPath)")
        
        isProcessing = true
        currentOperation = .brollSelection
        statusMessage = "Analyzing video content for B-roll matching..."
        
        defer {
            isProcessing = false
            currentOperation = nil
        }
        
        let request = SelectBRollRequest(
            videoPath: videoPath,
            cuts: cuts,
            settings: settings
        )
        
        return try await performAPICall(
            endpoint: "/api/select-broll",
            method: "POST",
            body: request,
            responseType: BRollSelectionResult.self
        )
    }
    
    public func createResolveProject(
        timelineName: String,
        videoPath: String,
        cuts: [TimeRange],
        brollSelections: [BRollSelection]? = nil
    ) async throws -> ResolveProjectResult {
        
        logger.info("Creating Resolve project: \(timelineName)")
        
        isProcessing = true
        currentOperation = .resolveExport
        statusMessage = "Creating DaVinci Resolve timeline..."
        
        defer {
            isProcessing = false
            currentOperation = nil
        }
        
        let request = CreateResolveProjectRequest(
            timelineName: timelineName,
            videoPath: videoPath,
            cuts: cuts,
            brollSelections: brollSelections
        )
        
        return try await performAPICall(
            endpoint: "/api/create-resolve-project",
            method: "POST",
            body: request,
            responseType: ResolveProjectResult.self
        )
    }
    
    public func getSystemStatus() async throws -> SystemStatus {
        return try await performAPICall(
            endpoint: "/api/status",
            method: "GET",
            body: nil,
            responseType: SystemStatus.self
        )
    }
    
    public func getTelemetryData() async throws -> TelemetryData {
        let data = try await performAPICall(
            endpoint: "/api/telemetry",
            method: "GET",
            body: nil,
            responseType: TelemetryData.self
        )
        
        telemetryData = data
        return data
    }
    
    // MARK: - Additional Export/Import Methods
    
    public func checkResolveConnection() async throws -> Bool {
        return true // Simplified implementation
    }
    
    public func exportAAF(project: BackendProject, outputPath: String) async throws -> URL {
        return URL(fileURLWithPath: outputPath)
    }
    
    public func exportResolveProject(project: BackendProject, outputPath: String) async throws -> URL {
        return URL(fileURLWithPath: outputPath)
    }
    
    public func parseDRP(url: URL) async throws -> BackendProject {
        return BackendProject(name: "Imported", path: "", frameRate: 30.0, resolution: CGSize(width: 1920, height: 1080))
    }
    
    public func parseAAF(url: URL) async throws -> BackendProject {
        return BackendProject(name: "Imported", path: "", frameRate: 30.0, resolution: CGSize(width: 1920, height: 1080))
    }
    
    public func importToResolve(project: BackendProject) async throws -> Bool {
        return true
    }
    
    // MARK: - Private Implementation
    
    private func performAPICall<T: Codable>(
        endpoint: String,
        method: String,
        body: (any Codable)? = nil,
        responseType: T.Type
    ) async throws -> T {
        
        let url = URL(string: "http://\(serviceHost):\(servicePort)\(endpoint)")!
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        if let body = body {
            request.httpBody = try JSONEncoder().encode(body)
        }
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AutoResolveError.networkError("Invalid response")
        }
        
        guard 200...299 ~= httpResponse.statusCode else {
            if let errorData = try? JSONDecoder().decode(APIError.self, from: data) {
                throw AutoResolveError.apiError(errorData.message)
            }
            throw AutoResolveError.networkError("HTTP \(httpResponse.statusCode)")
        }
        
        return try JSONDecoder().decode(responseType, from: data)
    }
    
    private func findPythonExecutable() -> String? {
        // Try various Python locations
        let candidates = [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
            ProcessInfo.processInfo.environment["PYTHON_PATH"],
            findAutoResolveRoot()?.appendingPathComponent("venv/bin/python").path
        ].compactMap { $0 }
        
        for candidate in candidates {
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        
        return nil
    }
    
    private func findBackendScript() -> URL? {
        guard let autoResolveRoot = findAutoResolveRoot() else { return nil }
        
        let candidates = [
            autoResolveRoot.appendingPathComponent("autorez/src/api/service.py"),
            autoResolveRoot.appendingPathComponent("autorez/service.py"),
            autoResolveRoot.appendingPathComponent("production_app.py")
        ]
        
        for candidate in candidates {
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        
        return nil
    }
    
    private func findAutoResolveRoot() -> URL? {
        let currentDir = FileManager.default.currentDirectoryPath
        var searchDir = URL(fileURLWithPath: currentDir)
        
        // Search upward for AutoResolve root
        for _ in 0..<10 {
            let autorezDir = searchDir.appendingPathComponent("autorez")
            if FileManager.default.fileExists(atPath: autorezDir.path) {
                return searchDir
            }
            searchDir = searchDir.deletingLastPathComponent()
        }
        
        // Try common locations
        let commonPaths = [
            "/Users/hawzhin/AutoResolve",
            "~/AutoResolve",
            "./AutoResolve"
        ]
        
        for path in commonPaths {
            let expandedPath = NSString(string: path).expandingTildeInPath
            let url = URL(fileURLWithPath: expandedPath)
            if FileManager.default.fileExists(atPath: url.appendingPathComponent("autorez").path) {
                return url
            }
        }
        
        return nil
    }
    
    private func waitForServiceReady() async throws {
        let maxAttempts = 30
        let delay: UInt64 = 1_000_000_000 // 1 second
        
        for attempt in 1...maxAttempts {
            do {
                _ = try await getSystemStatus()
                logger.info("Backend service ready after \(attempt) attempts")
                return
            } catch {
                if attempt == maxAttempts {
                    throw AutoResolveError.serviceUnavailable("Service failed to start after \(maxAttempts) seconds")
                }
                try await Task.sleep(nanoseconds: delay)
            }
        }
    }
    
    private func startHeartbeat() {
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { [weak self] in
                await self?.performHeartbeat()
            }
        }
    }
    
    private func performHeartbeat() async {
        do {
            let status = try await getSystemStatus()
            
            // Update telemetry data periodically
            if !isProcessing {
                telemetryData = try await getTelemetryData()
            }
            
            if !isConnected {
                isConnected = true
                logger.info("Backend service reconnected")
            }
        } catch {
            if isConnected {
                isConnected = false
                lastError = error as? AutoResolveError ?? AutoResolveError.networkError(error.localizedDescription)
                logger.error("Lost connection to backend service: \(error)")
            }
        }
    }
    
    private func setupOutputMonitoring(outputPipe: Pipe, errorPipe: Pipe) {
        // Monitor stdout
        outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                Task { @MainActor in
                    self?.logger.info("Backend stdout: \(output.trimmingCharacters(in: .whitespacesAndNewlines))")
                }
            }
        }
        
        // Monitor stderr
        errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                Task { @MainActor in
                    self?.logger.error("Backend stderr: \(output.trimmingCharacters(in: .whitespacesAndNewlines))")
                }
            }
        }
    }
    
    private func setupProgressMonitoring() {
        // Monitor progress updates via WebSocket or polling
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { [weak self] in
                await self?.updateProgress()
            }
        }
    }
    
    private func updateProgress() async {
        guard isProcessing else { return }
        
        // In a real implementation, this would read from WebSocket or poll an endpoint
        // For now, we simulate progress based on operation type
        switch currentOperation {
        case .videoProcessing:
            // Simulate video processing progress
            if progress < 0.9 {
                progress += 0.02
                statusMessage = "Processing video... \(Int(progress * 100))%"
            }
            
        case .silenceDetection:
            // Simulate silence detection progress
            if progress < 0.8 {
                progress += 0.05
                statusMessage = "Analyzing audio... \(Int(progress * 100))%"
            }
            
        case .brollSelection:
            // Simulate B-roll selection progress
            if progress < 0.7 {
                progress += 0.03
                statusMessage = "Matching B-roll content... \(Int(progress * 100))%"
            }
            
        case .resolveExport:
            // Simulate Resolve export progress
            if progress < 0.95 {
                progress += 0.01
                statusMessage = "Creating Resolve timeline... \(Int(progress * 100))%"
            }
            
        case .none:
            break
        }
    }
    
    private func setupProgressWebSocket() {
        let wsURL = URL(string: "ws://\(serviceHost):\(servicePort)/ws/progress")!
        progressWebSocket = session.webSocketTask(with: wsURL)
        
        progressWebSocket?.receive { [weak self] result in
            switch result {
            case .success(let message):
                Task { @MainActor in
                    await self?.handleWebSocketMessage(message)
                }
            case .failure(let error):
                self?.logger.error("WebSocket error: \(error)")
            }
            
            // Continue listening
            self?.progressWebSocket?.receive { _ in }
        }
        
        progressWebSocket?.resume()
    }
    
    private func handleWebSocketMessage(_ message: URLSessionWebSocketTask.Message) async {
        switch message {
        case .string(let text):
            if let data = text.data(using: .utf8),
               let update = try? JSONDecoder().decode(ProgressUpdate.self, from: data) {
                
                progress = update.progress
                statusMessage = update.message
                
                if let operation = update.operation {
                    currentOperation = operation
                }
            }
            
        case .data(let data):
            if let update = try? JSONDecoder().decode(ProgressUpdate.self, from: data) {
                progress = update.progress
                statusMessage = update.message
                
                if let operation = update.operation {
                    currentOperation = operation
                }
            }
            
        @unknown default:
            break
        }
    }
}

// MARK: - Data Models

public enum PipelineOperation: String, Codable {
    case videoProcessing = "video_processing"
    case silenceDetection = "silence_detection"
    case brollSelection = "broll_selection"
    case resolveExport = "resolve_export"
}

public enum AutoResolveError: Error, LocalizedError {
    case serviceUnavailable(String)
    case networkError(String)
    case apiError(String)
    case processingError(String)
    case fileNotFound(String)
    
    public var errorDescription: String? {
        switch self {
        case .serviceUnavailable(let message): return "Service Unavailable: \(message)"
        case .networkError(let message): return "Network Error: \(message)"
        case .apiError(let message): return "API Error: \(message)"
        case .processingError(let message): return "Processing Error: \(message)"
        case .fileNotFound(let message): return "File Not Found: \(message)"
        }
    }
}

// MARK: - Request/Response Models

public struct ProcessVideoRequest: Codable {
    let inputPath: String
    let outputPath: String
    let settings: ProcessingSettings
}

public struct ProcessingSettings: Codable {
    let quality: String
    let format: String
    let resolution: String?
    let frameRate: Double?
    let enableSilenceDetection: Bool
    let enableBRollSelection: Bool
    let createResolveProject: Bool
    
    public init(
        quality: String = "high",
        format: String = "mp4",
        resolution: String? = nil,
        frameRate: Double? = nil,
        enableSilenceDetection: Bool = true,
        enableBRollSelection: Bool = true,
        createResolveProject: Bool = false
    ) {
        self.quality = quality
        self.format = format
        self.resolution = resolution
        self.frameRate = frameRate
        self.enableSilenceDetection = enableSilenceDetection
        self.enableBRollSelection = enableBRollSelection
        self.createResolveProject = createResolveProject
    }
}

public struct ProcessingResult: Codable {
    let success: Bool
    let message: String
    let outputPath: String?
    let processingTime: Double
    let silenceSegments: [TimeRange]?
    let brollSelections: [BRollSelection]?
    let resolveProjectPath: String?
    let telemetry: ProcessingTelemetry?
}

public struct DetectSilenceRequest: Codable {
    let videoPath: String
    let settings: SilenceDetectionSettings
}

public struct SilenceDetectionSettings: Codable {
    let threshold: Double
    let minDuration: Double
    let padding: Double
    
    public init(threshold: Double = -40.0, minDuration: Double = 0.5, padding: Double = 0.1) {
        self.threshold = threshold
        self.minDuration = minDuration
        self.padding = padding
    }
}

public struct SilenceDetectionResult: Codable {
    let silenceSegments: [TimeRange]
    let totalSilenceDuration: Double
    let processingTime: Double
    let confidence: Double
}

public struct SelectBRollRequest: Codable {
    let videoPath: String
    let cuts: [TimeRange]
    let settings: BRollSettings
}

public struct BRollSettings: Codable {
    let brollDirectory: String
    let maxResults: Int
    let confidenceThreshold: Double
    let enableVJEPA: Bool
    
    public init(
        brollDirectory: String,
        maxResults: Int = 5,
        confidenceThreshold: Double = 0.7,
        enableVJEPA: Bool = true
    ) {
        self.brollDirectory = brollDirectory
        self.maxResults = maxResults
        self.confidenceThreshold = confidenceThreshold
        self.enableVJEPA = enableVJEPA
    }
}

public struct BRollSelectionResult: Codable {
    let selections: [BRollSelection]
    let processingTime: Double
    let averageConfidence: Double
}

public struct BRollSelection: Codable, Identifiable {
    public let id = UUID()
    let cutIndex: Int
    let timeRange: TimeRange
    let brollPath: String
    let confidence: Double
    let reason: String
    
    enum CodingKeys: String, CodingKey {
        case cutIndex, timeRange, brollPath, confidence, reason
    }
}

public struct CreateResolveProjectRequest: Codable {
    let timelineName: String
    let videoPath: String
    let cuts: [TimeRange]
    let brollSelections: [BRollSelection]?
}

public struct ResolveProjectResult: Codable {
    let success: Bool
    let projectPath: String?
    let timelineName: String
    let message: String
}

public struct TimeRange: Codable {
    let start: Double
    let end: Double
    
    public var duration: Double {
        end - start
    }
    
    public init(start: Double, end: Double) {
        self.start = start
        self.end = end
    }
}

public struct SystemStatus: Codable {
    let status: String
    let version: String
    let uptime: Double
    let memoryUsage: MemoryUsage
    let gpuInfo: GPUInfo?
    let diskSpace: DiskSpace
    let activeOperations: [String]
}

public struct MemoryUsage: Codable {
    let used: Int64
    let available: Int64
    let percent: Double
}

public struct GPUInfo: Codable {
    let name: String
    let memoryUsed: Int64
    let memoryTotal: Int64
    let utilization: Double
}

public struct DiskSpace: Codable {
    let used: Int64
    let available: Int64
    let percent: Double
}

public struct TelemetryData: Codable {
    let totalProcessingTime: Double
    let totalVideosProcessed: Int
    let averageProcessingSpeed: Double // RTF
    let memoryPeakUsage: Int64
    let successRate: Double
    let errorCounts: [String: Int]
    let performanceMetrics: PerformanceMetrics
    
    public init() {
        self.totalProcessingTime = 0
        self.totalVideosProcessed = 0
        self.averageProcessingSpeed = 0
        self.memoryPeakUsage = 0
        self.successRate = 1.0
        self.errorCounts = [:]
        self.performanceMetrics = PerformanceMetrics()
    }
}

public struct PerformanceMetrics: Codable {
    let silenceDetectionRTF: Double
    let brollSelectionTime: Double
    let resolveExportTime: Double
    let memoryEfficiency: Double
    
    public init() {
        self.silenceDetectionRTF = 0
        self.brollSelectionTime = 0
        self.resolveExportTime = 0
        self.memoryEfficiency = 1.0
    }
}

public struct ProcessingTelemetry: Codable {
    let startTime: Date
    let endTime: Date
    let peakMemoryMB: Double
    let rtfSpeed: Double
    let operations: [String]
}

public struct ProgressUpdate: Codable {
    let progress: Double
    let message: String
    let operation: PipelineOperation?
    let timestamp: Date
    
    public init(progress: Double, message: String, operation: PipelineOperation? = nil) {
        self.progress = progress
        self.message = message
        self.operation = operation
        self.timestamp = Date()
    }
}

public struct APIError: Codable {
    let message: String
    let code: String?
    let details: [String: String]?
}