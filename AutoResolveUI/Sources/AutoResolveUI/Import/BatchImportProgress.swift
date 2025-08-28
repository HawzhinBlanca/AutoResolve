// AUTORESOLVE V3.0 - BATCH IMPORT PROGRESS
// Professional batch import with detailed progress tracking

import SwiftUI
import AppKit
import Combine

// MARK: - Batch Import Progress View
struct BatchImportProgressView: View {
    @ObservedObject var batchManager: BatchImportManager
    @Binding var isPresented: Bool
    @State private var showDetails = false
    @State private var selectedOperation: ImportOperation?
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            BatchImportHeader(
                batchManager: batchManager,
                onCancel: { batchManager.cancel() }
            )
            
            Divider()
            
            // Main progress area
            VStack(spacing: 20) {
                // Overall progress
                OverallProgressView(batchManager: batchManager)
                
                // Current operation
                CurrentOperationView(batchManager: batchManager)
                
                // Operation queue
                OperationQueueView(
                    batchManager: batchManager,
                    selectedOperation: $selectedOperation,
                    showDetails: $showDetails
                )
                
                // Performance metrics
                if showDetails {
                    PerformanceMetricsView(batchManager: batchManager)
                }
            }
            .padding()
            
            Divider()
            
            // Footer
            BatchImportFooter(
                batchManager: batchManager,
                showDetails: $showDetails,
                onClose: {
                    if !batchManager.isProcessing {
                        isPresented = false
                    }
                }
            )
        }
        .frame(width: 800, height: showDetails ? 600 : 450)
        .background(DaVinciColors.panelBackground)
    }
}

// MARK: - Batch Import Manager
@MainActor
class BatchImportManager: ObservableObject {
    @Published var operations: [ImportOperation] = []
    @Published var currentOperation: ImportOperation?
    @Published var isProcessing = false
    @Published var isPaused = false
    @Published var overallProgress: Double = 0
    @Published var currentProgress: Double = 0
    @Published var estimatedTimeRemaining: TimeInterval = 0
    @Published var processedSize: Int64 = 0
    @Published var totalSize: Int64 = 0
    @Published var throughput: Double = 0 // MB/s
    @Published var errors: [ImportError] = []
    
    private var cancellables = Set<AnyCancellable>()
    private var startTime: Date?
    private var operationQueue = OperationQueue()
    
    init() {
        operationQueue.maxConcurrentOperationCount = 4
        operationQueue.qualityOfService = .userInitiated
    }
    
    // MARK: - Queue Management
    func addOperation(_ operation: ImportOperation) {
        operations.append(operation)
        totalSize += operation.fileSize
        
        if !isProcessing {
            startProcessing()
        }
    }
    
    func addBatch(_ files: [URL], settings: ImportSettings) {
        let newOperations = files.map { url in
            let size64: Int64 = {
                if let rv = try? url.resourceValues(forKeys: [.fileSizeKey]) {
                    return Int64(rv.fileSize ?? 0)
                }
                return 0
            }()
            return ImportOperation(
                id: UUID(),
                file: url,
                type: determineOperationType(for: url, settings: settings),
                fileSize: size64,
                settings: settings
            )
        }
        
        operations.append(contentsOf: newOperations)
        totalSize = operations.reduce(Int64(0)) { $0 + $1.fileSize }
        
        if !isProcessing {
            startProcessing()
        }
    }
    
    private func determineOperationType(for url: URL, settings: ImportSettings) -> ImportOperation.OperationType {
        if settings.createProxies {
            return .proxy
        } else if settings.createOptimized {
            return .optimize
        } else if settings.copyToProject {
            return .copy
        } else {
            return .import
        }
    }
    
    // MARK: - Processing
    func startProcessing() {
        guard !isProcessing else { return }
        
        isProcessing = true
        isPaused = false
        startTime = Date()
        
        Task {
            for operation in operations where operation.status == .pending {
                if isPaused {
                    await waitForResume()
                }
                
                if !isProcessing { break }
                
                await processOperation(operation)
            }
            
            isProcessing = false
            calculateFinalMetrics()
        }
    }
    
    private func processOperation(_ operation: ImportOperation) async {
        currentOperation = operation
        operation.status = .processing
        operation.startTime = Date()
        
        do {
            switch operation.type {
            case .import:
                await importFile(operation)
            case .copy:
                try await copyFile(operation)
            case .optimize:
                await optimizeFile(operation)
            case .proxy:
                await createProxy(operation)
            case .transcode:
                await transcodeFile(operation)
            }
            
            operation.status = .completed
            operation.endTime = Date()
            processedSize += operation.fileSize
            
        } catch {
            operation.status = .failed
            operation.error = error
            errors.append(ImportError(operation: operation, error: error))
        }
        
        updateProgress()
        updateThroughput()
    }
    
    // MARK: - Operation Implementations
    private func importFile(_ operation: ImportOperation) async {
        // Simulate import with progress updates
        for i in 0...100 {
            if !isProcessing { break }
            
            operation.progress = Double(i) / 100.0
            currentProgress = operation.progress
            
            // Simulate work
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }
    }
    
    private func copyFile(_ operation: ImportOperation) async throws {
        guard let projectFolder = getProjectFolder() else { return }
        
        let destination = projectFolder.appendingPathComponent(operation.file.lastPathComponent)
        
        do {
            // Copy with progress monitoring
            let fileManager = FileManager.default
            
            if fileManager.fileExists(atPath: destination.path) {
                if operation.settings.skipExisting {
                    operation.status = .skipped
                    return
                } else {
                    try fileManager.removeItem(at: destination)
                }
            }
            
            // Use NSFileCoordinator for safe copying
            var error: NSError?
            let coordinator = NSFileCoordinator(filePresenter: nil)
            
            coordinator.coordinate(writingItemAt: operation.file,
                                 options: .forReplacing,
                                 error: &error) { (url) in
                do {
                    try fileManager.copyItem(at: url, to: destination)
                } catch {
                    operation.error = error
                }
            }
            
            if let error = error {
                throw error
            }
            
            operation.progress = 1.0
            currentProgress = 1.0
            
        } catch { throw error }
    }
    
    private func optimizeFile(_ operation: ImportOperation) async {
        // Create optimized media (ProRes 422)
        operation.substeps = [
            "Analyzing source media",
            "Creating optimized codec",
            "Writing optimized file",
            "Verifying output"
        ]
        
        for (index, step) in operation.substeps.enumerated() {
            operation.currentSubstep = step
            operation.progress = Double(index) / Double(operation.substeps.count)
            currentProgress = operation.progress
            
            // Simulate optimization work
            try? await Task.sleep(nanoseconds: 500_000_000) // 500ms per step
        }
        
        operation.progress = 1.0
    }
    
    private func createProxy(_ operation: ImportOperation) async {
        // Create proxy media
        operation.substeps = [
            "Reading source dimensions",
            "Calculating proxy size",
            "Encoding proxy",
            "Linking to original"
        ]
        
        for (index, step) in operation.substeps.enumerated() {
            operation.currentSubstep = step
            operation.progress = Double(index) / Double(operation.substeps.count)
            currentProgress = operation.progress
            
            // Simulate proxy creation
            try? await Task.sleep(nanoseconds: 300_000_000) // 300ms per step
        }
        
        operation.progress = 1.0
    }
    
    private func transcodeFile(_ operation: ImportOperation) async {
        // Transcode to target format
        operation.substeps = [
            "Decoding source",
            "Applying settings",
            "Encoding output",
            "Finalizing"
        ]
        
        for (index, step) in operation.substeps.enumerated() {
            operation.currentSubstep = step
            operation.progress = Double(index) / Double(operation.substeps.count)
            currentProgress = operation.progress
            
            // Simulate transcoding
            try? await Task.sleep(nanoseconds: 750_000_000) // 750ms per step
        }
        
        operation.progress = 1.0
    }
    
    // MARK: - Progress Tracking
    private func updateProgress() {
        let completed = operations.filter { $0.status == .completed }.count
        overallProgress = Double(completed) / Double(operations.count)
        
        // Calculate time remaining
        if let startTime = startTime {
            let elapsed = Date().timeIntervalSince(startTime)
            let rate = overallProgress / elapsed
            if rate > 0 {
                estimatedTimeRemaining = (1 - overallProgress) / rate
            }
        }
    }
    
    private func updateThroughput() {
        guard let startTime = startTime else { return }
        
        let elapsed = Date().timeIntervalSince(startTime)
        if elapsed > 0 {
            let megabytes = Double(processedSize) / 1_048_576
            throughput = megabytes / elapsed
        }
    }
    
    // MARK: - Control Methods
    func pause() {
        isPaused = true
    }
    
    func resume() {
        isPaused = false
    }
    
    func cancel() {
        isProcessing = false
        operationQueue.cancelAllOperations()
        
        // Mark pending operations as cancelled
        for operation in operations where operation.status == .pending {
            operation.status = .cancelled
        }
    }
    
    private func waitForResume() async {
        while isPaused && isProcessing {
            try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
        }
    }
    
    private func calculateFinalMetrics() {
        // Calculate final statistics
        let successful = operations.filter { $0.status == .completed }.count
        let failed = operations.filter { $0.status == .failed }.count
        let skipped = operations.filter { $0.status == .skipped }.count
        
        print("Import complete: \(successful) successful, \(failed) failed, \(skipped) skipped")
    }
    
    private func getProjectFolder() -> URL? {
        // Return project media folder
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AutoResolve/Media")
    }
    
    // MARK: - Computed Properties
    var completedCount: Int {
        operations.filter { $0.status == .completed }.count
    }
    
    var failedCount: Int {
        operations.filter { $0.status == .failed }.count
    }
    
    var remainingCount: Int {
        operations.filter { $0.status == .pending || $0.status == .analyzing }.count
    }
    
    var progressPercentage: Int {
        Int(overallProgress * 100)
    }
    
    var formattedTimeRemaining: String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.unitsStyle = .abbreviated
        return formatter.string(from: estimatedTimeRemaining) ?? "Calculating..."
    }
    
    var formattedThroughput: String {
        String(format: "%.1f MB/s", throughput)
    }
}

// MARK: - Import Operation
class ImportOperation: ObservableObject, Identifiable {
    public let id: UUID
    let file: URL
    let type: OperationType
    let fileSize: Int64
    let settings: ImportSettings
    
    @Published var status: OperationStatus = .pending
    @Published var progress: Double = 0
    @Published var currentSubstep: String = ""
    @Published var substeps: [String] = []
    @Published var error: Error?
    @Published var startTime: Date?
    @Published var endTime: Date?
    
    enum OperationType {
        case `import`
        case copy
        case optimize
        case proxy
        case transcode
    }
    
    enum OperationStatus {
        case pending
        case processing
        case analyzing
        case completed
        case failed
        case cancelled
        case skipped
    }
    
    init(id: UUID, file: URL, type: OperationType, fileSize: Int64, settings: ImportSettings) {
        self.id = id
        self.file = file
        self.type = type
        self.fileSize = fileSize
        self.settings = settings
    }
    
    var fileName: String {
        file.lastPathComponent
    }
    
    var formattedSize: String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: fileSize)
    }
    
    var duration: TimeInterval? {
        guard let startTime = startTime, let endTime = endTime else { return nil }
        return endTime.timeIntervalSince(startTime)
    }
    
    var typeIcon: String {
        switch type {
        case .import: return "square.and.arrow.down"
        case .copy: return "doc.on.doc"
        case .optimize: return "sparkles"
        case .proxy: return "rectangle.compress.vertical"
        case .transcode: return "arrow.triangle.2.circlepath"
        }
    }
    
    var statusColor: Color {
        switch status {
        case .pending: return .secondary
        case .processing: return .blue
        case .analyzing: return .cyan
        case .completed: return .green
        case .failed: return .red
        case .cancelled: return .orange
        case .skipped: return .yellow
        }
    }
}

// MARK: - Import Error
struct ImportError: Identifiable {
    public let id = UUID()
    let operation: ImportOperation
    let error: Error
    let timestamp = Date()
}

// MARK: - Supporting Views
struct BatchImportHeader: View {
    @ObservedObject var batchManager: BatchImportManager
    let onCancel: () -> Void
    
    public var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Batch Import Progress")
                    .font(.system(size: 16, weight: .semibold))
                
                Text("\(batchManager.completedCount) of \(batchManager.operations.count) files")
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Control buttons
            HStack(spacing: 12) {
                if batchManager.isProcessing {
                    Button(action: {
                        if batchManager.isPaused {
                            batchManager.resume()
                        } else {
                            batchManager.pause()
                        }
                    }) {
                        Image(systemName: batchManager.isPaused ? "play.fill" : "pause.fill")
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                    Button(action: onCancel) {
                        Image(systemName: "stop.fill")
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
        }
        .padding()
        .background(DaVinciColors.headerBackground)
    }
}

struct OverallProgressView: View {
    @ObservedObject var batchManager: BatchImportManager
    
    public var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Overall Progress")
                    .font(.system(size: 13, weight: .medium))
                
                Spacer()
                
                Text("\(batchManager.progressPercentage)%")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            
            ProgressView(value: batchManager.overallProgress)
                .progressViewStyle(LinearProgressViewStyle())
            
            HStack {
                Label(batchManager.formattedTimeRemaining, systemImage: "clock")
                
                Spacer()
                
                Label(batchManager.formattedThroughput, systemImage: "speedometer")
            }
            .font(.system(size: 11))
            .foregroundColor(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.2))
        )
    }
}

struct CurrentOperationView: View {
    @ObservedObject var batchManager: BatchImportManager
    
    public var body: some View {
        if let operation = batchManager.currentOperation {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: operation.typeIcon)
                        .foregroundColor(.accentColor)
                    
                    Text(operation.fileName)
                        .font(.system(size: 12, weight: .medium))
                        .lineLimit(1)
                    
                    Spacer()
                    
                    Text(operation.formattedSize)
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
                
                if !operation.currentSubstep.isEmpty {
                    Text(operation.currentSubstep)
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
                
                ProgressView(value: operation.progress)
                    .progressViewStyle(LinearProgressViewStyle())
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.accentColor.opacity(0.3), lineWidth: 1)
            )
        }
    }
}

struct OperationQueueView: View {
    @ObservedObject var batchManager: BatchImportManager
    @Binding var selectedOperation: ImportOperation?
    @Binding var showDetails: Bool
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Queue")
                    .font(.system(size: 13, weight: .medium))
                
                Spacer()
                
                Button(action: { showDetails.toggle() }) {
                    Label(showDetails ? "Hide Details" : "Show Details", 
                          systemImage: showDetails ? "chevron.up" : "chevron.down")
                        .font(.system(size: 11))
                }
                .buttonStyle(LinkButtonStyle())
            }
            
            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(batchManager.operations) { operation in
                        OperationRow(
                            operation: operation,
                            isSelected: selectedOperation?.id == operation.id,
                            onSelect: { selectedOperation = operation }
                        )
                    }
                }
            }
            .frame(maxHeight: 200)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.black.opacity(0.1))
            )
        }
    }
}

struct OperationRow: View {
    @ObservedObject var operation: ImportOperation
    let isSelected: Bool
    let onSelect: () -> Void
    
    public var body: some View {
        HStack(spacing: 8) {
            // Status indicator
            Circle()
                .fill(operation.statusColor)
                .frame(width: 8, height: 8)
            
            // Type icon
            Image(systemName: operation.typeIcon)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .frame(width: 16)
            
            // File name
            Text(operation.fileName)
                .font(.system(size: 11))
                .lineLimit(1)
            
            Spacer()
            
            // Progress or status
            if operation.status == .analyzing {
                ProgressView(value: operation.progress)
                    .progressViewStyle(LinearProgressViewStyle())
                    .frame(width: 60)
            } else {
                Text(statusText)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture(perform: onSelect)
    }
    
    private var statusText: String {
        switch operation.status {
        case .pending: return "Waiting"
        case .processing: return "\(Int(operation.progress * 100))%"
        case .analyzing: return "Analyzing..."
        case .completed: return "Done"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        case .skipped: return "Skipped"
        }
    }
}

struct PerformanceMetricsView: View {
    @ObservedObject var batchManager: BatchImportManager
    
    public var body: some View {
        VStack(spacing: 12) {
            Text("Performance Metrics")
                .font(.system(size: 13, weight: .medium))
            
            HStack(spacing: 20) {
                MetricCard(
                    title: "Processed",
                    value: formatBytes(batchManager.processedSize),
                    trend: .stable,
                    color: .green
                )
                
                MetricCard(
                    title: "Remaining",
                    value: formatBytes(batchManager.totalSize - batchManager.processedSize),
                    trend: .stable,
                    color: .blue
                )
                
                MetricCard(
                    title: "Throughput",
                    value: batchManager.formattedThroughput,
                    trend: .stable,
                    color: .orange
                )
                
                MetricCard(
                    title: "Errors",
                    value: "\(batchManager.failedCount)",
                    trend: batchManager.failedCount > 0 ? .falling : .stable,
                    color: batchManager.failedCount > 0 ? .red : .secondary
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.2))
        )
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}


struct BatchImportFooter: View {
    @ObservedObject var batchManager: BatchImportManager
    @Binding var showDetails: Bool
    let onClose: () -> Void
    
    public var body: some View {
        HStack {
            if batchManager.failedCount > 0 {
                Button(action: { /* Show errors */ }) {
                    Label("\(batchManager.failedCount) errors", systemImage: "exclamationmark.triangle")
                        .foregroundColor(.red)
                }
                .buttonStyle(LinkButtonStyle())
            }
            
            Spacer()
            
            Button("Close") {
                onClose()
            }
            .disabled(batchManager.isProcessing)
            .keyboardShortcut(.escape)
        }
        .padding()
        .background(DaVinciColors.toolbarBackground)
    }
}
