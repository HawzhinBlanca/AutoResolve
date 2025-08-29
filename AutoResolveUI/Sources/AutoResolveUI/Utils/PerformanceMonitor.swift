import Foundation
import QuartzCore
import SwiftUI

public class PerformanceMonitor: ObservableObject {
    @Published public var currentFPS: Double = 60.0
    @Published public var frameTime: Double = 16.67
    @Published public var memoryUsage: Double = 0.0
    @Published public var frameDrift: Double = 0.0
    @Published public var gateStatus: GateStatus = .passing
    
    public let gates = PerformanceGates()
    private var frameTimer: Timer?
    private var lastFrameTime: CFTimeInterval = 0
    private var frameTimeHistory: [CFTimeInterval] = []
    private let maxHistorySize = 60 // 1 second of frames at 60fps
    
    public enum GateStatus {
        case passing
        case warning
        case failing
        
        public var color: Color {
            switch self {
            case .passing: return .green
            case .warning: return .yellow
            case .failing: return .red
            }
        }
    }
    
    public init() {
        setupFrameTracking()
        startMemoryTracking()
    }
    
    deinit {
        frameTimer?.invalidate()
    }
    
    private func setupFrameTracking() {
        frameTimer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { _ in
            self.updateFrameMetrics()
        }
    }
    
    private func updateFrameMetrics() {
        let currentTime = CFAbsoluteTimeGetCurrent()
        
        if lastFrameTime > 0 {
            let deltaTime = currentTime - lastFrameTime
            frameTimeHistory.append(deltaTime)
            
            // Keep only recent history
            if frameTimeHistory.count > maxHistorySize {
                frameTimeHistory.removeFirst()
            }
            
            // Calculate smoothed metrics
            let avgFrameTime = frameTimeHistory.reduce(0, +) / Double(frameTimeHistory.count)
            frameTime = avgFrameTime * 1000 // Convert to ms
            currentFPS = 1.0 / avgFrameTime
            
            // Calculate frame drift (deviation from target 16.67ms)
            let targetFrameTime = 1.0 / 60.0
            frameDrift = abs(avgFrameTime - targetFrameTime) * 1000
        }
        
        lastFrameTime = currentTime
        updateMemoryUsage()
        checkGates()
    }
    
    private func updateMemoryUsage() {
        let task = mach_task_self_
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(task, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if result == KERN_SUCCESS {
            memoryUsage = Double(info.resident_size) / 1024.0 / 1024.0 // MB
        }
    }
    
    private func startMemoryTracking() {
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            self.updateMemoryUsage()
        }
    }
    
    private func checkGates() {
        let fpsGate = currentFPS >= gates.minFPS
        let frameTimeGate = frameTime <= gates.maxFrameTime
        let memoryGate = memoryUsage <= gates.maxMemoryMB
        let driftGate = frameDrift <= gates.maxFrameDriftMS
        
        let passingCount = [fpsGate, frameTimeGate, memoryGate, driftGate].filter { $0 }.count
        
        switch passingCount {
        case 4:
            gateStatus = .passing
        case 2, 3:
            gateStatus = .warning
        default:
            gateStatus = .failing
        }
    }
    
    public func getGateDetails() -> [GateResult] {
        return [
            GateResult(
                name: "Frame Rate",
                value: String(format: "%.1f fps", currentFPS),
                target: "≥60 fps",
                passing: currentFPS >= gates.minFPS
            ),
            GateResult(
                name: "Frame Time", 
                value: String(format: "%.1f ms", frameTime),
                target: "≤16 ms",
                passing: frameTime <= gates.maxFrameTime
            ),
            GateResult(
                name: "Memory",
                value: String(format: "%.0f MB", memoryUsage),
                target: "≤200 MB", 
                passing: memoryUsage <= gates.maxMemoryMB
            ),
            GateResult(
                name: "Frame Drift",
                value: String(format: "%.2f ms", frameDrift),
                target: "≤1 ms",
                passing: frameDrift <= gates.maxFrameDriftMS
            )
        ]
    }
}

public struct PerformanceGates {
    public let minFPS: Double = 60.0
    public let maxFrameTime: Double = 16.0 // ms
    public let maxMemoryMB: Double = 200.0
    public let maxFrameDriftMS: Double = 1.0
}

public struct GateResult {
    public let name: String
    public let value: String
    public let target: String
    public let passing: Bool
}