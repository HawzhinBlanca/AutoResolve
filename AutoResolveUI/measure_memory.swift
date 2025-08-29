#!/usr/bin/env swift
//
// UI Memory Usage Measurement for AutoResolve v3.2
// Blueprint Compliance: REQ-009 - UI memory under 200MB
// 100% Compliance Protocol - Zero Tolerance for Deviation
//

import Foundation
import AppKit

struct MemoryMeasurement {
    let timestamp: Date
    let residentMemoryMB: Double
    let virtualMemoryMB: Double
    let peakMemoryMB: Double
    let memoryPressure: String
}

class UIMemoryProfiler {
    static func getCurrentMemoryUsage() -> MemoryMeasurement {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        let residentMB = Double(info.resident_size) / 1024.0 / 1024.0
        let virtualMB = Double(info.virtual_size) / 1024.0 / 1024.0
        
        // Get peak memory from system
        let peakMB = residentMB  // For now, track current as peak
        
        // Determine memory pressure
        let pressure: String
        if residentMB < 100 {
            pressure = "LOW"
        } else if residentMB < 150 {
            pressure = "MEDIUM"
        } else if residentMB < 200 {
            pressure = "HIGH"
        } else {
            pressure = "CRITICAL"
        }
        
        return MemoryMeasurement(
            timestamp: Date(),
            residentMemoryMB: residentMB,
            virtualMemoryMB: virtualMB,
            peakMemoryMB: peakMB,
            memoryPressure: pressure
        )
    }
    
    static func measureUIProcess(pid: Int32? = nil) -> MemoryMeasurement? {
        let targetPid = pid ?? getpid()
        
        // Use ps command to get accurate memory
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/ps")
        task.arguments = ["-o", "rss,vsz", "-p", "\(targetPid)"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        
        do {
            try task.run()
            task.waitUntilExit()
            
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8) {
                let lines = output.split(separator: "\n")
                if lines.count > 1 {
                    let values = lines[1].split(separator: " ").compactMap { Int($0) }
                    if values.count >= 2 {
                        let rssKB = Double(values[0])
                        let vszKB = Double(values[1])
                        
                        return MemoryMeasurement(
                            timestamp: Date(),
                            residentMemoryMB: rssKB / 1024.0,
                            virtualMemoryMB: vszKB / 1024.0,
                            peakMemoryMB: rssKB / 1024.0,
                            memoryPressure: rssKB / 1024.0 > 200 ? "CRITICAL" : "NORMAL"
                        )
                    }
                }
            }
        } catch {
            print("Error measuring memory: \(error)")
        }
        
        return nil
    }
    
    static func findUIProcess() -> Int32? {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/ps")
        task.arguments = ["aux"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        
        do {
            try task.run()
            task.waitUntilExit()
            
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8) {
                let lines = output.split(separator: "\n")
                for line in lines {
                    if line.contains("MainApp") || line.contains("AutoResolveUI") {
                        let components = line.split(separator: " ", maxSplits: 11)
                        if components.count > 1 {
                            if let pid = Int32(components[1]) {
                                return pid
                            }
                        }
                    }
                }
            }
        } catch {
            print("Error finding UI process: \(error)")
        }
        
        return nil
    }
    
    static func runComplianceTest() -> Bool {
        print("=" * 80)
        print("UI MEMORY COMPLIANCE TEST")
        print("Blueprint: REQ-009 - UI Memory Under 200MB")
        print("=" * 80)
        print()
        
        // Find running UI process
        if let pid = findUIProcess() {
            print("Found UI process with PID: \(pid)")
            
            // Take multiple measurements
            var measurements: [MemoryMeasurement] = []
            
            print("\nTaking measurements...")
            for i in 1...10 {
                if let measurement = measureUIProcess(pid: pid) {
                    measurements.append(measurement)
                    print("  Sample \(i): \(String(format: "%.1f", measurement.residentMemoryMB)) MB")
                    Thread.sleep(forTimeInterval: 0.5)
                }
            }
            
            if !measurements.isEmpty {
                // Calculate statistics
                let avgMemory = measurements.map { $0.residentMemoryMB }.reduce(0, +) / Double(measurements.count)
                let maxMemory = measurements.map { $0.residentMemoryMB }.max() ?? 0
                let minMemory = measurements.map { $0.residentMemoryMB }.min() ?? 0
                
                print("\n" + "-" * 80)
                print("RESULTS:")
                print("  Average Memory: \(String(format: "%.1f", avgMemory)) MB")
                print("  Peak Memory: \(String(format: "%.1f", maxMemory)) MB")
                print("  Min Memory: \(String(format: "%.1f", minMemory)) MB")
                print("  Requirement: < 200 MB")
                print("  Status: \(maxMemory < 200 ? "✅ PASSED" : "❌ FAILED")")
                print("-" * 80)
                
                // Save report
                let report = [
                    "timestamp": ISO8601DateFormatter().string(from: Date()),
                    "samples": measurements.count,
                    "average_mb": avgMemory,
                    "peak_mb": maxMemory,
                    "min_mb": minMemory,
                    "requirement_mb": 200,
                    "passed": maxMemory < 200
                ] as [String : Any]
                
                if let jsonData = try? JSONSerialization.data(withJSONObject: report, options: .prettyPrinted) {
                    let reportPath = "/Users/hawzhin/AutoResolve/autorez/artifacts/ui_memory_report.json"
                    try? jsonData.write(to: URL(fileURLWithPath: reportPath))
                    print("\nReport saved to: \(reportPath)")
                }
                
                return maxMemory < 200
            }
        } else {
            print("❌ No UI process found. Please launch MainApp or AutoResolveUI first.")
            print("\nTo launch UI:")
            print("  cd /Users/hawzhin/AutoResolve/AutoResolveUI")
            print("  swift build --target MainApp && .build/debug/MainApp")
            return false
        }
        
        return false
    }
}

// Extension for string multiplication
extension String {
    static func * (string: String, scalar: Int) -> String {
        return String(repeating: string, count: scalar)
    }
}

// Main execution
if CommandLine.arguments.contains("--test") {
    let passed = UIMemoryProfiler.runComplianceTest()
    exit(passed ? 0 : 1)
} else {
    // Single measurement
    if let pid = UIMemoryProfiler.findUIProcess() {
        if let measurement = UIMemoryProfiler.measureUIProcess(pid: pid) {
            print("UI Memory Usage: \(String(format: "%.1f", measurement.residentMemoryMB)) MB")
            print("Status: \(measurement.memoryPressure)")
            print("Compliance: \(measurement.residentMemoryMB < 200 ? "✅ PASSED" : "❌ FAILED")")
        }
    } else {
        print("No UI process found")
    }
}