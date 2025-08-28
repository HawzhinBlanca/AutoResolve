import Foundation
import AppKit

// Simple GUI app that connects to the backend
class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var statusLabel: NSTextField!
    var processButton: NSButton!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create window
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 400),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.center()
        window.title = "AutoResolve V3.0"
        window.makeKeyAndOrderFront(nil)
        
        // Create UI
        let contentView = NSView(frame: window.contentView!.bounds)
        window.contentView = contentView
        
        // Title
        let titleLabel = NSTextField(labelWithString: "AutoResolve V3.0 - Video Processing")
        titleLabel.font = .boldSystemFont(ofSize: 20)
        titleLabel.frame = NSRect(x: 150, y: 350, width: 300, height: 30)
        contentView.addSubview(titleLabel)
        
        // Status label
        statusLabel = NSTextField(labelWithString: "Backend Status: Checking...")
        statusLabel.frame = NSRect(x: 50, y: 300, width: 500, height: 30)
        contentView.addSubview(statusLabel)
        
        // Process button
        processButton = NSButton(title: "Process Test Video", target: self, action: #selector(processVideo))
        processButton.frame = NSRect(x: 200, y: 250, width: 200, height: 40)
        contentView.addSubview(processButton)
        
        // Export button
        let exportButton = NSButton(title: "Open Exports Folder", target: self, action: #selector(openExports))
        exportButton.frame = NSRect(x: 200, y: 200, width: 200, height: 40)
        contentView.addSubview(exportButton)
        
        // Check backend
        checkBackend()
    }
    
    func checkBackend() {
        let url = URL(string: "http://localhost:8000/health")!
        let task = URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let status = json["status"] as? String {
                    self?.statusLabel.stringValue = "✅ Backend Status: \(status) - Ready to process"
                    self?.processButton.isEnabled = true
                } else {
                    self?.statusLabel.stringValue = "❌ Backend not running - Start backend_service_final.py"
                    self?.processButton.isEnabled = false
                }
            }
        }
        task.resume()
    }
    
    @MainActor @objc func processVideo() {
        statusLabel.stringValue = "🔄 Processing video..."
        processButton.isEnabled = false
        
        let url = URL(string: "http://localhost:8000/api/pipeline/start")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = ["video_path": "/Users/hawzhin/Videos/test_video.mp4"]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let taskId = json["task_id"] as? String {
                    self?.statusLabel.stringValue = "✅ Processing started: \(taskId)"
                    self?.checkProgress(taskId: taskId)
                } else {
                    self?.statusLabel.stringValue = "❌ Failed to start processing"
                    self?.processButton.isEnabled = true
                }
            }
        }
        task.resume()
    }
    
    func checkProgress(taskId: String) {
        Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] timer in
            let url = URL(string: "http://localhost:8000/api/pipeline/status/\(taskId)")!
            let task = URLSession.shared.dataTask(with: url) { data, response, error in
                DispatchQueue.main.async {
                    if let data = data,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        let status = json["status"] as? String ?? "unknown"
                        let progress = json["progress"] as? Double ?? 0
                        
                        if status == "completed" {
                            self?.statusLabel.stringValue = "✅ Processing complete!"
                            self?.processButton.isEnabled = true
                            timer.invalidate()
                            
                            // Show alert
                            let alert = NSAlert()
                            alert.messageText = "Processing Complete"
                            alert.informativeText = "Video has been processed successfully. Check exports folder."
                            alert.runModal()
                        } else if status == "failed" {
                            self?.statusLabel.stringValue = "❌ Processing failed"
                            self?.processButton.isEnabled = true
                            timer.invalidate()
                        } else {
                            self?.statusLabel.stringValue = "🔄 Processing... \(Int(progress * 100))%"
                        }
                    }
                }
            }
            task.resume()
        }
    }
    
    @MainActor @objc func openExports() {
        NSWorkspace.shared.open(URL(fileURLWithPath: "/Users/hawzhin/AutoResolve/exports"))
    }
}

// Main
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()