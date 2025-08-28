import SwiftUI
import Combine

@main
struct AutoResolveApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 1200, minHeight: 800)
        }
    }
}

struct ContentView: View {
    @State private var isProcessing = false
    @State private var videoPath = ""
    @State private var logText = "AutoResolve V3.0 Ready\n"
    
    var body: some View {
        VStack(spacing: 20) {
            Text("AutoResolve V3.0")
                .font(.largeTitle)
                .bold()
            
            HStack {
                TextField("Video Path", text: $videoPath)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 400)
                
                Button("Browse") {
                    let panel = NSOpenPanel()
                    panel.allowedContentTypes = [.movie]
                    if panel.runModal() == .OK {
                        videoPath = panel.url?.path ?? ""
                    }
                }
            }
            
            HStack(spacing: 20) {
                Button("Process Video") {
                    processVideo()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessing || videoPath.isEmpty)
                
                Button("Export MP4") {
                    exportVideo()
                }
                .buttonStyle(.bordered)
                
                Button("Open Exports") {
                    NSWorkspace.shared.open(URL(fileURLWithPath: "/Users/hawzhin/AutoResolve/exports"))
                }
                .buttonStyle(.bordered)
            }
            
            if isProcessing {
                ProgressView()
                    .progressViewStyle(.circular)
            }
            
            ScrollView {
                Text(logText)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .frame(maxHeight: 400)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
        }
        .padding(40)
    }
    
    func processVideo() {
        isProcessing = true
        logText += "Processing: \(videoPath)\n"
        
        Task {
            do {
                let url = URL(string: "http://localhost:8000/api/pipeline/start")!
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                
                let body = ["video_path": videoPath]
                request.httpBody = try JSONEncoder().encode(body)
                
                let (data, _) = try await URLSession.shared.data(for: request)
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let taskId = json["task_id"] as? String {
                    logText += "Started: Task ID \(taskId)\n"
                    await checkProgress(taskId: taskId)
                }
            } catch {
                logText += "Error: \(error.localizedDescription)\n"
            }
            isProcessing = false
        }
    }
    
    func checkProgress(taskId: String) async {
        for _ in 0..<60 {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            
            let url = URL(string: "http://localhost:8000/api/pipeline/status/\(taskId)")!
            if let data = try? Data(contentsOf: url),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                
                let status = json["status"] as? String ?? "unknown"
                let progress = json["progress"] as? Double ?? 0
                
                logText += "Status: \(status) - Progress: \(Int(progress * 100))%\n"
                
                if status == "completed" {
                    logText += "✅ Processing complete!\n"
                    if let result = json["result"] as? [String: Any],
                       let clips = result["timeline_clips"] as? Int {
                        logText += "Generated \(clips) clips\n"
                    }
                    break
                } else if status == "failed" {
                    logText += "❌ Processing failed\n"
                    break
                }
            }
        }
    }
    
    func exportVideo() {
        logText += "Export functionality - connect to backend\n"
    }
}
