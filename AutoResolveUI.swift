import SwiftUI
import AVKit
import AppKit

@main
struct AutoResolveUI: App {
    @StateObject private var viewModel = AutoResolveViewModel()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
                .frame(minWidth: 1400, minHeight: 900)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Import Video...") {
                    viewModel.showImporter = true
                }
                .keyboardShortcut("i", modifiers: [.command])
            }
        }
    }
}

@MainActor
class AutoResolveViewModel: ObservableObject {
    @Published var statusMessage = "Ready"
    @Published var isProcessing = false
    @Published var showImporter = false
    @Published var videoURL: URL?
    @Published var player: AVPlayer?
    @Published var currentPage = "Cut"
    @Published var backendConnected = false
    private let backend = BackendClient()
    @Published var silenceDetected = false
    @Published var transcriptionComplete = false
    
    init() {
        checkBackendStatus()
    }
    
    func checkBackendStatus() {
        Task {
            let ok = (try? await backend.checkHealth()) == true
            await MainActor.run {
                self.backendConnected = ok
                self.statusMessage = ok ? "Backend Connected" : "Backend Offline"
            }
        }
    }
    
    func importVideo(url: URL) {
        videoURL = url
        player = AVPlayer(url: url)
        statusMessage = "Loaded: \(url.lastPathComponent)"
    }
    
    func runSilenceDetection() async {
        guard let videoPath = videoURL?.path else { return }
        isProcessing = true
        statusMessage = "Detecting silence..."
        
        do {
            let res = try await backend.analyzeSilence(path: videoPath)
            silenceDetected = !res.ranges.isEmpty
            statusMessage = "Silence detection complete"
        } catch {
            statusMessage = "Silence detection failed"
        }
        
        isProcessing = false
    }
    
    func runTranscription() async {
        guard let videoPath = videoURL?.path else { return }
        isProcessing = true
        statusMessage = "Transcribing..."
        
        do {
            _ = try await backend.asr(path: videoPath, lang: "en")
            transcriptionComplete = true
            statusMessage = "Transcription complete"
        } catch {
            statusMessage = "Transcription failed"
        }
        
        isProcessing = false
    }
    
    func runFullPipeline() async {
        guard let videoPath = videoURL?.path else { return }
        isProcessing = true
        statusMessage = "Running full AI pipeline..."
        
        do {
            let resp = try await backend.startPipeline(videoPath: videoPath, options: .init())
            statusMessage = "Pipeline started: \(resp.taskId)"
            await pollPipelineStatus(taskId: resp.taskId)
        } catch {
            statusMessage = "Pipeline failed"
        }
        
        isProcessing = false
    }
    
    func pollPipelineStatus(taskId: String) async {
        for _ in 0..<60 {
            do {
                let st = try await backend.getPipelineStatus(taskId: taskId)
                if st.status == "completed" {
                    statusMessage = "Pipeline complete"
                    silenceDetected = true
                    transcriptionComplete = true
                    return
                } else if st.status == "failed" {
                    statusMessage = "Pipeline failed"
                    return
                }
                statusMessage = st.message ?? "Processing..."
                try await Task.sleep(nanoseconds: 1_000_000_000)
            } catch {
                break
            }
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var viewModel: AutoResolveViewModel
    @State private var isDragging = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HeaderView()
            
            // Main content
            HStack(spacing: 0) {
                // Sidebar
                SidebarView()
                    .frame(width: 250)
                
                Divider()
                
                // Main area
                VStack(spacing: 0) {
                    // Page selector
                    PageSelector()
                    
                    // Content area
                    ZStack {
                        if viewModel.videoURL != nil {
                            switch viewModel.currentPage {
                            case "Cut":
                                CutPageView()
                            case "Edit":
                                EditPageView()
                            case "Deliver":
                                DeliverPageView()
                            default:
                                EmptyView()
                            }
                        } else {
                            ImportPromptView()
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            
            Divider()
            
            // Status bar
            StatusBarView()
        }
        .background(Color(NSColor.windowBackgroundColor))
        .sheet(isPresented: $viewModel.showImporter) {
            ImporterView()
        }
    }
}

struct HeaderView: View {
    @EnvironmentObject var viewModel: AutoResolveViewModel
    
    var body: some View {
        HStack {
            Text("AutoResolve")
                .font(.title2)
                .fontWeight(.bold)
            
            Spacer()
            
            HStack(spacing: 20) {
                Button("Import") {
                    viewModel.showImporter = true
                }
                .buttonStyle(.borderedProminent)
                
                if viewModel.videoURL != nil {
                    Menu("AI Tools") {
                        Button("Detect Silence") {
                            Task { await viewModel.runSilenceDetection() }
                        }
                        Button("Transcribe") {
                            Task { await viewModel.runTranscription() }
                        }
                        Divider()
                        Button("Run Full Pipeline") {
                            Task { await viewModel.runFullPipeline() }
                        }
                    }
                    .menuStyle(.borderedButton)
                }
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct SidebarView: View {
    @EnvironmentObject var viewModel: AutoResolveViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Project")
                .font(.headline)
                .padding()
            
            if let url = viewModel.videoURL {
                VStack(alignment: .leading, spacing: 10) {
                    Label(url.lastPathComponent, systemImage: "video.fill")
                        .padding(.horizontal)
                    
                    if viewModel.silenceDetected {
                        Label("Silence Detected", systemImage: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .padding(.horizontal)
                    }
                    
                    if viewModel.transcriptionComplete {
                        Label("Transcription Complete", systemImage: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            
            Spacer()
            
            // Backend status
            HStack {
                Circle()
                    .fill(viewModel.backendConnected ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                Text(viewModel.backendConnected ? "Backend Connected" : "Backend Offline")
                    .font(.caption)
            }
            .padding()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
    }
}

struct PageSelector: View {
    @EnvironmentObject var viewModel: AutoResolveViewModel
    
    var body: some View {
        HStack(spacing: 0) {
            ForEach(["Cut", "Edit", "Deliver"], id: \.self) { page in
                Button(action: { viewModel.currentPage = page }) {
                    Text(page)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(viewModel.currentPage == page ? Color.accentColor : Color.clear)
                        .foregroundColor(viewModel.currentPage == page ? .white : .primary)
                }
                .buttonStyle(.plain)
            }
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct CutPageView: View {
    @EnvironmentObject var viewModel: AutoResolveViewModel
    
    var body: some View {
        VStack {
            if let player = viewModel.player {
                VideoPlayer(player: player)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                
                // Transport controls
                HStack {
                    Button(action: {
                        if player.rate == 0 {
                            player.play()
                        } else {
                            player.pause()
                        }
                    }) {
                        Image(systemName: player.rate == 0 ? "play.fill" : "pause.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    
                    Text("Press Space to play/pause")
                        .foregroundColor(.secondary)
                }
                .padding()
            }
        }
    }
}

struct EditPageView: View {
    var body: some View {
        VStack {
            Text("Edit Page")
                .font(.largeTitle)
                .foregroundColor(.secondary)
            Text("Timeline editing features")
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct DeliverPageView: View {
    var body: some View {
        VStack(spacing: 20) {
            Text("Deliver")
                .font(.largeTitle)
            
            VStack(spacing: 15) {
                Button("Export FCPXML") {
                    print("Export FCPXML")
                }
                .buttonStyle(.borderedProminent)
                
                Button("Export EDL") {
                    print("Export EDL")
                }
                .buttonStyle(.borderedProminent)
                
                Button("Export to DaVinci Resolve") {
                    print("Export to Resolve")
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct ImportPromptView: View {
    @EnvironmentObject var viewModel: AutoResolveViewModel
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "video.badge.plus")
                .font(.system(size: 64))
                .foregroundColor(.secondary)
            
            Text("Import a video to begin")
                .font(.title2)
            
            Button("Import Video") {
                viewModel.showImporter = true
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct ImporterView: View {
    @EnvironmentObject var viewModel: AutoResolveViewModel
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        VStack {
            Text("Select a video file")
                .font(.title2)
                .padding()
            
            Button("Choose File...") {
                let panel = NSOpenPanel()
                panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
                panel.allowsMultipleSelection = false
                
                if panel.runModal() == .OK, let url = panel.url {
                    viewModel.importVideo(url: url)
                    dismiss()
                }
            }
            .buttonStyle(.borderedProminent)
            
            Button("Cancel") {
                dismiss()
            }
            .padding()
        }
        .padding()
        .frame(width: 400, height: 200)
    }
}

struct StatusBarView: View {
    @EnvironmentObject var viewModel: AutoResolveViewModel
    
    var body: some View {
        HStack {
            if viewModel.isProcessing {
                ProgressView()
                    .scaleEffect(0.7)
            }
            
            Text(viewModel.statusMessage)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 4)
        .background(Color(NSColor.controlBackgroundColor))
    }
}