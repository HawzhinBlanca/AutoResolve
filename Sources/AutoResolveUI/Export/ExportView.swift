import SwiftUI
import AutoResolveCore

struct ExportView: View {
    @EnvironmentObject var appState: AppState
    @State private var exportFormat = ExportFormat.mov
    @State private var exportQuality = ExportQuality.high
    @State private var isExporting = false
    
    enum ExportFormat: String, CaseIterable {
        case mov = "QuickTime"
        case mp4 = "MP4"
        case prores = "ProRes"
    }
    
    enum ExportQuality: String, CaseIterable {
        case low = "Low"
        case medium = "Medium"
        case high = "High"
        case maximum = "Maximum"
    }
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Export Settings")
                .font(.title2)
            
            Form {
                Picker("Format", selection: $exportFormat) {
                    ForEach(ExportFormat.allCases, id: \.self) { format in
                        Text(format.rawValue).tag(format)
                    }
                }
                
                Picker("Quality", selection: $exportQuality) {
                    ForEach(ExportQuality.allCases, id: \.self) { quality in
                        Text(quality.rawValue).tag(quality)
                    }
                }
            }
            
            HStack {
                Button("Cancel") {
                    // Dismiss
                }
                
                Button("Export") {
                    startExport()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isExporting)
            }
            
            if isExporting {
                ProgressView("Exporting...")
                    .progressViewStyle(.linear)
            }
        }
        .padding()
        .frame(width: 400, height: 300)
    }
    
    private func startExport() {
        isExporting = true
        
        Task {
            // Perform export
            await performExport()
            isExporting = false
        }
    }
    
    private func performExport() async {
        // Export implementation
    }
}
