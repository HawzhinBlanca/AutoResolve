import SwiftUI

struct TimelineToolbar: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HStack {
            Button(action: { appState.blade() }) {
                Image(systemName: "scissors")
            }
            .disabled(!appState.canBlade)
            
            Button(action: { appState.deleteSelection() }) {
                Image(systemName: "trash")
            }
            .disabled(!appState.hasSelection)
            
            Divider()
            
            Button(action: { appState.zoomOut() }) {
                Image(systemName: "minus.magnifyingglass")
            }
            
            Button(action: { appState.zoomToFit() }) {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
            }
            
            Button(action: { appState.zoomIn() }) {
                Image(systemName: "plus.magnifyingglass")
            }
            
            Spacer()
            
            if appState.aiAnalyzing {
                ProgressView()
                    .scaleEffect(0.7)
                Text("AI Analyzing...")
                    .font(.caption)
            }
        }
        .padding(.horizontal)
        .frame(height: 30)
        .background(Color.gray.opacity(0.1))
    }
}
