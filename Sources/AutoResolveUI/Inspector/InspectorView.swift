import SwiftUI
import AutoResolveCore

struct InspectorView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack {
            if !appState.selectedClips.isEmpty {
                ClipInspector()
            } else if let _ = appState.currentProject {
                ProjectInspector()
            } else {
                EmptyInspector()
            }
        }
        .frame(width: 300)
        .background(Color.gray.opacity(0.1))
    }
}

struct ClipInspector: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        Form {
            Section("Clip Properties") {
                // Properties
            }
            
            Section("Transform") {
                // Transform controls
            }
            
            Section("Effects") {
                // Effects list
            }
        }
        .padding()
    }
}

struct ProjectInspector: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        Form {
            Section("Project Settings") {
                if let project = appState.currentProject {
                    LabeledContent("Name", value: project.name)
                    LabeledContent("Resolution", value: "1920x1080")
                    LabeledContent("Frame Rate", value: "30 fps")
                }
            }
        }
        .padding()
    }
}

struct EmptyInspector: View {
    var body: some View {
        VStack {
            Spacer()
            Text("No Selection")
                .foregroundColor(.gray)
            Spacer()
        }
    }
}
