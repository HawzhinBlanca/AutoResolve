
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var projectStore: VideoProjectStore

    var body: some View {
        VStack {
            Text("AutoResolve UI")
                .font(.largeTitle)
            Text("Project: \(projectStore.project.name)")
                .font(.headline)
            Spacer()
            TimelineView()
            Spacer()
        }
        .padding()
    }
}
