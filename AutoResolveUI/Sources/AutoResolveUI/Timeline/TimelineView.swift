import SwiftUI

struct TimelineView: View {
    @EnvironmentObject var projectStore: VideoProjectStore

    var body: some View {
        List {
            ForEach(projectStore.project.timeline.clips) { clip in
                Text(clip.name)
            }
            .onMove(perform: moveClip)
        }
    }

    private let backendService = BackendService()

    private func moveClip(from source: IndexSet, to destination: Int) {
        guard let fromIndex = source.first else { return }
        let clipId = projectStore.project.timeline.clips[fromIndex].id.uuidString

        backendService.moveClip(clipId: clipId, fromIndex: fromIndex, toIndex: destination) { success in
            if success {
                DispatchQueue.main.async {
                    self.projectStore.project.timeline.moveClip(from: source, to: destination)
                }
            } else {
                // Handle error
                print("Failed to move clip on backend")
            }
        }
    }
}