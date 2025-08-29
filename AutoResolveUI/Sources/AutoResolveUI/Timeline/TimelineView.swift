import SwiftUI

struct TimelineView: View {
    @EnvironmentObject var projectStore: BackendVideoProjectStore

    var body: some View {
        List {
            ForEach(projectStore.timeline.tracks.flatMap { $0.clips }) { clip in
                Text(clip.name)
            }
            .onMove(perform: moveClip)
        }
    }

    private let backendService = BackendClient()

    private func moveClip(from source: IndexSet, to destination: Int) {
        guard let fromIndex = source.first else { return }
        let clips = projectStore.timeline.tracks.flatMap { $0.clips }
        guard fromIndex < clips.count else { return }
        let clipId = clips[fromIndex].id.uuidString

        Task {
            do {
                let success = try await backendService.moveClip(clipId: clipId, fromIndex: fromIndex, toIndex: destination)
                if success {
                    // Move logic handled by backend, update local state if needed
                    print("Clip moved successfully")
                } else {
                    print("Failed to move clip on backend")
                }
            } catch {
                print("Error moving clip: \(error)")
            }
        }
    }
}