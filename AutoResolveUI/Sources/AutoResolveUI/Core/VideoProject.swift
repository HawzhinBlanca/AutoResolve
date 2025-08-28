
import Foundation

// MARK: - Project Model

struct VideoProject: Codable {
    var id: UUID = UUID()
    var name: String = "Untitled"
    var timeline: Timeline = Timeline()

    // Add other project-level settings here
}

// MARK: - Timeline Model

struct Timeline: Codable {
    var clips: [Clip] = []
    // Add other timeline-level properties here (e.g., tracks)

    mutating func moveClip(from source: IndexSet, to destination: Int) {
        clips.move(fromOffsets: source, toOffset: destination)
    }
}

// MARK: - Clip Model

struct Clip: Codable, Identifiable {
    var id: UUID = UUID()
    var name: String
    var assetURL: URL
    var start: TimeInterval
    var end: TimeInterval

    var duration: TimeInterval {
        end - start
    }
}
