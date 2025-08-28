
import Foundation

class VideoProjectStore: ObservableObject {
    @Published var project: VideoProject

    private var fileURL: URL

    init(fileURL: URL) {
        self.fileURL = fileURL
        if let data = try? Data(contentsOf: fileURL) {
            if let decodedProject = try? JSONDecoder().decode(VideoProject.self, from: data) {
                self.project = decodedProject
                return
            }
        }
        self.project = VideoProject()
    }

    func save() {
        do {
            let data = try JSONEncoder().encode(project)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            print("Error saving project: \(error)")
        }
    }
}
