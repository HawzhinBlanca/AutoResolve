
import SwiftUI

struct MenuBarCommands: Commands {
    @EnvironmentObject var projectStore: VideoProjectStore

    var body: some Commands {
        CommandMenu("File") {
            Button("Export to FCPXML") {
                let exporter = FCPXMLExporter()
                exporter.export(projectStore.project.timeline)
            }
        }
    }
}
