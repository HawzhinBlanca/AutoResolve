
import SwiftUI

struct MenuBarCommands: Commands {
    @EnvironmentObject var appState: AppState

    var body: some Commands {
        CommandMenu("File") {
            Button("Export to FCPXML") {
                Task {
                    await appState.exportFCPXML()
                }
            }
        }
    }
}
