import SwiftUI

struct KeyboardShortcuts: ViewModifier {
    @EnvironmentObject var appState: AppState
    
    func body(content: Content) -> some View {
        content
            .onKeyPress(.space) {
                appState.playPause()
                return .handled
            }
            .onKeyPress(.leftArrow) {
                appState.stepBackward()
                return .handled
            }
            .onKeyPress(.rightArrow) {
                appState.stepForward()
                return .handled
            }
            .onKeyPress(.delete) {
                if appState.hasSelection {
                    appState.deleteSelection()
                    return .handled
                }
                return .ignored
            }
    }
}

extension View {
    func keyboardShortcuts() -> some View {
        modifier(KeyboardShortcuts())
    }
}
