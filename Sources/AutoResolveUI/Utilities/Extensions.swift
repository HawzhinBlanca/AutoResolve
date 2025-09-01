import SwiftUI
import AutoResolveCore

extension View {
    func onHover(perform: @escaping (Bool) -> Void) -> some View {
        self.onHover { isHovering in
            perform(isHovering)
        }
    }
}

extension Color {
    static let timelineBackground = Color(white: 0.1)
    static let trackBackground = Color(white: 0.15)
    static let clipDefault = Color.blue.opacity(0.7)
    static let clipSelected = Color.blue
    static let playhead = Color.red
}

extension Tick {
    var displayString: String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        let frames = Int((seconds - Double(totalSeconds)) * 30)
        
        if hours > 0 {
            return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
        } else {
            return String(format: "%02d:%02d:%02d", minutes, secs, frames)
        }
    }
}
