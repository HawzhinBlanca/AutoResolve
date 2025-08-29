import SwiftUI
import AVFoundation

/// Playhead view for timeline scrubbing
public struct PlayheadView: View {
    @Binding var currentTime: TimeInterval
    let duration: TimeInterval
    let height: CGFloat
    
    public init(currentTime: Binding<TimeInterval>, duration: TimeInterval, height: CGFloat = 200) {
        self._currentTime = currentTime
        self.duration = duration
        self.height = height
    }
    
    public var body: some View {
        GeometryReader { geometry in
            let position = currentTime / duration * geometry.size.width
            
            Rectangle()
                .fill(Color.red)
                .frame(width: 2, height: height)
                .position(x: position, y: geometry.size.height / 2)
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            let newTime = Double(value.location.x / geometry.size.width) * duration
                            currentTime = max(0, min(duration, newTime))
                        }
                )
        }
        .frame(height: height)
    }
}
