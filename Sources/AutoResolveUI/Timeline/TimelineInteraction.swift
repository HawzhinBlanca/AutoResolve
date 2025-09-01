import SwiftUI
import AutoResolveCore

struct TimelineInteraction: ViewModifier {
    @EnvironmentObject var appState: AppState
    @State private var dragStart: CGPoint?
    @State private var selection: CGRect?
    
    func body(content: Content) -> some View {
        content
            .onTapGesture { location in
                handleClick(at: location)
            }
            .gesture(
                DragGesture()
                    .onChanged { value in
                        handleDrag(value)
                    }
                    .onEnded { _ in
                        endDrag()
                    }
            )
    }
    
    private func handleClick(at location: CGPoint) {
        // Update playhead position
        let tick = locationToTick(location)
        appState.currentTime = tick
    }
    
    private func handleDrag(_ value: DragGesture.Value) {
        if dragStart == nil {
            dragStart = value.startLocation
        }
        
        // Update selection rectangle
        selection = CGRect(
            x: min(value.startLocation.x, value.location.x),
            y: min(value.startLocation.y, value.location.y),
            width: abs(value.location.x - value.startLocation.x),
            height: abs(value.location.y - value.startLocation.y)
        )
    }
    
    private func endDrag() {
        dragStart = nil
        selection = nil
    }
    
    private func locationToTick(_ location: CGPoint) -> Tick {
        let seconds = location.x / (50.0 * appState.zoomLevel)
        return Tick.from(seconds: seconds)
    }
}

extension View {
    func timelineInteraction() -> some View {
        modifier(TimelineInteraction())
    }
}
