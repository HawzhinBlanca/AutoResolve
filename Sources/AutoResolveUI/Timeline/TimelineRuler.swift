import SwiftUI
import AutoResolveCore

struct TimelineRuler: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                drawRuler(context: context, size: size)
            }
        }
        .frame(height: 30)
        .background(Color.gray.opacity(0.2))
    }
    
    private func drawRuler(context: GraphicsContext, size: CGSize) {
        let pixelsPerSecond = 50.0 * appState.zoomLevel
        let totalSeconds = Int(size.width / pixelsPerSecond)
        
        for second in 0...totalSeconds {
            let x = CGFloat(second) * pixelsPerSecond
            
            // Major tick every 5 seconds
            let isMajor = second % 5 == 0
            let height: CGFloat = isMajor ? 15 : 8
            
            context.stroke(
                Path { path in
                    path.move(to: CGPoint(x: x, y: size.height - height))
                    path.addLine(to: CGPoint(x: x, y: size.height))
                },
                with: .color(.gray)
            )
            
            if isMajor {
                context.draw(
                    Text(formatTime(second))
                        .font(.caption2)
                        .foregroundColor(.gray),
                    at: CGPoint(x: x, y: 10)
                )
            }
        }
    }
    
    private func formatTime(_ seconds: Int) -> String {
        let minutes = seconds / 60
        let secs = seconds % 60
        return String(format: "%d:%02d", minutes, secs)
    }
}
