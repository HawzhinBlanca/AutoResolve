import SwiftUI
import AutoResolveCore

struct TimelineView: View {
    @EnvironmentObject var appState: AppState
    @State private var dragLocation: CGPoint = .zero
    @State private var isDragging = false
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background
                Color.black.opacity(0.9)
                
                // Timeline tracks
                ScrollView([.horizontal, .vertical]) {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(0..<3, id: \.self) { trackIndex in
                            TrackView(trackIndex: trackIndex)
                                .frame(height: 80)
                        }
                    }
                    .frame(width: max(1000, geometry.size.width * appState.zoomLevel))
                }
                
                // Playhead
                PlayheadView()
                
                // AI Suggestions overlay
                if appState.showAISuggestions {
                    AISuggestionsOverlay()
                }
            }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            handleDrop(providers)
            return true
        }
    }
    
    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        // Handle media drop
        return true
    }
}

struct TrackView: View {
    let trackIndex: Int
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HStack(spacing: 2) {
            if let timeline = appState.timeline,
               trackIndex < timeline.tracks.count {
                ForEach(timeline.tracks[trackIndex].clips, id: \.id) { clip in
                    ClipView(clip: clip)
                }
            }
            Spacer()
        }
        .background(Color.gray.opacity(0.2))
        .border(Color.gray.opacity(0.5), width: 1)
    }
}

struct ClipView: View {
    let clip: Clip
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(Color.blue.opacity(0.7))
            .frame(width: clipWidth)
            .overlay(
                Text(clip.name ?? "Clip")
                    .font(.caption)
                    .foregroundColor(.white)
            )
            .onTapGesture {
                if appState.selectedClips.contains(clip.id) {
                    appState.selectedClips.remove(clip.id)
                } else {
                    appState.selectedClips.insert(clip.id)
                }
            }
    }
    
    private var clipWidth: CGFloat {
        let duration = clip.duration.seconds
        return CGFloat(duration * 50 * appState.zoomLevel)
    }
}

struct PlayheadView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        GeometryReader { geometry in
            Rectangle()
                .fill(Color.red)
                .frame(width: 2, height: geometry.size.height)
                .offset(x: playheadPosition)
        }
        .allowsHitTesting(false)
    }
    
    private var playheadPosition: CGFloat {
        CGFloat(appState.currentTime.seconds * 50 * appState.zoomLevel)
    }
}

struct AISuggestionsOverlay: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        GeometryReader { geometry in
            ForEach(appState.aiSuggestions, id: \.id) { suggestion in
                SuggestionMarker(suggestion: suggestion)
            }
        }
    }
}

struct SuggestionMarker: View {
    let suggestion: EditSuggestion
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack {
            Image(systemName: iconName)
                .foregroundColor(color)
                .font(.caption)
            
            Text(String(format: "%.0f%%", suggestion.confidence * 100))
                .font(.caption2)
                .foregroundColor(.white)
        }
        .offset(x: position, y: 10)
        .onTapGesture {
            appState.applySuggestion(suggestion)
        }
    }
    
    private var position: CGFloat {
        CGFloat(suggestion.tick.seconds * 50 * appState.zoomLevel)
    }
    
    private var iconName: String {
        switch suggestion.type {
        case .cut: return "scissors"
        case .trim: return "arrow.left.and.right"
        case .delete: return "trash"
        case .transition: return "arrow.triangle.2.circlepath"
        }
    }
    
    private var color: Color {
        if suggestion.confidence > 0.8 {
            return .green
        } else if suggestion.confidence > 0.6 {
            return .yellow
        } else {
            return .orange
        }
    }
}
