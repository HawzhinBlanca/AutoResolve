import SwiftUI
import CoreMedia

struct TimelineTrackView: View {
    let track: UITimelineTrack
    let zoomLevel: Double
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HStack(spacing: 0) {
            // Track header
            VStack {
                Text(track.name)
                    .font(.caption)
                    .foregroundColor(.white)
                Text(track.type.rawValue)
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            .frame(width: 60)
            .background(UITheme.Colors.surface)
            
            // Track content area
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Track background
                    Rectangle()
                        .fill(Color.black.opacity(0.2))
                        .frame(height: track.height)
                    
                    // Clips
                    ForEach(track.clips) { clip in
                        TimelineClipView(clip: clip, zoomLevel: zoomLevel)
                            .offset(x: CGFloat(clip.startTime * zoomLevel * 100))
                    }
                }
            }
            .frame(height: track.height)
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            handleTrackDrop(providers: providers, track: track)
        }
    }
    
    private func handleTrackDrop(providers: [NSItemProvider], track: UITimelineTrack) -> Bool {
        for provider in providers {
            if provider.canLoadObject(ofClass: URL.self) {
                provider.loadObject(ofClass: URL.self) { url, error in
                    guard let url = url, error == nil else { return }
                    
                    Task { @MainActor in
                        // Add video specifically to this track
                        appState.addVideoToSpecificTrack(url: url, track: track, at: appState.transport.currentTime)
                    }
                }
                return true
            }
        }
        return false
    }
}

struct TimelineClipView: View {
    let clip: SimpleTimelineClip
    let zoomLevel: Double
    @EnvironmentObject var appState: AppState
    
    var isSelected: Bool {
        appState.selectedClips.contains(clip.id.uuidString)
    }
    
    var body: some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(clip.color)
            .stroke(isSelected ? Color.yellow : Color.clear, lineWidth: 2)
            .frame(
                width: max(20, CGFloat(clip.duration * zoomLevel * 100)),
                height: UITheme.Sizes.timelineTrackHeight - 8
            )
            .overlay(
                VStack {
                    Text(clip.name)
                        .font(.caption2)
                        .foregroundColor(.white)
                        .lineLimit(1)
                    Spacer()
                    if clip.duration > 0 {
                        Text(String(format: "%.1fs", clip.duration))
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
                .padding(2),
                alignment: .topLeading
            )
            .onTapGesture {
                appState.selectClip(clip.id.uuidString)
            }
            .contextMenu {
                Button("Delete") {
                    appState.deleteClip(clip.id.uuidString)
                }
                Button("Duplicate") {
                    appState.duplicateClip(clip.id.uuidString)
                }
                Divider()
                Button("Properties...") {
                    appState.showClipProperties(clip.id.uuidString)
                }
            }
    }
}