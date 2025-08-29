import SwiftUI
import Foundation

/// Renders AI analysis data as timeline overlays
public struct AILaneRenderer: View {
    @EnvironmentObject var appState: AppState
    let zoomLevel: Double
    let timelineWidth: CGFloat
    
    public var body: some View {
        VStack(spacing: 2) {
            // Silence lane
            if appState.showSilence {
                SilenceLane(zoomLevel: zoomLevel, timelineWidth: timelineWidth)
                    .frame(height: 20)
            }
            
            // Story beats lane  
            if appState.showStoryBeats {
                StoryBeatsLane(zoomLevel: zoomLevel, timelineWidth: timelineWidth)
                    .frame(height: 20)
            }
            
            // B-roll lane
            if appState.showBRoll {
                BRollLane(zoomLevel: zoomLevel, timelineWidth: timelineWidth)
                    .frame(height: 20)
            }
        }
    }
}

struct SilenceLane: View {
    @EnvironmentObject var appState: AppState
    let zoomLevel: Double
    let timelineWidth: CGFloat
    
    var body: some View {
        ZStack(alignment: .leading) {
            Rectangle()
                .fill(UITheme.Colors.surface)
            
            // Render silence regions from artifacts
            if let silenceResult = appState.silenceResult {
                ForEach(silenceResult.silenceSegments, id: \.start) { segment in
                    let startX = CGFloat(segment.start * zoomLevel * 100)
                    let width = CGFloat((segment.end - segment.start) * zoomLevel * 100)
                    
                    Rectangle()
                        .fill(UITheme.Colors.silence.opacity(0.6))
                        .frame(width: width)
                        .offset(x: startX)
                }
            } else {
                // Load from artifacts/cuts.json
                SilenceArtifactLoader()
            }
        }
        .cornerRadius(4)
    }
}

struct StoryBeatsLane: View {
    @EnvironmentObject var appState: AppState
    let zoomLevel: Double
    let timelineWidth: CGFloat
    
    var body: some View {
        ZStack(alignment: .leading) {
            Rectangle()
                .fill(UITheme.Colors.surface)
            
            // Load from artifacts/creative_director.json
            StoryBeatsArtifactLoader(zoomLevel: zoomLevel)
        }
        .cornerRadius(4)
    }
}

struct BRollLane: View {
    @EnvironmentObject var appState: AppState
    let zoomLevel: Double
    let timelineWidth: CGFloat
    
    var body: some View {
        ZStack(alignment: .leading) {
            Rectangle()
                .fill(UITheme.Colors.surface)
            
            // Render B-roll suggestions
            if let brollResult = appState.brollResult {
                ForEach(Array(brollResult.enumerated()), id: \.offset) { index, broll in
                    let startX = CGFloat(broll.timeRange.start * zoomLevel * 100)
                    let width = CGFloat(broll.timeRange.duration * zoomLevel * 100)
                    
                    RoundedRectangle(cornerRadius: 2)
                        .stroke(UITheme.Colors.broll, lineWidth: 2)
                        .background(
                            RoundedRectangle(cornerRadius: 2)
                                .fill(UITheme.Colors.broll.opacity(0.3))
                        )
                        .frame(width: width)
                        .offset(x: startX)
                        .overlay(
                            Text("B\(index + 1)")
                                .font(.caption2)
                                .foregroundColor(.white)
                                .offset(x: 4)
                        )
                }
            }
        }
        .cornerRadius(4)
    }
}

/// Loads silence data from artifacts/cuts.json
struct SilenceArtifactLoader: View {
    @EnvironmentObject var appState: AppState
    @State private var silenceSegments: [(start: Double, end: Double)] = []
    
    var body: some View {
        ForEach(Array(silenceSegments.enumerated()), id: \.offset) { index, segment in
            let startX = CGFloat(segment.start * appState.zoomLevel * 100)
            let width = CGFloat((segment.end - segment.start) * appState.zoomLevel * 100)
            
            Rectangle()
                .fill(UITheme.Colors.silence.opacity(0.6))
                .frame(width: width)
                .offset(x: startX)
        }
        .onAppear {
            loadSilenceData()
        }
    }
    
    private func loadSilenceData() {
        let url = Bundle.main.url(forResource: "cuts", withExtension: "json") ??
                  URL(fileURLWithPath: "/Users/hawzhin/AutoResolve/autorez/artifacts/cuts.json")
        
        do {
            let data = try Data(contentsOf: url)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            if let keepWindows = json?["keep_windows"] as? [[String: Any]] {
                silenceSegments = keepWindows.compactMap { window in
                    guard let start = window["start"] as? Double,
                          let end = window["end"] as? Double else { return nil }
                    return (start: start, end: end)
                }
            }
        } catch {
            print("Failed to load silence data: \(error)")
        }
    }
}

/// Loads story beats from artifacts/creative_director.json
struct StoryBeatsArtifactLoader: View {
    @EnvironmentObject var appState: AppState
    let zoomLevel: Double
    @State private var storyBeats: [(time: Double, type: String)] = []
    
    var body: some View {
        ForEach(Array(storyBeats.enumerated()), id: \.offset) { index, beat in
            let x = CGFloat(beat.time * zoomLevel * 100)
            
            VStack(spacing: 0) {
                // Beat marker
                Path { path in
                    switch beat.type {
                    case "incident":
                        // Triangle (▲)
                        path.move(to: CGPoint(x: 0, y: 16))
                        path.addLine(to: CGPoint(x: -6, y: 4))
                        path.addLine(to: CGPoint(x: 6, y: 4))
                        path.closeSubpath()
                    case "climax":
                        // Diamond (◆)
                        path.move(to: CGPoint(x: 0, y: 16))
                        path.addLine(to: CGPoint(x: -4, y: 10))
                        path.addLine(to: CGPoint(x: 0, y: 4))
                        path.addLine(to: CGPoint(x: 4, y: 10))
                        path.closeSubpath()
                    default:
                        // Circle (●)
                        path.addEllipse(in: CGRect(x: -3, y: 7, width: 6, height: 6))
                    }
                }
                .fill(UITheme.Colors.storyBeat)
                .frame(width: 12, height: 16)
                .offset(x: x)
            }
        }
        .onAppear {
            loadStoryBeats()
        }
    }
    
    private func loadStoryBeats() {
        let url = URL(fileURLWithPath: "/Users/hawzhin/AutoResolve/autorez/artifacts/creative_director.json")
        
        do {
            let data = try Data(contentsOf: url)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            if let narrative = json?["narrative"] as? [String: Any],
               let beats = narrative["beats"] as? [[String: Any]] {
                storyBeats = beats.compactMap { beat in
                    guard let time = beat["timestamp"] as? Double,
                          let type = beat["type"] as? String else { return nil }
                    return (time: time, type: type)
                }
            }
        } catch {
            print("Failed to load story beats: \(error)")
        }
    }
}