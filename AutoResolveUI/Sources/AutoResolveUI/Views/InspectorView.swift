import SwiftUI
import CoreMedia

public struct InspectorView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Text("Inspector")
                .font(UITheme.Typography.headline)
                .foregroundColor(UITheme.Colors.textPrimary)
                .padding(UITheme.Sizes.spacingM)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(UITheme.Colors.surface)
            
            ScrollView {
                VStack(alignment: .leading, spacing: UITheme.Sizes.spacingL) {
                    // Video properties
                    if appState.videoURL != nil {
                        VideoPropertiesSection()
                    }
                    
                    // Selected clip properties
                    if !appState.selectedClips.isEmpty {
                        SelectedClipSection()
                    }
                    
                    // AI Analysis
                    AIAnalysisSection()
                    
                    // Performance metrics
                    PerformanceSection()
                    
                    // Performance gates
                    PerformanceGatesSection()
                }
                .padding(UITheme.Sizes.spacingM)
            }
        }
        .frame(width: UITheme.Sizes.inspectorWidth)
        .background(UITheme.Colors.surface)
    }
}

struct VideoPropertiesSection: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var transport: Transport
    
    var body: some View {
        VStack(alignment: .leading, spacing: UITheme.Sizes.spacingS) {
            SectionHeader(title: "Video Properties")
            
            if let url = appState.videoURL {
                PropertyRow(label: "File", value: url.lastPathComponent)
                PropertyRow(label: "Duration", value: appState.timebase.timecodeFromTime(transport.duration))
                PropertyRow(label: "FPS", value: "\(Int(appState.timebase.fps))")
                PropertyRow(label: "Format", value: url.pathExtension.uppercased())
            }
        }
        .panelStyle()
    }
}

struct SelectedClipSection: View {
    @EnvironmentObject var appState: AppState
    
    var selectedClip: SimpleTimelineClip? {
        guard let timeline = appState.timeline,
              let clipId = appState.selectedClips.first else { return nil }
        
        for track in timeline.tracks {
            if let clip = track.clips.first(where: { $0.id.uuidString == clipId }) {
                return clip
            }
        }
        return nil
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: UITheme.Sizes.spacingS) {
            SectionHeader(title: "Selected Clip")
            
            if let clip = selectedClip {
                PropertyRow(label: "Name", value: clip.name)
                PropertyRow(label: "Start", value: appState.timebase.timecodeFromTime(
                    CMTime(seconds: clip.startTime, preferredTimescale: 600)
                ))
                PropertyRow(label: "Duration", value: appState.timebase.timecodeFromTime(
                    CMTime(seconds: clip.duration, preferredTimescale: 600)
                ))
                PropertyRow(label: "Track", value: "Track \(clip.trackIndex + 1)")
            }
        }
        .panelStyle()
    }
}

struct AIAnalysisSection: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack(alignment: .leading, spacing: UITheme.Sizes.spacingS) {
            SectionHeader(title: "AI Analysis")
            
            // Silence detection
            if let silence = appState.silenceResult {
                HStack {
                    Circle()
                        .fill(UITheme.Colors.silence)
                        .frame(width: 8, height: 8)
                    Text("Silence")
                        .font(UITheme.Typography.body)
                        .foregroundColor(UITheme.Colors.textPrimary)
                    Spacer()
                    Text("\(silence.silenceSegments.count) segments")
                        .font(UITheme.Typography.caption)
                        .foregroundColor(UITheme.Colors.textSecondary)
                }
                
                PropertyRow(
                    label: "Total",
                    value: String(format: "%.1fs",
                                 silence.silenceSegments.reduce(0) { $0 + $1.duration })
                )
            }
            
            // Transcription
            if let transcription = appState.transcriptionResult {
                HStack {
                    Circle()
                        .fill(UITheme.Colors.transcription)
                        .frame(width: 8, height: 8)
                    Text("Transcription")
                        .font(UITheme.Typography.body)
                        .foregroundColor(UITheme.Colors.textPrimary)
                    Spacer()
                    Text("Complete text")
                        .font(UITheme.Typography.caption)
                        .foregroundColor(UITheme.Colors.textSecondary)
                }
                
                PropertyRow(label: "Language", value: transcription.language.uppercased())
            }
            
            // Story beats
            if let beats = appState.storyBeatsResult {
                HStack {
                    Circle()
                        .fill(UITheme.Colors.storyBeat)
                        .frame(width: 8, height: 8)
                    Text("Story Beats")
                        .font(UITheme.Typography.body)
                        .foregroundColor(UITheme.Colors.textPrimary)
                    Spacer()
                    if let beatsArray = beats["beats"] as? [[String: Any]] {
                        Text("\(beatsArray.count) beats")
                            .font(UITheme.Typography.caption)
                            .foregroundColor(UITheme.Colors.textSecondary)
                    }
                }
                
                if let pacing = beats["pacing"] as? String {
                    PropertyRow(label: "Pacing", value: pacing)
                }
            }
            
            // B-roll
            if let broll = appState.brollResult {
                HStack {
                    Circle()
                        .fill(UITheme.Colors.broll)
                        .frame(width: 8, height: 8)
                    Text("B-Roll")
                        .font(UITheme.Typography.body)
                        .foregroundColor(UITheme.Colors.textPrimary)
                    Spacer()
                    Text("\(broll.count) segments")
                        .font(UITheme.Typography.caption)
                        .foregroundColor(UITheme.Colors.textSecondary)
                }
                
                // Calculate coverage
                let totalDuration = broll.reduce(0) { $0 + ($1.timeRange.end - $1.timeRange.start) }
                let coverage = totalDuration / (appState.transport.duration.seconds)
                PropertyRow(
                    label: "Coverage",
                    value: String(format: "%.0f%%", coverage * 100)
                )
            }
            
            if appState.silenceResult == nil &&
               appState.transcriptionResult == nil &&
               appState.storyBeatsResult == nil &&
               appState.brollResult == nil {
                Text("Run AI analysis to see results")
                    .font(UITheme.Typography.caption)
                    .foregroundColor(UITheme.Colors.textDisabled)
            }
        }
        .panelStyle()
    }
}

struct PerformanceSection: View {
    @EnvironmentObject var appState: AppState
    @State private var refreshTimer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()
    @State private var frameRate: Double = 60.0
    @State private var frameTime: Double = 16.67
    
    var memoryUsage: String {
        let task = mach_task_self_
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(task,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let mb = Double(info.resident_size) / 1024.0 / 1024.0
            return String(format: "%.0f MB", mb)
        }
        
        return "N/A"
    }
    
    var memoryStatusColor: Color {
        let task = mach_task_self_
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(task, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let mb = Double(info.resident_size) / 1024.0 / 1024.0
            return mb > 200 ? UITheme.Colors.error : UITheme.Colors.success
        }
        return UITheme.Colors.textSecondary
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: UITheme.Sizes.spacingS) {
            SectionHeader(title: "Performance")
            
            // Real-time metrics
            HStack {
                Text("Memory")
                    .font(UITheme.Typography.caption)
                    .foregroundColor(UITheme.Colors.textSecondary)
                Spacer()
                Circle()
                    .fill(memoryStatusColor)
                    .frame(width: 6, height: 6)
                Text(memoryUsage)
                    .font(UITheme.Typography.body)
                    .foregroundColor(UITheme.Colors.textPrimary)
            }
            
            PropertyRow(
                label: "Frame Rate", 
                value: String(format: "%.0f fps", frameRate)
            )
            PropertyRow(
                label: "Frame Time", 
                value: String(format: "%.1f ms", frameTime)
            )
            
            Divider()
                .padding(.vertical, UITheme.Sizes.spacingXS)
            
            // Backend status
            HStack {
                Text("Backend")
                    .font(UITheme.Typography.caption)
                    .foregroundColor(UITheme.Colors.textSecondary)
                Spacer()
                Circle()
                    .fill(appState.backendClient.isConnected ? UITheme.Colors.success : UITheme.Colors.error)
                    .frame(width: 6, height: 6)
                Text(appState.backendClient.isConnected ? "Connected" : "Offline")
                    .font(UITheme.Typography.body)
                    .foregroundColor(UITheme.Colors.textPrimary)
            }
            
            if let lastProcessTime = appState.lastProcessingTime {
                PropertyRow(
                    label: "Last Process",
                    value: String(format: "%.2fs", lastProcessTime)
                )
            }
            
            PropertyRow(label: "Zoom", value: "\(Int(appState.zoomLevel * 100))%")
            
            if appState.isProcessing {
                Divider()
                    .padding(.vertical, UITheme.Sizes.spacingXS)
                
                HStack {
                    ProgressView()
                        .scaleEffect(0.7)
                    VStack(alignment: .leading) {
                        Text(appState.statusMessage)
                            .font(UITheme.Typography.caption)
                            .foregroundColor(UITheme.Colors.textPrimary)
                        if let progress = appState.processingProgress {
                            Text("\(Int(progress * 100))% complete")
                                .font(UITheme.Typography.caption)
                                .foregroundColor(UITheme.Colors.textSecondary)
                        }
                    }
                    Spacer()
                }
            }
        }
        .panelStyle()
        .onReceive(refreshTimer) { _ in
            updatePerformanceMetrics()
        }
    }
    
    private func updatePerformanceMetrics() {
        // Update frame rate from CADisplayLink if available
        if let displayLink = appState.displayLink {
            frameRate = 1.0 / (displayLink.targetTimestamp - displayLink.timestamp)
            frameTime = (displayLink.targetTimestamp - displayLink.timestamp) * 1000
        }
    }
}

struct SectionHeader: View {
    let title: String
    
    var body: some View {
        Text(title)
            .font(UITheme.Typography.headline)
            .foregroundColor(UITheme.Colors.textPrimary)
    }
}

struct PropertyRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(UITheme.Typography.caption)
                .foregroundColor(UITheme.Colors.textSecondary)
            Spacer()
            Text(value)
                .font(UITheme.Typography.body)
                .foregroundColor(UITheme.Colors.textPrimary)
        }
    }
}

struct PerformanceGatesSection: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack(alignment: .leading, spacing: UITheme.Sizes.spacingS) {
            HStack {
                SectionHeader(title: "Performance Gates")
                Spacer()
                Circle()
                    .fill(appState.performanceMonitor.gateStatus.color)
                    .frame(width: 8, height: 8)
            }
            
            VStack(alignment: .leading, spacing: UITheme.Sizes.spacingXS) {
                ForEach(appState.performanceMonitor.getGateDetails(), id: \.name) { gate in
                    HStack {
                        Circle()
                            .fill(gate.passing ? UITheme.Colors.success : UITheme.Colors.error)
                            .frame(width: 6, height: 6)
                        
                        Text(gate.name)
                            .font(UITheme.Typography.caption)
                            .foregroundColor(UITheme.Colors.textSecondary)
                        
                        Spacer()
                        
                        VStack(alignment: .trailing, spacing: 0) {
                            Text(gate.value)
                                .font(UITheme.Typography.caption)
                                .foregroundColor(gate.passing ? UITheme.Colors.textPrimary : UITheme.Colors.error)
                            Text(gate.target)
                                .font(UITheme.Typography.caption)
                                .foregroundColor(UITheme.Colors.textDisabled)
                        }
                    }
                }
            }
        }
        .panelStyle()
    }
}