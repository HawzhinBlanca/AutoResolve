import SwiftUI
import AVFoundation
import CoreMedia
import Vision
import Combine

// MARK: - Multi-Camera Editor

@MainActor
public class MultiCamEditor: ObservableObject {
    @Published public var cameras: [CameraAngle] = []
    @Published public var selectedCamera: CameraAngle?
    @Published public var syncMode: SyncMode = .timecode
    @Published public var isSyncing = false
    @Published public var syncProgress: Double = 0.0
    @Published public var previewMode: PreviewMode = .quad
    @Published public var isPlaying = false
    @Published public var currentTime: CMTime = .zero
    @Published public var duration: CMTime = .zero
    
    private let syncQueue = DispatchQueue(label: "multicam.sync", qos: .userInitiated)
    private let renderQueue = DispatchQueue(label: "multicam.render", qos: .userInitiated)
    private var cancellables = Set<AnyCancellable>()
    private let logger = Logger.shared
    
    public init() {
        setupBindings()
    }
    
    // MARK: - Camera Angle
    
    public struct CameraAngle: Identifiable, Hashable {
        public let id: UUID
        public var name: String
        public var url: URL
        public var asset: AVAsset?
        public var videoTrack: AVAssetTrack?
        public var audioTrack: AVAssetTrack?
        public var syncOffset: CMTime = .zero
        public var colorCorrection: ColorCorrection = ColorCorrection()
        public var isEnabled = true
        public var angle: AngleType = .main
        public var metadata: CameraMetadata?
        
        public init(id: UUID = UUID(), name: String, url: URL, asset: AVAsset? = nil, videoTrack: AVAssetTrack? = nil, audioTrack: AVAssetTrack? = nil) {
            self.id = id
            self.name = name
            self.url = url
            self.asset = asset
            self.videoTrack = videoTrack
            self.audioTrack = audioTrack
        }
        
        // Custom Hashable implementation that excludes non-hashable properties
        public nonisolated func hash(into hasher: inout Hasher) {
            hasher.combine(id)
            hasher.combine(url)
            hasher.combine(name)
        }
        
        public nonisolated static func == (lhs: CameraAngle, rhs: CameraAngle) -> Bool {
            return lhs.id == rhs.id && lhs.url == rhs.url && lhs.name == rhs.name
        }
        
        public enum AngleType: String, CaseIterable {
            case main = "Main"
            case wide = "Wide"
            case medium = "Medium"
            case closeUp = "Close-up"
            case overhead = "Overhead"
            case side = "Side"
            case reaction = "Reaction"
            case bRoll = "B-Roll"
            case custom = "Custom"
        }
        
        public struct ColorCorrection {
            public var exposure: Float = 0
            public var contrast: Float = 1
            public var saturation: Float = 1
            public var temperature: Float = 0
            public var tint: Float = 0
            public var highlights: Float = 0
            public var shadows: Float = 0
            public var whites: Float = 0
            public var blacks: Float = 0
        }
        
        public struct CameraMetadata {
            public var cameraModel: String?
            public var lensInfo: String?
            public var iso: Int?
            public var aperture: Float?
            public var shutterSpeed: String?
            public var frameRate: Double?
            public var resolution: CGSize?
            public var colorSpace: String?
            public var timecode: String?
        }
    }
    
    // MARK: - Sync Modes
    
    public enum SyncMode: String, CaseIterable {
        case timecode = "Timecode"
        case audio = "Audio Waveform"
        case clapperboard = "Clapperboard"
        case manual = "Manual"
        case content = "Content Analysis"
        case metadata = "Metadata"
    }
    
    // MARK: - Preview Modes
    
    public enum PreviewMode: String, CaseIterable {
        case single = "Single"
        case dual = "Dual"
        case quad = "Quad"
        case grid = "Grid"
        case pip = "Picture-in-Picture"
        case comparison = "Comparison"
    }
    
    // MARK: - Camera Management
    
    public func addCamera(from url: URL) async throws {
        logger.info("Adding camera from: \(url.lastPathComponent)")
        
        let asset = AVAsset(url: url)
        
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw MultiCamError.noVideoTrack
        }
        
        let audioTrack = try? await asset.loadTracks(withMediaType: .audio).first
        
        let metadata = try await extractMetadata(from: asset)
        
        var camera = CameraAngle(
            name: url.deletingPathExtension().lastPathComponent,
            url: url,
            asset: asset,
            videoTrack: videoTrack,
            audioTrack: audioTrack
        )
        camera.metadata = metadata
        
        await MainActor.run {
            cameras.append(camera)
            if selectedCamera == nil {
                selectedCamera = camera
            }
        }
        
        logger.info("Camera added: \(camera.name)")
    }
    
    public func removeCamera(_ camera: CameraAngle) {
        cameras.removeAll { $0.id == camera.id }
        if selectedCamera?.id == camera.id {
            selectedCamera = cameras.first
        }
    }
    
    // MARK: - Synchronization
    
    public func syncCameras() async throws {
        await MainActor.run {
            isSyncing = true
            syncProgress = 0
        }
        
        defer {
            Task { @MainActor in
                isSyncing = false
            }
        }
        
        switch self.syncMode {
        case .timecode:
            try await syncByTimecode()
        case .audio:
            try await syncByAudio()
        case .clapperboard:
            try await syncByClapperboard()
        case .content:
            try await syncByContent()
        case .metadata:
            try await syncByMetadata()
        case .manual:
            break
        }
        
        await MainActor.run {
            syncProgress = 1.0
        }
        
        logger.info("Cameras synchronized using \(self.syncMode.rawValue)")
    }
    
    private func syncByTimecode() async throws {
        guard cameras.count > 1 else { return }
        
        var timecodes: [(CameraAngle, CMTime)] = []
        
        for camera in cameras {
            if let asset = camera.asset,
               let timecode = try await extractTimecode(from: asset) {
                timecodes.append((camera, timecode))
            }
        }
        
        guard !timecodes.isEmpty else {
            throw MultiCamError.noTimecodeFound
        }
        
        let referenceTime = timecodes.first?.1 ?? .zero
        
        for (index, (camera, timecode)) in timecodes.enumerated() {
            let offset = CMTimeSubtract(referenceTime, timecode)
            
            if let cameraIndex = cameras.firstIndex(where: { $0.id == camera.id }) {
                await MainActor.run {
                    cameras[cameraIndex].syncOffset = offset
                    syncProgress = Double(index + 1) / Double(timecodes.count)
                }
            }
        }
    }
    
    private func syncByAudio() async throws {
        guard cameras.count > 1,
              let referenceCamera = cameras.first,
              let referenceAudio = referenceCamera.audioTrack else {
            throw MultiCamError.noAudioTrack
        }
        
        let referenceWaveform = try await extractAudioFingerprint(from: referenceAudio)
        
        for (index, camera) in cameras.enumerated().dropFirst() {
            guard let audioTrack = camera.audioTrack else { continue }
            
            let waveform = try await extractAudioFingerprint(from: audioTrack)
            let offset = calculateAudioOffset(reference: referenceWaveform, compare: waveform)
            
            if let cameraIndex = cameras.firstIndex(where: { $0.id == camera.id }) {
                await MainActor.run {
                    cameras[cameraIndex].syncOffset = offset
                    syncProgress = Double(index + 1) / Double(cameras.count)
                }
            }
        }
    }
    
    private func syncByClapperboard() async throws {
        for (index, camera) in cameras.enumerated() {
            guard let asset = camera.asset else { continue }
            
            let clapTime = try await detectClapperboard(in: asset)
            
            if let cameraIndex = cameras.firstIndex(where: { $0.id == camera.id }) {
                await MainActor.run {
                    cameras[cameraIndex].syncOffset = clapTime
                    syncProgress = Double(index + 1) / Double(cameras.count)
                }
            }
        }
        
        let referenceClap = cameras.first?.syncOffset ?? .zero
        
        for (index, camera) in cameras.enumerated() {
            let relativeOffset = CMTimeSubtract(camera.syncOffset, referenceClap)
            
            await MainActor.run {
                cameras[index].syncOffset = relativeOffset
            }
        }
    }
    
    private func syncByContent() async throws {
        guard cameras.count > 1 else { return }
        
        var contentSignatures: [(CameraAngle, [ContentFeature])] = []
        
        for camera in cameras {
            if let asset = camera.asset {
                let features = try await extractContentFeatures(from: asset)
                contentSignatures.append((camera, features))
            }
        }
        
        guard !contentSignatures.isEmpty else {
            throw MultiCamError.contentAnalysisFailed
        }
        
        let referenceFeatures = contentSignatures.first?.1 ?? []
        
        for (index, (camera, features)) in contentSignatures.enumerated() {
            let offset = matchContentFeatures(reference: referenceFeatures, compare: features)
            
            if let cameraIndex = cameras.firstIndex(where: { $0.id == camera.id }) {
                await MainActor.run {
                    cameras[cameraIndex].syncOffset = offset
                    syncProgress = Double(index + 1) / Double(contentSignatures.count)
                }
            }
        }
    }
    
    private func syncByMetadata() async throws {
        guard cameras.count > 1 else { return }
        
        var recordingTimes: [(CameraAngle, Date)] = []
        
        for camera in cameras {
            if let asset = camera.asset,
               let recordingDate = try await extractRecordingDate(from: asset) {
                recordingTimes.append((camera, recordingDate))
            }
        }
        
        guard !recordingTimes.isEmpty else {
            throw MultiCamError.noMetadataFound
        }
        
        let sortedTimes = recordingTimes.sorted { (a, b) in a.1 < b.1 }
        let referenceTime = sortedTimes.first?.1 ?? Date()
        
        for (camera, recordingDate) in sortedTimes {
            let offset = recordingDate.timeIntervalSince(referenceTime)
            let cmOffset = CMTime(seconds: offset, preferredTimescale: 600)
            
            if let cameraIndex = cameras.firstIndex(where: { $0.id == camera.id }) {
                await MainActor.run {
                    cameras[cameraIndex].syncOffset = cmOffset
                }
            }
        }
    }
    
    // MARK: - Analysis Methods
    
    private func extractMetadata(from asset: AVAsset) async throws -> CameraAngle.CameraMetadata {
        let metadata = CameraAngle.CameraMetadata()
        
        return metadata
    }
    
    private func extractTimecode(from asset: AVAsset) async throws -> CMTime? {
        return nil
    }
    
    private func extractAudioFingerprint(from track: AVAssetTrack) async throws -> [Float] {
        return []
    }
    
    private func calculateAudioOffset(reference: [Float], compare: [Float]) -> CMTime {
        return .zero
    }
    
    private func detectClapperboard(in asset: AVAsset) async throws -> CMTime {
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw MultiCamError.noVideoTrack
        }
        
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: videoTrack,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
        )
        
        reader.add(output)
        reader.startReading()
        
        var clapTime: CMTime = .zero
        let request = VNDetectRectanglesRequest()
        
        while let sampleBuffer = output.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
            try handler.perform([request])
            
            if let results = request.results,
               !results.isEmpty {
                clapTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                break
            }
        }
        
        reader.cancelReading()
        return clapTime
    }
    
    private struct ContentFeature {
        let timestamp: CMTime
        let descriptor: [Float]
    }
    
    private func extractContentFeatures(from asset: AVAsset) async throws -> [ContentFeature] {
        return []
    }
    
    private func matchContentFeatures(reference: [ContentFeature], compare: [ContentFeature]) -> CMTime {
        return .zero
    }
    
    private func extractRecordingDate(from asset: AVAsset) async throws -> Date? {
        return nil
    }
    
    // MARK: - Multi-Camera Composition
    
    public func createMultiCamComposition() async throws -> AVComposition {
        let composition = AVMutableComposition()
        
        for camera in cameras where camera.isEnabled {
            guard let asset = camera.asset,
                  let videoTrack = camera.videoTrack else { continue }
            
            let compositionTrack = composition.addMutableTrack(
                withMediaType: .video,
                preferredTrackID: kCMPersistentTrackID_Invalid
            )
            
            let startTime = camera.syncOffset
            let timeRange = CMTimeRange(
                start: .zero,
                duration: asset.duration
            )
            
            try compositionTrack?.insertTimeRange(
                timeRange,
                of: videoTrack,
                at: startTime
            )
            
            if let audioTrack = camera.audioTrack {
                let audioCompositionTrack = composition.addMutableTrack(
                    withMediaType: .audio,
                    preferredTrackID: kCMPersistentTrackID_Invalid
                )
                
                try audioCompositionTrack?.insertTimeRange(
                    timeRange,
                    of: audioTrack,
                    at: startTime
                )
            }
        }
        
        return composition
    }
    
    // MARK: - Angle Switching
    
    public func switchToCamera(_ camera: CameraAngle, at time: CMTime) {
        selectedCamera = camera
        currentTime = time
        
        logger.info("Switched to camera: \(camera.name) at \(time.seconds)s")
    }
    
    public func createAngleCuts() -> [AngleCut] {
        var cuts: [AngleCut] = []
        
        return cuts
    }
    
    public struct AngleCut {
        public let startTime: CMTime
        public let endTime: CMTime
        public let camera: CameraAngle
        public let transition: TransitionType
        
        public enum TransitionType {
            case cut
            case dissolve(duration: CMTime)
            case wipe(direction: WipeDirection)
            case fade(color: CGColor)
            
            public enum WipeDirection {
                case left, right, up, down
            }
        }
    }
    
    // MARK: - Color Matching
    
    public func matchColors(to reference: CameraAngle) async {
        guard let referenceAsset = reference.asset else { return }
        
        let referenceColorProfile = await analyzeColorProfile(of: referenceAsset)
        
        for (index, camera) in cameras.enumerated() where camera.id != reference.id {
            guard let asset = camera.asset else { continue }
            
            let cameraColorProfile = await analyzeColorProfile(of: asset)
            let correction = calculateColorCorrection(
                from: cameraColorProfile,
                to: referenceColorProfile
            )
            
            await MainActor.run {
                cameras[index].colorCorrection = correction
            }
        }
        
        logger.info("Color matched all cameras to reference: \(reference.name)")
    }
    
    private func analyzeColorProfile(of asset: AVAsset) async -> ColorProfile {
        return ColorProfile()
    }
    
    private func calculateColorCorrection(
        from source: ColorProfile,
        to target: ColorProfile
    ) -> CameraAngle.ColorCorrection {
        return CameraAngle.ColorCorrection()
    }
    
    private struct ColorProfile {
        var averageRGB: (r: Float, g: Float, b: Float) = (0, 0, 0)
        var histogram: [Float] = []
        var whitePoint: CGPoint = .zero
        var blackPoint: CGPoint = .zero
    }
    
    // MARK: - Export
    
    public func exportMultiCamEdit(
        to outputURL: URL,
        preset: String
    ) async throws {
        let composition = try await createMultiCamComposition()
        
        guard let exportSession = AVAssetExportSession(
            asset: composition,
            presetName: preset
        ) else {
            throw MultiCamError.exportFailed
        }
        
        exportSession.outputURL = outputURL
        exportSession.outputFileType = .mov
        
        await exportSession.export()
        
        if exportSession.status == .failed {
            throw exportSession.error ?? MultiCamError.exportFailed
        }
        
        logger.info("Multi-cam edit exported to: \(outputURL.lastPathComponent)")
    }
    
    // MARK: - Setup
    
    private func setupBindings() {
        $cameras
            .sink { [weak self] cameras in
                self?.updateDuration()
            }
            .store(in: &cancellables)
    }
    
    private func updateDuration() {
        var maxDuration: CMTime = .zero
        
        for camera in cameras {
            if let asset = camera.asset {
                let adjustedDuration = CMTimeAdd(asset.duration, camera.syncOffset)
                if CMTimeCompare(adjustedDuration, maxDuration) > 0 {
                    maxDuration = adjustedDuration
                }
            }
        }
        
        duration = maxDuration
    }
}

// MARK: - Multi-Cam Errors

public enum MultiCamError: LocalizedError {
    case noVideoTrack
    case noAudioTrack
    case noTimecodeFound
    case noMetadataFound
    case syncFailed
    case contentAnalysisFailed
    case exportFailed
    
    public var errorDescription: String? {
        switch self {
        case .noVideoTrack:
            return "No video track found in file"
        case .noAudioTrack:
            return "No audio track found for synchronization"
        case .noTimecodeFound:
            return "No timecode found in cameras"
        case .noMetadataFound:
            return "No metadata found for synchronization"
        case .syncFailed:
            return "Failed to synchronize cameras"
        case .contentAnalysisFailed:
            return "Content analysis failed"
        case .exportFailed:
            return "Failed to export multi-cam edit"
        }
    }
}

// MARK: - Multi-Cam View

public struct MultiCamEditorView: View {
    @StateObject private var editor = MultiCamEditor()
    @State private var showAddCamera = false
    @State private var draggedCamera: CameraAngle?
    
    public var body: some View {
        HSplitView {
            cameraList
                .frame(minWidth: 200, maxWidth: 300)
            
            VStack(spacing: 0) {
                previewArea
                timeline
                    .frame(height: 200)
            }
            
            inspector
                .frame(width: 300)
        }
        .toolbar {
            ToolbarItemGroup {
                Button(action: { showAddCamera = true }) {
                    Label("Add Camera", systemImage: "plus.circle")
                }
                
                Picker("Sync Mode", selection: $editor.syncMode) {
                    ForEach(MultiCamEditor.SyncMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .frame(width: 150)
                
                Button(action: syncCameras) {
                    Label("Sync", systemImage: "arrow.triangle.2.circlepath")
                }
                .disabled(editor.cameras.count < 2)
                
                Picker("Preview", selection: $editor.previewMode) {
                    ForEach(MultiCamEditor.PreviewMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
            }
        }
        .fileImporter(
            isPresented: $showAddCamera,
            allowedContentTypes: [.movie],
            allowsMultipleSelection: true
        ) { result in
            handleFileImport(result)
        }
    }
    
    private var cameraList: some View {
        List(selection: $editor.selectedCamera) {
            ForEach(editor.cameras) { camera in
                CameraRow(camera: camera)
                    .tag(camera)
            }
            .onMove(perform: moveCameras)
            .onDelete(perform: deleteCameras)
        }
        .listStyle(SidebarListStyle())
    }
    
    private var previewArea: some View {
        ZStack {
            Color.black
            
            switch editor.previewMode {
            case .single:
                SingleCameraView(camera: editor.selectedCamera)
            case .dual:
                DualCameraView(cameras: Array(editor.cameras.prefix(2)))
            case .quad:
                QuadCameraView(cameras: Array(editor.cameras.prefix(4)))
            case .grid:
                GridCameraView(cameras: editor.cameras)
            case .pip:
                PIPCameraView(main: editor.cameras.first, pip: editor.selectedCamera)
            case .comparison:
                ComparisonView(cameras: Array(editor.cameras.prefix(2)))
            }
        }
    }
    
    private var timeline: some View {
        MultiCamTimeline(
            cameras: editor.cameras,
            currentTime: $editor.currentTime,
            duration: editor.duration
        )
    }
    
    private var inspector: some View {
        ScrollView {
            if let camera = editor.selectedCamera {
                CameraInspector(camera: camera)
            }
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private func syncCameras() {
        Task {
            do {
                try await editor.syncCameras()
            } catch {
                print("Sync failed: \(error)")
            }
        }
    }
    
    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            Task {
                for url in urls {
                    try await editor.addCamera(from: url)
                }
            }
        case .failure(let error):
            print("Import failed: \(error)")
        }
    }
    
    private func moveCameras(from source: IndexSet, to destination: Int) {
        editor.cameras.move(fromOffsets: source, toOffset: destination)
    }
    
    private func deleteCameras(at offsets: IndexSet) {
        editor.cameras.remove(atOffsets: offsets)
    }
}

// MARK: - Supporting Views

struct CameraRow: View {
    let camera: MultiCamEditor.CameraAngle
    
    public var body: some View {
        HStack {
            Image(systemName: iconForAngle(camera.angle))
                .foregroundColor(.secondary)
            
            VStack(alignment: .leading) {
                Text(camera.name)
                    .font(.system(.body, design: .rounded))
                
                if let metadata = camera.metadata {
                    Text("\(metadata.resolution?.width ?? 0)×\(metadata.resolution?.height ?? 0)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            Toggle("", isOn: .constant(camera.isEnabled))
                .labelsHidden()
        }
        .padding(.vertical, 4)
    }
    
    private func iconForAngle(_ angle: MultiCamEditor.CameraAngle.AngleType) -> String {
        switch angle {
        case .main: return "camera"
        case .wide: return "camera.metering.matrix"
        case .medium: return "camera.metering.center.weighted.average"
        case .closeUp: return "camera.metering.spot"
        case .overhead: return "camera.metering.partial"
        case .side: return "camera.metering.center.weighted"
        case .reaction: return "person.crop.rectangle"
        case .bRoll: return "film"
        case .custom: return "camera.on.rectangle"
        }
    }
}

struct SingleCameraView: View {
    let camera: MultiCamEditor.CameraAngle?
    
    public var body: some View {
        if let camera = camera {
            MultiCamVideoPlayerView(url: camera.url)
        } else {
            Text("No Camera Selected")
                .foregroundColor(.secondary)
        }
    }
}

struct DualCameraView: View {
    let cameras: [MultiCamEditor.CameraAngle]
    
    public var body: some View {
        HStack(spacing: 2) {
            ForEach(cameras.prefix(2)) { camera in
                MultiCamVideoPlayerView(url: camera.url)
            }
        }
    }
}

struct QuadCameraView: View {
    let cameras: [MultiCamEditor.CameraAngle]
    
    public var body: some View {
        VStack(spacing: 2) {
            HStack(spacing: 2) {
                ForEach(cameras.prefix(2)) { camera in
                    MultiCamVideoPlayerView(url: camera.url)
                }
            }
            HStack(spacing: 2) {
                ForEach(cameras.dropFirst(2).prefix(2)) { camera in
                    MultiCamVideoPlayerView(url: camera.url)
                }
            }
        }
    }
}

struct GridCameraView: View {
    let cameras: [MultiCamEditor.CameraAngle]
    
    public var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 200))], spacing: 2) {
            ForEach(cameras) { camera in
                MultiCamVideoPlayerView(url: camera.url)
                    .aspectRatio(16/9, contentMode: .fit)
            }
        }
    }
}

struct PIPCameraView: View {
    let main: MultiCamEditor.CameraAngle?
    let pip: MultiCamEditor.CameraAngle?
    
    public var body: some View {
        ZStack {
            if let main = main {
                MultiCamVideoPlayerView(url: main.url)
            }
            
            if let pip = pip, pip.id != main?.id {
                MultiCamVideoPlayerView(url: pip.url)
                    .frame(width: 200, height: 112)
                    .overlay(
                        RoundedRectangle(cornerRadius: 4)
                            .stroke(Color.white, lineWidth: 2)
                    )
                    .position(x: 100, y: 60)
            }
        }
    }
}

struct ComparisonView: View {
    let cameras: [MultiCamEditor.CameraAngle]
    @State private var dividerPosition: CGFloat = 0.5
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                if cameras.count >= 1 {
                    MultiCamVideoPlayerView(url: cameras[0].url)
                }
                
                if cameras.count >= 2 {
                    MultiCamVideoPlayerView(url: cameras[1].url)
                        .mask(
                            Rectangle()
                                .frame(width: geometry.size.width * dividerPosition)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        )
                }
                
                Rectangle()
                    .fill(Color.white)
                    .frame(width: 2)
                    .position(x: geometry.size.width * dividerPosition, y: geometry.size.height / 2)
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                dividerPosition = value.location.x / geometry.size.width
                            }
                    )
            }
        }
    }
}

struct MultiCamVideoPlayerView: View {
    let url: URL
    
    public var body: some View {
        Color.gray
            .overlay(
                Text(url.lastPathComponent)
                    .foregroundColor(.white)
            )
    }
}

struct MultiCamTimeline: View {
    let cameras: [MultiCamEditor.CameraAngle]
    @Binding var currentTime: CMTime
    let duration: CMTime
    
    public var body: some View {
        ScrollView(.vertical) {
            VStack(spacing: 2) {
                ForEach(cameras) { camera in
                    CameraTimelineTrack(camera: camera, duration: duration)
                        .frame(height: 40)
                }
            }
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct CameraTimelineTrack: View {
    let camera: MultiCamEditor.CameraAngle
    let duration: CMTime
    
    public var body: some View {
        HStack(spacing: 0) {
            Text(camera.name)
                .font(.caption)
                .frame(width: 100, alignment: .leading)
                .padding(.horizontal, 8)
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.blue.opacity(0.3))
                    
                    if camera.syncOffset != .zero {
                        let offset = CGFloat(camera.syncOffset.seconds / duration.seconds) * geometry.size.width
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.blue.opacity(0.6))
                            .offset(x: offset)
                    }
                }
            }
        }
    }
}

struct CameraInspector: View {
    let camera: MultiCamEditor.CameraAngle
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(camera.name)
                .font(.headline)
                .padding(.horizontal)
            
            if let metadata = camera.metadata {
                MetadataSection(metadata: metadata)
            }
            
            ColorCorrectionSection(correction: camera.colorCorrection)
        }
        .padding(.vertical)
    }
}

struct MetadataSection: View {
    let metadata: MultiCamEditor.CameraAngle.CameraMetadata
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Metadata")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            if let model = metadata.cameraModel {
                LabeledContent("Camera", value: model)
            }
            
            if let lens = metadata.lensInfo {
                LabeledContent("Lens", value: lens)
            }
        }
        .padding(.horizontal)
    }
}

struct ColorCorrectionSection: View {
    let correction: MultiCamEditor.CameraAngle.ColorCorrection
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Color Correction")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            LabeledContent("Exposure", value: String(format: "%.2f", correction.exposure))
            LabeledContent("Contrast", value: String(format: "%.2f", correction.contrast))
            LabeledContent("Saturation", value: String(format: "%.2f", correction.saturation))
        }
        .padding(.horizontal)
    }
}

// MARK: - Logger

import os

