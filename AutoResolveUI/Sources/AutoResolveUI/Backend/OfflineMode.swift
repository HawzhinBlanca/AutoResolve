// AUTORESOLVE V3.0 - OFFLINE MODE & GRACEFUL DEGRADATION
// Fallback functionality when backend is unavailable

import Foundation
import SwiftUI
import Combine
import AVFoundation

/// Manages offline functionality and caching
public class OfflineManager: ObservableObject {
    public static let shared = OfflineManager()
    
    @Published public var isOfflineMode = false
    @Published public var cachedProjects: [CachedProject] = []
    @Published public var cachedMedia: [CachedMedia] = []
    @Published public var offlineCapabilities: Set<OfflineCapability> = []
    
    private let cacheDirectory: URL
    private let userDefaults = UserDefaults.standard
    private var cancellables = Set<AnyCancellable>()
    
    private init() {
        // Setup cache directory
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        cacheDirectory = documentsPath.appendingPathComponent("AutoResolveCache")
        
        // Create cache directory if needed
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        
        // Load cached data
        loadCachedData()
        
        // Monitor connection state
        setupConnectionMonitoring()
        
        // Determine offline capabilities
        updateOfflineCapabilities()
    }
    
    // MARK: - Connection Monitoring
    
    private func setupConnectionMonitoring() {
        // ConnectionManager monitoring disabled temporarily
        // TODO: Integrate with actual ConnectionManager when ready
        isOfflineMode = false
    }
    
    // MARK: - Offline Mode Management
    
    private func enableOfflineMode() {
        print("[OfflineManager] Enabling offline mode")
        updateOfflineCapabilities()
        loadCachedData()
    }
    
    private func updateOfflineCapabilities() {
        var capabilities: Set<OfflineCapability> = []
        
        // Basic editing always available
        capabilities.insert(.basicEditing)
        capabilities.insert(.localFileAccess)
        capabilities.insert(.projectManagement)
        
        // Check for cached models
        if hasLocalModels() {
            capabilities.insert(.basicAnalysis)
        }
        
        // Check for cached projects
        if !cachedProjects.isEmpty {
            capabilities.insert(.cachedProjectAccess)
        }
        
        offlineCapabilities = capabilities
    }
    
    // MARK: - Caching
    
    public func cacheProject(_ project: Project) {
        let cached = CachedProject(
            id: project.id,
            name: project.name,
            lastModified: Date(),
            thumbnailData: nil, // TODO: Generate thumbnail
            timelineData: encodeTimeline(project)
        )
        
        cachedProjects.append(cached)
        saveCachedProjects()
    }
    
    public func cacheMedia(_ url: URL) {
        guard let data = try? Data(contentsOf: url) else { return }
        
        let cached = CachedMedia(
            id: UUID().uuidString,
            name: url.lastPathComponent,
            type: url.pathExtension,
            data: data,
            metadata: extractMetadata(from: url)
        )
        
        cachedMedia.append(cached)
        saveCachedMedia()
    }
    
    private func loadCachedData() {
        // Load projects
        if let data = userDefaults.data(forKey: "cachedProjects"),
           let projects = try? JSONDecoder().decode([CachedProject].self, from: data) {
            cachedProjects = projects
        }
        
        // Load media references (not actual data to save memory)
        if let data = userDefaults.data(forKey: "cachedMediaRefs"),
           let refs = try? JSONDecoder().decode([CachedMediaReference].self, from: data) {
            // Load media on demand
            cachedMedia = refs.compactMap { loadMediaFromDisk($0) }
        }
    }
    
    private func saveCachedProjects() {
        if let data = try? JSONEncoder().encode(cachedProjects) {
            userDefaults.set(data, forKey: "cachedProjects")
        }
    }
    
    private func saveCachedMedia() {
        // Save only references to disk
        let refs = cachedMedia.map { CachedMediaReference(id: $0.id, path: saveMediaToDisk($0)) }
        if let data = try? JSONEncoder().encode(refs) {
            userDefaults.set(data, forKey: "cachedMediaRefs")
        }
    }
    
    // MARK: - Sync
    
    private func syncWithBackend() {
        print("[OfflineManager] Syncing with backend")
        // TODO: Upload local changes
        // TODO: Download remote changes
    }
    
    // MARK: - Offline Operations
    
    public func performOfflineAnalysis(_ url: URL) -> AnyPublisher<AnalysisResult, Error> {
        Future { promise in
            DispatchQueue.global().async {
                // Basic analysis without backend
                let asset = AVAsset(url: url)
                
                Task {
                    do {
                        let duration = try await asset.load(.duration)
                        let tracks = try await asset.load(.tracks)
                        
                        let result = AnalysisResult(
                            duration: duration.seconds,
                            hasVideo: tracks.contains { $0.mediaType == .video },
                            hasAudio: tracks.contains { $0.mediaType == .audio },
                            silenceRegions: [], // Can't detect without backend
                            sceneChanges: [],   // Can't detect without backend
                            confidence: 0.3     // Low confidence for offline
                        )
                        
                        promise(.success(result))
                    } catch {
                        promise(.failure(error))
                    }
                }
            }
        }
        .eraseToAnyPublisher()
    }
    
    // MARK: - Helpers
    
    private func hasLocalModels() -> Bool {
        // Check if we have any local ML models
        let modelsPath = Bundle.main.url(forResource: "Models", withExtension: nil)
        return modelsPath != nil
    }
    
    private func encodeTimeline(_ project: Project) -> Data? {
        // Create encodable representation
        let data = [
            "id": project.id.uuidString,
            "name": project.name,
            "createdAt": project.createdAt.timeIntervalSince1970,
            "modifiedAt": project.modifiedAt.timeIntervalSince1970
        ] as [String: Any]
        
        return try? JSONSerialization.data(withJSONObject: data)
    }
    
    private func extractMetadata(from url: URL) -> MediaMetadata {
        // Extract basic metadata
        let asset = AVAsset(url: url)
        return MediaMetadata(
            duration: asset.duration.seconds,
            fileSize: (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0,
            format: url.pathExtension
        )
    }
    
    private func saveMediaToDisk(_ media: CachedMedia) -> String {
        let path = cacheDirectory.appendingPathComponent("\(media.id).\(media.type)")
        try? media.data.write(to: path)
        return path.path
    }
    
    private func loadMediaFromDisk(_ ref: CachedMediaReference) -> CachedMedia? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: ref.path)) else { return nil }
        let url = URL(fileURLWithPath: ref.path)
        return CachedMedia(
            id: ref.id,
            name: url.lastPathComponent,
            type: url.pathExtension,
            data: data,
            metadata: extractMetadata(from: url)
        )
    }
}

// MARK: - Supporting Types

public enum OfflineCapability {
    case basicEditing
    case basicAnalysis
    case localFileAccess
    case cachedProjectAccess
    case projectManagement
}

public struct CachedProject: Codable, Identifiable {
    public let id: UUID
    public let name: String
    public let lastModified: Date
    public let thumbnailData: Data?
    public let timelineData: Data?
}

public struct CachedMedia: Identifiable {
    public let id: String
    public let name: String
    public let type: String
    public let data: Data
    public let metadata: MediaMetadata
}

public struct CachedMediaReference: Codable {
    public let id: String
    let path: String
}

public struct MediaMetadata: Codable {
    public let duration: TimeInterval
    public let fileSize: Int64
    public let format: String
}

public struct AnalysisResult {
    public let duration: TimeInterval
    public let hasVideo: Bool
    public let hasAudio: Bool
    public let silenceRegions: [TimeRange]
    public let sceneChanges: [TimeInterval]
    public let confidence: Double
}


// MARK: - Offline Mode UI

public struct OfflineModeIndicator: View {
    @ObservedObject private var offlineManager = OfflineManager.shared
    @State private var showDetails = false
    
    public var body: some View {
        if offlineManager.isOfflineMode {
            HStack(spacing: 8) {
                Image(systemName: "wifi.slash")
                    .font(.system(size: 12))
                    .foregroundColor(.orange)
                
                Text("Offline Mode")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.orange)
                
                Button(action: { showDetails.toggle() }) {
                    Image(systemName: "info.circle")
                        .font(.system(size: 12))
                        .foregroundColor(.orange.opacity(0.8))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.orange.opacity(0.15))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.orange.opacity(0.3), lineWidth: 1)
                    )
            )
            .popover(isPresented: $showDetails) {
                OfflineDetailsView()
                    .frame(width: 300, height: 200)
            }
        }
    }
}

struct OfflineDetailsView: View {
    @ObservedObject private var offlineManager = OfflineManager.shared
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Offline Mode Active")
                .font(.system(size: 16, weight: .semibold))
            
            Text("Available features:")
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.secondary)
            
            ForEach(Array(offlineManager.offlineCapabilities), id: \.self) { capability in
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 10))
                        .foregroundColor(.green)
                    
                    Text(capability.description)
                        .font(.system(size: 11))
                }
            }
            
            Spacer()
            
            Button(action: { /* TODO: Reconnect */ }) {
                Text("Try Reconnecting")
                    .font(.system(size: 12, weight: .medium))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(6)
            }
            .buttonStyle(.plain)
        }
        .padding()
    }
}

extension OfflineCapability {
    var description: String {
        switch self {
        case .basicEditing: return "Basic editing tools"
        case .basicAnalysis: return "Local media analysis"
        case .localFileAccess: return "Access local files"
        case .cachedProjectAccess: return "Access cached projects"
        case .projectManagement: return "Manage projects"
        }
    }
}
