import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

// MARK: - Professional Media Browser

public struct MediaBrowser: View {
    @StateObject private var mediaManager = MediaManager()
    @State private var selectedMedia: [MediaItem] = []
    @State private var searchText = ""
    @State private var sortOrder: SortOrder = .dateAdded
    @State private var viewMode: ViewMode = .grid
    @State private var showImportSheet = false
    @State private var selectedFolder: MediaFolder?
    @State private var showCreateFolder = false
    @State private var newFolderName = ""
    
    enum ViewMode: String, CaseIterable {
        case grid = "Grid"
        case list = "List"
        case thumbnails = "Thumbnails"
        
        var icon: String {
            switch self {
            case .grid: return "square.grid.2x2"
            case .list: return "list.bullet"
            case .thumbnails: return "rectangle.grid.1x2"
            }
        }
    }
    
    enum SortOrder: String, CaseIterable {
        case name = "Name"
        case dateAdded = "Date Added"
        case dateModified = "Date Modified"
        case duration = "Duration"
        case fileSize = "File Size"
        case type = "Type"
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header with search and controls
            MediaBrowserHeader(
                searchText: $searchText,
                sortOrder: $sortOrder,
                viewMode: $viewMode,
                onImport: { showImportSheet = true },
                onCreateFolder: { showCreateFolder = true }
            )
            
            Divider()
            
            // Main content area
            HStack(spacing: 0) {
                // Sidebar with folders
                MediaFolderSidebar(
                    folders: mediaManager.folders,
                    selectedFolder: $selectedFolder,
                    onCreateFolder: { showCreateFolder = true }
                )
                .frame(width: 200)
                
                Divider()
                
                // Media content area
                MediaContentView(
                    mediaItems: filteredMediaItems,
                    selectedMedia: $selectedMedia,
                    viewMode: viewMode,
                    onAddToTimeline: addToTimeline
                )
            }
            
            if !selectedMedia.isEmpty {
                Divider()
                
                // Selection info and actions
                MediaSelectionBar(
                    selectedItems: selectedMedia,
                    onClearSelection: { selectedMedia.removeAll() },
                    onAddToTimeline: addSelectedToTimeline,
                    onDelete: deleteSelectedMedia
                )
            }
        }
        .frame(width: 600, height: 500)
        .background(Color(NSColor.controlBackgroundColor))
        .sheet(isPresented: $showImportSheet) {
            MediaImportSheet(
                mediaManager: mediaManager,
                targetFolder: selectedFolder
            )
        }
        .alert("Create Folder", isPresented: $showCreateFolder) {
            TextField("Folder name", text: $newFolderName)
            Button("Create") {
                createFolder()
            }
            Button("Cancel", role: .cancel) {}
        }
        .onAppear {
            mediaManager.loadMedia()
        }
    }
    
    private var filteredMediaItems: [MediaItem] {
        var items = selectedFolder?.items ?? mediaManager.allMedia
        
        // Apply search filter
        if !searchText.isEmpty {
            items = items.filter { item in
                item.name.localizedCaseInsensitiveContains(searchText) ||
                item.metadata.keywords.contains { $0.localizedCaseInsensitiveContains(searchText) }
            }
        }
        
        // Apply sort order
        switch sortOrder {
        case .name:
            items.sort { $0.name < $1.name }
        case .dateAdded:
            items.sort { $0.dateAdded > $1.dateAdded }
        case .dateModified:
            items.sort { $0.dateModified > $1.dateModified }
        case .duration:
            items.sort { $0.duration > $1.duration }
        case .fileSize:
            items.sort { $0.fileSize > $1.fileSize }
        case .type:
            items.sort { $0.type.rawValue < $1.type.rawValue }
        }
        
        return items
    }
    
    private func createFolder() {
        guard !newFolderName.isEmpty else { return }
        
        let folder = MediaFolder(
            name: newFolderName,
            color: MediaFolder.randomColor()
        )
        mediaManager.createFolder(folder)
        newFolderName = ""
    }
    
    private func addToTimeline(_ item: MediaItem) {
        // Add single item to timeline
        print("Adding \(item.name) to timeline")
    }
    
    private func addSelectedToTimeline() {
        // Add all selected items to timeline
        for item in selectedMedia {
            addToTimeline(item)
        }
        selectedMedia.removeAll()
    }
    
    private func deleteSelectedMedia() {
        mediaManager.deleteMedia(selectedMedia)
        selectedMedia.removeAll()
    }
}

// MARK: - Media Browser Header

struct MediaBrowserHeader: View {
    @Binding var searchText: String
    @Binding var sortOrder: MediaBrowser.SortOrder
    @Binding var viewMode: MediaBrowser.ViewMode
    let onImport: () -> Void
    let onCreateFolder: () -> Void
    
    public var body: some View {
        HStack {
            // Import button
            Button(action: onImport) {
                Label("Import", systemImage: "plus")
            }
            .buttonStyle(.bordered)
            
            // Create folder button
            Button(action: onCreateFolder) {
                Image(systemName: "folder.badge.plus")
            }
            .buttonStyle(.plain)
            
            Spacer()
            
            // Search field
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                
                TextField("Search media...", text: $searchText)
                
                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(NSColor.textBackgroundColor))
            .cornerRadius(6)
            .frame(width: 200)
            
            // Sort order
            Menu {
                Picker("Sort", selection: $sortOrder) {
                    ForEach(MediaBrowser.SortOrder.allCases, id: \.self) { order in
                        Text(order.rawValue).tag(order)
                    }
                }
            } label: {
                Label("Sort", systemImage: "arrow.up.arrow.down")
            }
            .menuStyle(.borderlessButton)
            
            // View mode
            Picker("View", selection: $viewMode) {
                ForEach(MediaBrowser.ViewMode.allCases, id: \.self) { mode in
                    Label(mode.rawValue, systemImage: mode.icon)
                        .tag(mode)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .frame(width: 150)
        }
        .padding()
    }
}

// MARK: - Media Folder Sidebar

struct MediaFolderSidebar: View {
    let folders: [MediaFolder]
    @Binding var selectedFolder: MediaFolder?
    let onCreateFolder: () -> Void
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Sidebar header
            HStack {
                Text("Media")
                    .font(.headline)
                
                Spacer()
                
                Button(action: onCreateFolder) {
                    Image(systemName: "plus")
                }
                .buttonStyle(.plain)
            }
            .padding()
            
            Divider()
            
            // All Media item
            SidebarItem(
                icon: "photo.on.rectangle.angled",
                title: "All Media",
                count: folders.reduce(0) { $0 + $1.items.count },
                isSelected: selectedFolder == nil
            ) {
                selectedFolder = nil
            }
            
            // Folders
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(folders) { folder in
                        SidebarFolderItem(
                            folder: folder,
                            isSelected: selectedFolder?.id == folder.id
                        ) {
                            selectedFolder = folder
                        }
                    }
                }
            }
            
            Spacer()
        }
        .background(Color(NSColor.windowBackgroundColor))
    }
}

struct SidebarItem: View {
    let icon: String
    let title: String
    let count: Int
    let isSelected: Bool
    let onSelect: () -> Void
    
    public var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(isSelected ? .accentColor : .secondary)
                .frame(width: 20)
            
            Text(title)
                .fontWeight(isSelected ? .medium : .regular)
            
            Spacer()
            
            Text("\(count)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .onTapGesture {
            onSelect()
        }
    }
}

struct SidebarFolderItem: View {
    let folder: MediaFolder
    let isSelected: Bool
    let onSelect: () -> Void
    
    public var body: some View {
        HStack {
            Circle()
                .fill(folder.color)
                .frame(width: 12, height: 12)
            
            Text(folder.name)
                .fontWeight(isSelected ? .medium : .regular)
                .lineLimit(1)
            
            Spacer()
            
            Text("\(folder.items.count)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .onTapGesture {
            onSelect()
        }
    }
}

// MARK: - Media Content View

struct MediaContentView: View {
    let mediaItems: [MediaItem]
    @Binding var selectedMedia: [MediaItem]
    let viewMode: MediaBrowser.ViewMode
    let onAddToTimeline: (MediaItem) -> Void
    
    public var body: some View {
        if mediaItems.isEmpty {
            EmptyMediaView()
        } else {
            switch viewMode {
            case .grid:
                MediaGridView(
                    items: mediaItems,
                    selectedItems: $selectedMedia,
                    onAddToTimeline: onAddToTimeline
                )
            case .list:
                MediaListView(
                    items: mediaItems,
                    selectedItems: $selectedMedia,
                    onAddToTimeline: onAddToTimeline
                )
            case .thumbnails:
                MediaThumbnailView(
                    items: mediaItems,
                    selectedItems: $selectedMedia,
                    onAddToTimeline: onAddToTimeline
                )
            }
        }
    }
}

// MARK: - Media Grid View

struct MediaGridView: View {
    let items: [MediaItem]
    @Binding var selectedItems: [MediaItem]
    let onAddToTimeline: (MediaItem) -> Void
    
    private let columns = Array(repeating: GridItem(.flexible()), count: 4)
    
    public var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 12) {
                ForEach(items) { item in
                    MediaGridCard(
                        item: item,
                        isSelected: selectedItems.contains { $0.id == item.id },
                        onSelect: { toggleSelection(item) },
                        onAddToTimeline: { onAddToTimeline(item) }
                    )
                }
            }
            .padding()
        }
    }
    
    private func toggleSelection(_ item: MediaItem) {
        if let index = selectedItems.firstIndex(where: { $0.id == item.id }) {
            selectedItems.remove(at: index)
        } else {
            selectedItems.append(item)
        }
    }
}

struct MediaGridCard: View {
    let item: MediaItem
    let isSelected: Bool
    let onSelect: () -> Void
    let onAddToTimeline: () -> Void
    
    public var body: some View {
        VStack(spacing: 8) {
            // Thumbnail
            AsyncImage(url: item.thumbnailURL) { image in
                image
                    .resizable()
                    .aspectRatio(16/9, contentMode: .fill)
            } placeholder: {
                Rectangle()
                    .fill(Color.secondary.opacity(0.3))
                    .overlay(
                        Image(systemName: item.type.icon)
                            .font(.title)
                            .foregroundColor(.secondary)
                    )
                    .aspectRatio(16/9, contentMode: .fill)
            }
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
            )
            .overlay(
                // Duration badge
                Group {
                    if item.duration ?? 0 > 0 {
                        VStack {
                            Spacer()
                            HStack {
                                Spacer()
                                Text(formatDuration(item.duration ?? 0))
                                    .font(.caption)
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 2)
                                    .background(Color.black.opacity(0.7))
                                    .foregroundColor(.white)
                                    .cornerRadius(4)
                                    .padding(4)
                            }
                        }
                    }
                }
            )
            
            // Info
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name)
                    .font(.caption)
                    .fontWeight(.medium)
                    .lineLimit(2)
                
                Text(item.formattedFileSize)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(width: 120)
        .onTapGesture(count: 2) {
            onAddToTimeline()
        }
        .onTapGesture(count: 1) {
            onSelect()
        }
        .contextMenu {
            Button("Add to Timeline") {
                onAddToTimeline()
            }
            
            Button("Reveal in Finder") {
                NSWorkspace.shared.selectFile(item.url.path, inFileViewerRootedAtPath: "")
            }
            
            Divider()
            
            Button("Delete", role: .destructive) {
                // Delete item
            }
        }
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return "\(minutes):\(String(format: "%02d", seconds))"
    }
}

// MARK: - Media List View

struct MediaListView: View {
    let items: [MediaItem]
    @Binding var selectedItems: [MediaItem]
    let onAddToTimeline: (MediaItem) -> Void
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Name")
                    .frame(maxWidth: .infinity, alignment: .leading)
                
                Text("Duration")
                    .frame(width: 80)
                
                Text("Size")
                    .frame(width: 80)
                
                Text("Type")
                    .frame(width: 60)
            }
            .font(.caption.bold())
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Items
//             ScrollView {
//                 LazyVStack(spacing: 0) {
//                     ForEach(items) { item in
//                         MediaListRow(
//                             item: item,
//                             isSelected: selectedItems.contains { $0.id == item.id },
//                             onSelect: { toggleSelection(item) },
//                             onAddToTimeline: { onAddToTimeline(item) }
//                         )
//                     }
//                 }
            }
        }
    }

// MARK: - Media Thumbnail View

struct MediaThumbnailView: View {
    let items: [MediaItem]
    @Binding var selectedItems: [MediaItem]
    let onAddToTimeline: (MediaItem) -> Void
    
    private let columns = Array(repeating: GridItem(.flexible()), count: 6)
    
    public var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 8) {
                ForEach(items) { item in
                    MediaThumbnailCard(
                        item: item,
                        isSelected: selectedItems.contains { $0.id == item.id },
                        onSelect: { toggleSelection(item) },
                        onAddToTimeline: { onAddToTimeline(item) }
                    )
                }
            }
            .padding()
        }
    }
    
    private func toggleSelection(_ item: MediaItem) {
        if let index = selectedItems.firstIndex(where: { $0.id == item.id }) {
            selectedItems.remove(at: index)
        } else {
            selectedItems.append(item)
        }
    }
}

struct MediaThumbnailCard: View {
    let item: MediaItem
    let isSelected: Bool
    let onSelect: () -> Void
    let onAddToTimeline: () -> Void
    
    public var body: some View {
        AsyncImage(url: item.thumbnailURL) { image in
            image
                .resizable()
                .aspectRatio(16/9, contentMode: .fill)
        } placeholder: {
            Rectangle()
                .fill(Color.secondary.opacity(0.3))
                .overlay(
                    Image(systemName: item.type.icon)
                        .font(.title2)
                        .foregroundColor(.secondary)
                )
                .aspectRatio(16/9, contentMode: .fill)
        }
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .overlay(
            RoundedRectangle(cornerRadius: 4)
                .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
        )
        .frame(width: 80)
        .onTapGesture(count: 2) {
            onAddToTimeline()
        }
        .onTapGesture(count: 1) {
            onSelect()
        }
    }
}

// MARK: - Empty Media View

struct EmptyMediaView: View {
    public var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            
            Text("No Media Found")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("Import media files to get started")
                .foregroundColor(.secondary)
            
            Button("Import Media") {
                // Trigger import
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Media Selection Bar

struct MediaSelectionBar: View {
    let selectedItems: [MediaItem]
    let onClearSelection: () -> Void
    let onAddToTimeline: () -> Void
    let onDelete: () -> Void
    
    public var body: some View {
        HStack {
            Text("\(selectedItems.count) item\(selectedItems.count == 1 ? "" : "s") selected")
                .font(.caption)
            
            Spacer()
            
            Button("Add to Timeline") {
                onAddToTimeline()
            }
            .buttonStyle(.bordered)
            
            Button("Delete") {
                onDelete()
            }
            .buttonStyle(.bordered)
            .foregroundColor(.red)
            
            Button("Clear") {
                onClearSelection()
            }
            .buttonStyle(.plain)
        }
        .padding()
        .background(Color(NSColor.windowBackgroundColor))
    }
}

// MARK: - Media Import Sheet

struct MediaImportSheet: View {
    @ObservedObject var mediaManager: MediaManager
    let targetFolder: MediaFolder?
    @Environment(\.dismiss) private var dismiss
    
    @State private var selectedFiles: [URL] = []
    @State private var isImporting = false
    @State private var importProgress: Double = 0
    
    public var body: some View {
        VStack(spacing: 20) {
            Text("Import Media")
                .font(.headline)
            
            if selectedFiles.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    
                    Text("Select media files to import")
                        .foregroundColor(.secondary)
                    
                    Button("Choose Files...") {
                        selectFiles()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    Text("\(selectedFiles.count) file\(selectedFiles.count == 1 ? "" : "s") selected")
                        .font(.caption)
                    
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(selectedFiles, id: \.self) { url in
                                Text(url.lastPathComponent)
                                    .font(.caption)
                            }
                        }
                    }
                    .frame(height: 100)
                    .padding(8)
                    .background(Color(NSColor.textBackgroundColor))
                    .cornerRadius(4)
                    
                    if let folder = targetFolder {
                        Text("Import to: \(folder.name)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                if isImporting {
                    ProgressView(value: importProgress)
                        .progressViewStyle(.linear)
                }
            }
            
            HStack {
                if !selectedFiles.isEmpty {
                    Button("Choose Different Files") {
                        selectedFiles.removeAll()
                    }
                    .buttonStyle(.plain)
                }
                
                Spacer()
                
                Button("Cancel") {
                    dismiss()
                }
                
                Button("Import") {
                    startImport()
                }
                .disabled(selectedFiles.isEmpty || isImporting)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .frame(width: 400, height: 300)
    }
    
    private func selectFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.movie, .audio, .image]
        
        if panel.runModal() == .OK {
            selectedFiles = panel.urls
        }
    }
    
    private func startImport() {
        isImporting = true
        importProgress = 0
        
        Task {
            await mediaManager.importFiles(selectedFiles, to: targetFolder) { progress in
                Task { @MainActor in
                    importProgress = progress
                }
            }
            
            Task { @MainActor in
                dismiss()
            }
        }
    }
}

// MARK: - Media Models

public class MediaManager: ObservableObject {
    @Published var folders: [MediaFolder] = []
    @Published var allMedia: [MediaItem] = []
    
    func loadMedia() {
        // Load media from storage
        loadSampleData()
    }
    
    func createFolder(_ folder: MediaFolder) {
        folders.append(folder)
    }
    
    func deleteMedia(_ items: [MediaItem]) {
        // Remove items from storage
        allMedia.removeAll { item in
            items.contains { $0.id == item.id }
        }
    }
    
    func importFiles(_ urls: [URL], to folder: MediaFolder?, progress: @escaping (Double) -> Void) async {
        for (index, url) in urls.enumerated() {
            // Create media item
            let item = await createMediaItem(from: url)
            
            await MainActor.run {
                allMedia.append(item)
                if let folder = folder {
                    if let folderIndex = folders.firstIndex(where: { $0.id == folder.id }) {
                        folders[folderIndex].items.append(item)
                    }
                }
            }
            
            progress(Double(index + 1) / Double(urls.count))
            
            // Simulate processing time
            try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
        }
    }
    
    private func createMediaItem(from url: URL) async -> MediaItem {
        let asset = AVAsset(url: url)
        let duration = (try? await asset.load(.duration).seconds) ?? 0
        
        return MediaItem(
            name: url.deletingPathExtension().lastPathComponent,
            url: url,
            type: AutoResolveUILib.MediaType.from(url: url),
            duration: duration,
            fileSize: fileSize(of: url),
            dateAdded: Date(),
            dateModified: modificationDate(of: url) ?? Date()
        )
    }
    
    private func fileSize(of url: URL) -> Int64 {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
    }
    
    private func modificationDate(of url: URL) -> Date? {
        try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date
    }
    
    private func loadSampleData() {
        // Load some sample data for demo
        folders = [
            MediaFolder(name: "B-Roll", color: .blue),
            MediaFolder(name: "Interviews", color: .green),
            MediaFolder(name: "Music", color: .orange),
            MediaFolder(name: "Graphics", color: .purple)
        ]
    }
}

public struct MediaItem: Identifiable, Hashable {
    public let id = UUID()
    public let name: String
    public let url: URL
    public let type: AutoResolveUILib.MediaType
    public let duration: TimeInterval
    public let fileSize: Int64
    public let dateAdded: Date
    public let dateModified: Date
    public var metadata = MediaBrowserMetadata()
    
    public var thumbnailURL: URL? {
        // Generate or retrieve thumbnail URL
        nil
    }
    
    public var formattedFileSize: String {
        ByteCountFormatter().string(fromByteCount: fileSize)
    }
}

public struct MediaFolder: Identifiable, Hashable {
    public let id = UUID()
    public let name: String
    public let color: Color
    public var items: [MediaItem] = []
    
    public static func randomColor() -> Color {
        [.red, .blue, .green, .orange, .purple, .pink, .yellow].randomElement() ?? .blue
    }
}

public enum MediaType: String, CaseIterable {
    case video = "video"
    case audio = "audio"
    case image = "image"
    case other = "other"
    
    public var icon: String {
        switch self {
        case .video: return "play.rectangle"
        case .audio: return "waveform"
        case .image: return "photo"
        case .other: return "doc"
        }
    }
    
    public static func from(url: URL) -> MediaType {
        let pathExtension = url.pathExtension.lowercased()
        
        if ["mp4", "mov", "avi", "mkv", "m4v"].contains(pathExtension) {
            return .video
        } else if ["mp3", "wav", "aiff", "m4a", "flac"].contains(pathExtension) {
            return .audio
        } else if ["jpg", "jpeg", "png", "tiff", "bmp", "gif"].contains(pathExtension) {
            return .image
        } else {
            return .other
        }
    }
}

public struct MediaBrowserMetadata: Hashable, Equatable {
    public var keywords: [String] = []
    public var description = ""
    public var rating: Int = 0
    public var colorLabel = ""
    public var duration: TimeInterval? = nil
    public var resolution: CGSize? = nil
    public var frameRate: Double? = nil
}

extension MediaMetadata {
    var resolution: CGSize {
        CGSize(width: 1920, height: 1080) // Default
    }
}
