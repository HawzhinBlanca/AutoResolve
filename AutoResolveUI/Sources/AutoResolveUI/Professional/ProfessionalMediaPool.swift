// AUTORESOLVE V3.0 - PROFESSIONAL MEDIA POOL
// DaVinci Resolve-style media management with bins, smart bins, and metadata

import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

// MARK: - Professional Media Pool
public struct ProfessionalMediaPool: View {
    @EnvironmentObject private var store: UnifiedStore
    @State private var selectedView: MediaView = .thumbnail
    @State private var selectedBin: MediaBin? = nil
    @State private var searchText = ""
    @State private var showingImportMenu = false
    @State private var selectedItems: Set<String> = []
    @State private var sortOrder: SortOrder = .dateAdded
    @State private var showMetadata = true
    
    enum MediaView: String, CaseIterable {
        case thumbnail = "square.grid.2x2"
        case list = "list.bullet"
        case filmstrip = "film"
        
        var tooltip: String {
            switch self {
            case .thumbnail: return "Thumbnail View"
            case .list: return "List View"
            case .filmstrip: return "Filmstrip View"
            }
        }
    }
    
    enum SortOrder: String, CaseIterable {
        case name = "Name"
        case dateAdded = "Date Added"
        case duration = "Duration"
        case fileSize = "File Size"
        case type = "Type"
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Media Pool Header
            MediaPoolHeader(
                selectedView: $selectedView,
                searchText: $searchText,
                showingImportMenu: $showingImportMenu,
                sortOrder: $sortOrder
            )
            .frame(height: 36)
            .background(Color(white: 0.14))
            
            Divider()
            
            // Main Content
            HSplitView {
                // Bins & Smart Bins
                MediaBinsView(selectedBin: $selectedBin)
                    .frame(minWidth: 150, idealWidth: 200, maxWidth: 250)
                
                // Media Items
                MediaItemsView(
                    view: selectedView,
                    selectedItems: $selectedItems,
                    searchText: searchText,
                    sortOrder: sortOrder,
                    selectedBin: selectedBin
                )
                .frame(minWidth: 300)
            }
            
            // Media Info Bar
            if showMetadata {
                MediaInfoBar(selectedItems: selectedItems)
                    .frame(height: 60)
                    .background(Color(white: 0.12))
            }
        }
        .fileImporter(
            isPresented: $showingImportMenu,
            allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie, .image],
            allowsMultipleSelection: true
        ) { result in
            handleImport(result)
        }
    }
    
    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            for url in urls {
                store.importMedia(url: url, toBin: selectedBin)
            }
        case .failure(let error):
            print("Import error: \(error)")
        }
    }
}

// MARK: - Media Pool Header
struct MediaPoolHeader: View {
    @Binding var selectedView: ProfessionalMediaPool.MediaView
    @Binding var searchText: String
    @Binding var showingImportMenu: Bool
    @Binding var sortOrder: ProfessionalMediaPool.SortOrder
    @EnvironmentObject private var store: UnifiedStore
    
    public var body: some View {
        HStack(spacing: 8) {
            // Import Button
            Menu {
                Button("Import Media Files...") {
                    showingImportMenu = true
                }
                .keyboardShortcut("i", modifiers: .command)
                
                Divider()
                
                Button("Import Folder...") {
                    // Import folder
                }
                
                Button("Import from Camera...") {
                    // Camera import
                }
                
                Button("Import EDL/XML/AAF...") {
                    // Timeline import
                }
            } label: {
                Image(systemName: "plus.circle.fill")
                    .font(.system(size: 16))
                    .foregroundColor(.blue)
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
            .help("Import Media")
            
            // Create Bin
            Button(action: createNewBin) {
                Image(systemName: "folder.badge.plus")
                    .font(.system(size: 14))
            }
            .buttonStyle(.plain)
            .foregroundColor(.gray)
            .help("New Bin")
            
            Divider()
                .frame(height: 16)
            
            // Search
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.gray)
                    .font(.system(size: 12))
                
                TextField("Search media...", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 11))
                
                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.gray)
                            .font(.system(size: 10))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background(Color.black.opacity(0.3))
            .cornerRadius(4)
            .frame(maxWidth: 200)
            
            Spacer()
            
            // Sort Order
            Menu {
                ForEach(ProfessionalMediaPool.SortOrder.allCases, id: \.self) { order in
                    Button(order.rawValue) {
                        sortOrder = order
                    }
                }
            } label: {
                HStack(spacing: 2) {
                    Image(systemName: "arrow.up.arrow.down")
                        .font(.system(size: 10))
                    Text(sortOrder.rawValue)
                        .font(.system(size: 10))
                }
            }
            .menuStyle(.borderlessButton)
            .foregroundColor(.gray)
            
            // View Options
            HStack(spacing: 2) {
                ForEach(ProfessionalMediaPool.MediaView.allCases, id: \.self) { view in
                    Button(action: { selectedView = view }) {
                        Image(systemName: view.rawValue)
                            .font(.system(size: 12))
                            .frame(width: 24, height: 24)
                            .foregroundColor(selectedView == view ? .white : .gray)
                            .background(
                                selectedView == view ?
                                Color.blue.opacity(0.3) : Color.clear
                            )
                            .cornerRadius(4)
                    }
                    .buttonStyle(.plain)
                    .help(view.tooltip)
                }
            }
        }
        .padding(.horizontal, 8)
    }
    
    private func createNewBin() {
        store.createMediaBin(name: "New Bin")
    }
}

// MARK: - Media Bins View
struct MediaBinsView: View {
    @Binding var selectedBin: MediaBin?
    @EnvironmentObject private var store: UnifiedStore
    @State private var expandedBins: Set<String> = []
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Master Bin
            BinRow(
                bin: MediaBin.master,
                isSelected: selectedBin == nil,
                isExpanded: false,
                hasChildren: false,
                onSelect: { selectedBin = nil }
            )
            
            Divider()
            
            // Smart Bins Section
            DisclosureGroup(
                isExpanded: .constant(true),
                content: {
                    ForEach(smartBins) { bin in
                        SmartBinRow(bin: bin, isSelected: selectedBin?.id == bin.id)
                            .onTapGesture {
                                selectedBin = bin
                            }
                    }
                },
                label: {
                    Label("Smart Bins", systemImage: "gearshape.2")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.gray)
                }
            )
            .padding(.vertical, 4)
            
            Divider()
            
            // User Bins
            ScrollView {
                VStack(alignment: .leading, spacing: 1) {
                    ForEach(store.mediaBins) { bin in
                        BinRow(
                            bin: bin,
                            isSelected: selectedBin?.id == bin.id,
                            isExpanded: expandedBins.contains(bin.id),
                            hasChildren: !bin.children.isEmpty,
                            onSelect: { selectedBin = bin },
                            onToggleExpand: {
                                if expandedBins.contains(bin.id) {
                                    expandedBins.remove(bin.id)
                                } else {
                                    expandedBins.insert(bin.id)
                                }
                            }
                        )
                    }
                }
            }
        }
        .background(Color(white: 0.13))
    }
    
    private var smartBins: [MediaBin] {
        [
            MediaBin(id: "smart_video", name: "All Video", color: .blue),
            MediaBin(id: "smart_audio", name: "All Audio", color: .green),
            MediaBin(id: "smart_images", name: "All Images", color: .orange),
            MediaBin(id: "smart_recent", name: "Recently Added", color: .purple),
            MediaBin(id: "smart_unused", name: "Unused Media", color: .red)
        ]
    }
}

// MARK: - Bin Row
struct BinRow: View {
    let bin: MediaBin
    let isSelected: Bool
    let isExpanded: Bool
    let hasChildren: Bool
    let onSelect: () -> Void
    var onToggleExpand: (() -> Void)? = nil
    
    public var body: some View {
        HStack(spacing: 4) {
            if hasChildren {
                Button(action: { onToggleExpand?() }) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 8))
                        .frame(width: 12)
                }
                .buttonStyle(.plain)
                .foregroundColor(.gray)
            } else {
                Spacer()
                    .frame(width: 12)
            }
            
            Image(systemName: bin.icon)
                .font(.system(size: 11))
                .foregroundColor(bin.color)
            
            Text(bin.name)
                .font(.system(size: 11))
                .foregroundColor(isSelected ? .white : .gray)
            
            Spacer()
            
            if bin.itemCount > 0 {
                Text("\(bin.itemCount)")
                    .font(.system(size: 9))
                    .foregroundColor(.gray)
                    .padding(.horizontal, 4)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isSelected ? Color.blue.opacity(0.3) : Color.clear)
        .onTapGesture(perform: onSelect)
    }
}

// MARK: - Smart Bin Row
struct SmartBinRow: View {
    let bin: MediaBin
    let isSelected: Bool
    
    public var body: some View {
        HStack(spacing: 4) {
            Image(systemName: bin.icon)
                .font(.system(size: 10))
                .foregroundColor(bin.color)
            
            Text(bin.name)
                .font(.system(size: 10))
                .foregroundColor(isSelected ? .white : .gray)
            
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 3)
        .background(isSelected ? Color.blue.opacity(0.3) : Color.clear)
    }
}

// MARK: - Media Items View
struct MediaItemsView: View {
    let view: ProfessionalMediaPool.MediaView
    @Binding var selectedItems: Set<String>
    let searchText: String
    let sortOrder: ProfessionalMediaPool.SortOrder
    let selectedBin: MediaBin?
    @EnvironmentObject private var store: UnifiedStore
    
    var filteredItems: [MediaPoolItem] {
        var items = store.mediaItems
        
        // Filter by bin
        if let bin = selectedBin {
            items = items.filter { bin.contains($0) }
        }
        
        // Filter by search
        if !searchText.isEmpty {
            items = items.filter {
                $0.name.localizedCaseInsensitiveContains(searchText)
            }
        }
        
        // Sort
        switch sortOrder {
        case .name:
            items.sort { $0.name < $1.name }
        case .dateAdded:
            items.sort(by: { $0.id.uuidString > $1.id.uuidString })
        case .duration:
            items.sort(by: { $0.duration > $1.duration })
        case .fileSize:
            items.sort { $0.fileSize > $1.fileSize }
        case .type:
            items.sort { 
                let type0 = $0.url.pathExtension.lowercased()
                let type1 = $1.url.pathExtension.lowercased()
                return type0 < type1
            }
        }
        
        return items
    }
    
    public var body: some View {
        ScrollView {
            switch view {
            case .thumbnail:
                ThumbnailGridView(items: filteredItems, selectedItems: $selectedItems)
            case .list:
                ListView(items: filteredItems, selectedItems: $selectedItems)
            case .filmstrip:
                FilmstripView(items: filteredItems, selectedItems: $selectedItems)
            }
        }
        .background(Color(white: 0.11))
    }
}

// MARK: - Thumbnail Grid View
struct ThumbnailGridView: View {
    let items: [MediaPoolItem]
    @Binding var selectedItems: Set<String>
    
    public var body: some View {
        LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 120, maximum: 160))],
            spacing: 8
        ) {
            ForEach(items) { item in
                MediaThumbnail(
                    item: item,
                    isSelected: selectedItems.contains(item.id.uuidString)
                )
                .onTapGesture {
                    toggleSelection(item.id.uuidString)
                }
                .onDrag {
                    // Create NSItemProvider with the media URL
                    let provider = NSItemProvider(object: item.url as NSURL)
                    provider.suggestedName = item.name
                    return provider
                }
            }
        }
        .padding(8)
    }
    
    private func toggleSelection(_ id: String) {
        if selectedItems.contains(id) {
            selectedItems.remove(id)
        } else {
            selectedItems.insert(id)
        }
    }
}

// MARK: - Media Thumbnail
struct MediaThumbnail: View {
    let item: MediaPoolItem
    let isSelected: Bool
    
    public var body: some View {
        VStack(spacing: 4) {
            // Thumbnail
            ZStack {
                if let thumbnail = item.thumbnail {
                    thumbnail
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    Rectangle()
                        .fill(Color.gray.opacity(0.2))
                        .overlay(
                            Image(systemName: item.hasVideo ? "video" : "waveform")
                                .font(.largeTitle)
                                .foregroundColor(.gray)
                        )
                }
                
                // Duration overlay
                if item.duration > 0 {
                    VStack {
                        Spacer()
                        HStack {
                            Spacer()
                            Text(formatDuration(item.duration))
                                .font(.system(size: 9))
                                .foregroundColor(.white)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(2)
                        }
                    }
                    .padding(4)
                }
            }
            .frame(height: 90)
            .cornerRadius(4)
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .stroke(isSelected ? Color.orange : Color.clear, lineWidth: 2)
            )
            
            // Name
            Text(item.name)
                .font(.system(size: 10))
                .foregroundColor(isSelected ? .white : .gray)
                .lineLimit(2)
                .multilineTextAlignment(.center)
        }
        .frame(width: 120)
    }
    
    private func formatDuration(_ seconds: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: seconds) ?? "00:00"
    }
}

// MARK: - List View
struct ListView: View {
    let items: [MediaPoolItem]
    @Binding var selectedItems: Set<String>
    
    public var body: some View {
        VStack(spacing: 1) {
            ForEach(items) { item in
                MediaListRow(
                    item: item,
                    isSelected: selectedItems.contains(item.id.uuidString)
                )
                .onTapGesture {
                    toggleSelection(item.id.uuidString)
                }
            }
        }
    }
    
    private func toggleSelection(_ id: String) {
        if selectedItems.contains(id) {
            selectedItems.remove(id)
        } else {
            selectedItems.insert(id)
        }
    }
}

// MARK: - Media List Row
struct MediaListRow: View {
    let item: MediaPoolItem
    let isSelected: Bool
    
    public var body: some View {
        HStack(spacing: 8) {
            // Icon
            Image(systemName: item.hasVideo ? "video" : "waveform")
                .font(.system(size: 12))
                .foregroundColor(.gray)
                .frame(width: 20)
            
            // Name
            Text(item.name)
                .font(.system(size: 11))
                .foregroundColor(isSelected ? .white : .gray)
            
            Spacer()
            
            // Duration
            if item.duration > 0 {
                Text(formatDuration(item.duration))
                    .font(.system(size: 10))
                    .foregroundColor(.gray)
            }
            
            // File size
            Text(formatFileSize(item.fileSize))
                .font(.system(size: 10))
                .foregroundColor(.gray)
                .frame(width: 60, alignment: .trailing)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isSelected ? Color.orange.opacity(0.2) : Color.clear)
    }
    
    private func formatDuration(_ seconds: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: seconds) ?? "00:00"
    }
    
    private func formatFileSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

// MARK: - Filmstrip View
struct FilmstripView: View {
    let items: [MediaPoolItem]
    @Binding var selectedItems: Set<String>
    
    public var body: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 2) {
                ForEach(items) { item in
                    FilmstripFrame(
                        item: item,
                        isSelected: selectedItems.contains(item.id.uuidString)
                    )
                    .onTapGesture {
                        toggleSelection(item.id.uuidString)
                    }
                }
            }
            .padding()
        }
    }
    
    private func toggleSelection(_ id: String) {
        if selectedItems.contains(id) {
            selectedItems.remove(id)
        } else {
            selectedItems.insert(id)
        }
    }
}

// MARK: - Filmstrip Frame
struct FilmstripFrame: View {
    let item: MediaPoolItem
    let isSelected: Bool
    
    public var body: some View {
        VStack(spacing: 2) {
            if let thumbnail = item.thumbnail {
                thumbnail
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                Rectangle()
                    .fill(Color.gray.opacity(0.2))
            }
        }
        .frame(width: 80, height: 60)
        .overlay(
            Rectangle()
                .stroke(isSelected ? Color.orange : Color.clear, lineWidth: 2)
        )
    }
}

// MARK: - Media Info Bar
struct MediaInfoBar: View {
    let selectedItems: Set<String>
    @EnvironmentObject private var store: UnifiedStore
    
    var selectedItem: MediaPoolItem? {
        guard selectedItems.count == 1,
              let id = selectedItems.first else { return nil }
        return store.mediaItems.first { $0.id.uuidString == id }
    }
    
    public var body: some View {
        HStack(spacing: 16) {
            if let item = selectedItem {
                // Thumbnail
                if let thumbnail = item.thumbnail {
                    thumbnail
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 80, height: 45)
                        .cornerRadius(4)
                }
                
                // Info
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.name)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.white)
                    
                    HStack(spacing: 8) {
                        if item.duration > 0 {
                            Label(formatDuration(item.duration), systemImage: "clock")
                        }
                        // Resolution not available in MediaPoolItem
                        // TODO: Add resolution property to MediaPoolItem
                        Label(formatFileSize(item.fileSize), systemImage: "doc")
                    }
                    .font(.system(size: 9))
                    .foregroundColor(.gray)
                }
                
                Spacer()
                
                // Actions
                HStack(spacing: 4) {
                    Button(action: addToTimeline) {
                        Image(systemName: "plus.circle")
                            .font(.system(size: 14))
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.green)
                    .help("Add to Timeline")
                    
                    Button(action: showInFinder) {
                        Image(systemName: "folder")
                            .font(.system(size: 14))
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.gray)
                    .help("Show in Finder")
                }
            } else if selectedItems.count > 1 {
                Text("\(selectedItems.count) items selected")
                    .font(.system(size: 11))
                    .foregroundColor(.gray)
            } else {
                Text("No selection")
                    .font(.system(size: 11))
                    .foregroundColor(.gray.opacity(0.5))
            }
        }
        .padding(.horizontal, 12)
    }
    
    private func formatDuration(_ seconds: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: seconds) ?? "00:00:00"
    }
    
    private func formatFileSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
    
    private func addToTimeline() {
        if let item = selectedItem {
            store.addToTimeline(item)
        }
    }
    
    private func showInFinder() {
        if let item = selectedItem {
            NSWorkspace.shared.selectFile(item.url.path, inFileViewerRootedAtPath: "")
        }
    }
}

// MARK: - Media Bin Model
public struct MediaBin: Identifiable, Equatable {
    public let id: String
    public let name: String
    public let icon: String
    public let color: Color
    public var children: [MediaBin] = []
    public var itemCount: Int = 0
    
    public init(id: String = UUID().uuidString, name: String, icon: String = "folder", color: Color = .gray, children: [MediaBin] = []) {
        self.id = id
        self.name = name
        self.icon = icon
        self.color = color
        self.children = children
    }
    
    static let master = MediaBin(id: "master", name: "Master", color: .blue)
    
    func contains(_ item: MediaPoolItem) -> Bool {
        // Logic to determine if item belongs to this bin
        true // Placeholder
    }
}
