// AUTORESOLVE V3.0 - PROFESSIONAL IMPORT DIALOG
// DaVinci Resolve-style import with preview, validation, and batch processing

import SwiftUI
import AppKit
import AVFoundation
import UniformTypeIdentifiers

// MARK: - Professional Import Dialog
struct ProfessionalImportDialog: View {
    @StateObject private var importManager = ImportDialogManager()
    @Binding var isPresented: Bool
    @ObservedObject var mediaPool: MediaPoolViewModel
    
    @State private var selectedTab = ImportTab.media
    @State private var showingFileBrowser = false
    @State private var dragOver = false
    
    enum ImportTab: String, CaseIterable {
        case media = "Media"
        case project = "Project"
        case timeline = "Timeline"
        case batch = "Batch"
        
        var icon: String {
            switch self {
            case .media: return "photo.on.rectangle"
            case .project: return "folder"
            case .timeline: return "timeline.selection"
            case .batch: return "square.stack.3d.up"
            }
        }
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            ImportDialogHeader(
                selectedTab: $selectedTab,
                importManager: importManager,
                onClose: { isPresented = false }
            )
            
            Divider()
            
            // Content area
            HStack(spacing: 0) {
                // Source browser
                SourceBrowser(importManager: importManager)
                    .frame(width: 300)
                
                Divider()
                
                // Preview and file list
                VStack(spacing: 0) {
                    // Preview area
                    MediaPreviewArea(importManager: importManager)
                        .frame(height: 300)
                    
                    Divider()
                    
                    // File list
                    ImportFileList(importManager: importManager)
                }
                
                Divider()
                
                // Import settings
                ImportSettingsPanel(importManager: importManager)
                    .frame(width: 280)
            }
            
            Divider()
            
            // Footer with import controls
            ImportDialogFooter(
                importManager: importManager,
                mediaPool: mediaPool,
                onImport: { handleImport() },
                onCancel: { isPresented = false }
            )
        }
        .frame(width: 1200, height: 800)
        .background(DaVinciColors.panelBackground)
        .onDrop(of: [.fileURL], isTargeted: $dragOver) { providers in
            handleDrop(providers)
            return true
        }
        .overlay(
            // Drag overlay
            dragOver ? DragOverlay() : nil
        )
    }
    
    private func handleImport() {
        Task {
            await importManager.performImport(to: mediaPool)
            if importManager.importErrors.isEmpty {
                isPresented = false
            }
        }
    }
    
    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { data, _ in
                guard let data = data as? Data,
                      let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
                
                DispatchQueue.main.async {
                    importManager.addFile(url)
                }
            }
        }
        return true
    }
}

// MARK: - Import Dialog Header
struct ImportDialogHeader: View {
    @Binding var selectedTab: ProfessionalImportDialog.ImportTab
    @ObservedObject var importManager: ImportDialogManager
    let onClose: () -> Void
    
    public var body: some View {
        HStack {
            // Title and tabs
            HStack(spacing: 20) {
                Text("Import Media")
                    .font(.system(size: 16, weight: .semibold))
                
                Divider()
                    .frame(height: 20)
                
                // Tab selector
                HStack(spacing: 12) {
                    ForEach(ProfessionalImportDialog.ImportTab.allCases, id: \.self) { tab in
                        TabButton(
                            title: tab.rawValue,
                            icon: tab.icon,
                            isSelected: selectedTab == tab,
                            action: { selectedTab = tab }
                        )
                    }
                }
            }
            
            Spacer()
            
            // Import stats
            if !importManager.selectedFiles.isEmpty {
                HStack(spacing: 12) {
                    Label("\(importManager.selectedFiles.count) files", systemImage: "doc.stack")
                    Label(importManager.totalSizeFormatted, systemImage: "internaldrive")
                }
                .font(.system(size: 11))
                .foregroundColor(.secondary)
            }
            
            // Close button
            Button(action: onClose) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 16))
                    .foregroundColor(.secondary)
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding()
        .background(DaVinciColors.headerBackground)
    }
}

// MARK: - Source Browser
struct SourceBrowser: View {
    @ObservedObject var importManager: ImportDialogManager
    @State private var selectedLocation: SourceLocation = .desktop
    @State private var currentPath = FileManager.default.homeDirectoryForCurrentUser
    @State private var favorites: [URL] = []
    
    enum SourceLocation: String, CaseIterable {
        case desktop = "Desktop"
        case documents = "Documents"
        case downloads = "Downloads"
        case movies = "Movies"
        case external = "External"
        case recent = "Recent"
        
        var icon: String {
            switch self {
            case .desktop: return "menubar.dock.rectangle"
            case .documents: return "doc.text"
            case .downloads: return "arrow.down.circle"
            case .movies: return "film"
            case .external: return "externaldrive"
            case .recent: return "clock"
            }
        }
        
        var url: URL? {
            let fm = FileManager.default
            switch self {
            case .desktop: return fm.urls(for: .desktopDirectory, in: .userDomainMask).first
            case .documents: return fm.urls(for: .documentDirectory, in: .userDomainMask).first
            case .downloads: return fm.urls(for: .downloadsDirectory, in: .userDomainMask).first
            case .movies: return fm.urls(for: .moviesDirectory, in: .userDomainMask).first
            case .external: return nil
            case .recent: return nil
            }
        }
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Location selector
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(SourceLocation.allCases, id: \.self) { location in
                        SourceLocationButton(
                            location: location,
                            isSelected: selectedLocation == location,
                            action: { selectLocation(location) }
                        )
                    }
                    
                    Divider()
                        .padding(.vertical, 8)
                    
                    // Favorites
                    Text("FAVORITES")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 12)
                        .padding(.bottom, 4)
                    
                    ForEach(favorites, id: \.self) { url in
                        FavoriteButton(url: url) {
                            currentPath = url
                            loadDirectory()
                        }
                    }
                }
                .padding(.vertical, 8)
            }
            
            Divider()
            
            // Path bar
            PathBar(currentPath: $currentPath, onNavigate: loadDirectory)
                .padding(8)
            
            Divider()
            
            // File browser
            FileBrowserView(
                currentPath: $currentPath,
                importManager: importManager
            )
        }
        .background(DaVinciColors.sidebarBackground)
        .onAppear {
            loadFavorites()
            selectLocation(.desktop)
        }
    }
    
    private func selectLocation(_ location: SourceLocation) {
        selectedLocation = location
        if let url = location.url {
            currentPath = url
            loadDirectory()
        }
    }
    
    private func loadDirectory() {
        importManager.loadDirectory(at: currentPath)
    }
    
    private func loadFavorites() {
        // Load user favorites from preferences
        favorites = [
            FileManager.default.homeDirectoryForCurrentUser,
            FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first
        ].compactMap { $0 }
    }
}

// MARK: - Media Preview Area
struct MediaPreviewArea: View {
    @ObservedObject var importManager: ImportDialogManager
    @State private var currentFrame: NSImage?
    @State private var isPlaying = false
    
    public var body: some View {
        ZStack {
            if let selectedFile = importManager.selectedFiles.first {
                if let preview = importManager.getPreview(for: selectedFile) {
                    Image(nsImage: preview)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    // Generate preview
                    VideoPreviewView(url: selectedFile)
                }
                
                // Overlay controls
                VStack {
                    Spacer()
                    
                    PreviewControls(
                        isPlaying: $isPlaying,
                        file: selectedFile,
                        importManager: importManager
                    )
                    .padding()
                }
            } else {
                // Empty state
                VStack(spacing: 12) {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    
                    Text("Select files to preview")
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)
                }
            }
        }
        .background(Color.black)
    }
}

// MARK: - Import File List
struct ImportFileList: View {
    @ObservedObject var importManager: ImportDialogManager
    @State private var sortOrder = ImportDialogManager.SortOrder.name
    @State private var selectedFiles: Set<URL> = []
    
    public var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            HStack {
                Button(action: { importManager.selectAll() }) {
                    Text("Select All")
                        .font(.system(size: 11))
                }
                .buttonStyle(LinkButtonStyle())
                
                Button(action: { importManager.deselectAll() }) {
                    Text("Deselect All")
                        .font(.system(size: 11))
                }
                .buttonStyle(LinkButtonStyle())
                
                Spacer()
                
                // Sort menu
                Menu {
                    ForEach(ImportDialogManager.SortOrder.allCases, id: \.self) { order in
                        Button(order.rawValue) {
                            importManager.sort(by: order)
                        }
                    }
                } label: {
                    Label("Sort", systemImage: "arrow.up.arrow.down")
                        .font(.system(size: 11))
                }
                .menuStyle(BorderlessButtonMenuStyle())
                .frame(width: 80)
            }
            .padding(8)
            .background(DaVinciColors.toolbarBackground)
            
            // File list
            ScrollView {
                LazyVStack(spacing: 1) {
                    ForEach(importManager.selectedFiles, id: \.self) { file in
                        ImportFileRow(
                            file: file,
                            isSelected: selectedFiles.contains(file),
                            importManager: importManager,
                            onSelect: { toggleSelection(file) }
                        )
                    }
                }
            }
        }
    }
    
    private func toggleSelection(_ file: URL) {
        if selectedFiles.contains(file) {
            selectedFiles.remove(file)
        } else {
            selectedFiles.insert(file)
        }
        importManager.updateSelection(selectedFiles)
    }
}

// MARK: - Import Settings Panel
struct ImportSettingsPanel: View {
    @ObservedObject var importManager: ImportDialogManager
    @State private var expandedSections: Set<String> = ["General", "Video", "Audio"]
    
    public var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                // General settings
                SettingsSection(
                    title: "General",
                    isExpanded: expandedSections.contains("General")
                ) {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Copy to project folder", isOn: $importManager.settings.copyToProject)
                        Toggle("Create optimized media", isOn: $importManager.settings.createOptimized)
                        Toggle("Create proxies", isOn: $importManager.settings.createProxies)
                        
                        if importManager.settings.createProxies {
                            ProxySettingsView(settings: $importManager.settings)
                                .padding(.leading, 20)
                        }
                    }
                }
                
                // Video settings
                SettingsSection(
                    title: "Video",
                    isExpanded: expandedSections.contains("Video")
                ) {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Auto-detect frame rate", isOn: $importManager.settings.autoDetectFrameRate)
                        Toggle("Remove pulldown", isOn: $importManager.settings.removePulldown)
                        Toggle("Deinterlace", isOn: $importManager.settings.deinterlace)
                        
                        HStack {
                            Text("Alpha handling:")
                            Picker("", selection: $importManager.settings.alphaHandling) {
                                Text("None").tag(AlphaHandling.none)
                                Text("Straight").tag(AlphaHandling.straight)
                                Text("Premultiplied").tag(AlphaHandling.premultiplied)
                            }
                            .pickerStyle(MenuPickerStyle())
                            .frame(width: 120)
                        }
                    }
                }
                
                // Audio settings
                SettingsSection(
                    title: "Audio",
                    isExpanded: expandedSections.contains("Audio")
                ) {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Auto-sync audio", isOn: $importManager.settings.autoSyncAudio)
                        Toggle("Normalize audio levels", isOn: $importManager.settings.normalizeAudio)
                        
                        HStack {
                            Text("Sample rate:")
                            Picker("", selection: $importManager.settings.sampleRate) {
                                Text("Project").tag(0)
                                Text("44.1 kHz").tag(44100)
                                Text("48 kHz").tag(48000)
                                Text("96 kHz").tag(96000)
                            }
                            .pickerStyle(MenuPickerStyle())
                            .frame(width: 100)
                        }
                    }
                }
                
                // Metadata settings
                SettingsSection(
                    title: "Metadata",
                    isExpanded: expandedSections.contains("Metadata")
                ) {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Import metadata", isOn: $importManager.settings.importMetadata)
                        Toggle("Import markers", isOn: $importManager.settings.importMarkers)
                        Toggle("Import LUTs", isOn: $importManager.settings.importLUTs)
                    }
                }
                
                Divider()
                
                // Presets
                ImportPresetSelector(importManager: importManager)
            }
            .padding()
        }
        .background(DaVinciColors.inspectorBackground)
    }
}

// MARK: - Import Dialog Footer
struct ImportDialogFooter: View {
    @ObservedObject var importManager: ImportDialogManager
    @ObservedObject var mediaPool: MediaPoolViewModel
    let onImport: () -> Void
    let onCancel: () -> Void
    
    public var body: some View {
        HStack {
            // Error indicator
            if !importManager.importErrors.isEmpty {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.yellow)
                    Text("\(importManager.importErrors.count) errors")
                        .font(.system(size: 11))
                }
            }
            
            Spacer()
            
            // Progress indicator
            if importManager.isImporting {
                HStack(spacing: 8) {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                        .scaleEffect(0.8)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text(importManager.currentOperation)
                            .font(.system(size: 11))
                        
                        ProgressView(value: importManager.progress)
                            .frame(width: 200)
                    }
                }
            }
            
            Spacer()
            
            // Action buttons
            Button("Cancel", action: onCancel)
                .keyboardShortcut(.escape)
            
            Button("Import", action: onImport)
                .keyboardShortcut(.return)
                .disabled(importManager.selectedFiles.isEmpty || importManager.isImporting)
                .buttonStyle(BorderedProminentButtonStyle())
        }
        .padding()
        .background(DaVinciColors.toolbarBackground)
    }
}

// MARK: - Supporting Views
struct TabButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void
    
    public var body: some View {
        Button(action: action) {
            Label(title, systemImage: icon)
                .font(.system(size: 12, weight: isSelected ? .medium : .regular))
                .foregroundColor(isSelected ? .accentColor : .secondary)
                .padding(.vertical, 4)
                .padding(.horizontal, 8)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .fill(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
                )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct SourceLocationButton: View {
    let location: SourceBrowser.SourceLocation
    let isSelected: Bool
    let action: () -> Void
    
    public var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: location.icon)
                    .font(.system(size: 12))
                    .frame(width: 20)
                
                Text(location.rawValue)
                    .font(.system(size: 12))
                
                Spacer()
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 12)
            .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
            .contentShape(Rectangle())
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct DragOverlay: View {
    public var body: some View {
        ZStack {
            Rectangle()
                .fill(Color.accentColor.opacity(0.1))
                .overlay(
                    Rectangle()
                        .stroke(Color.accentColor, lineWidth: 2)
                )
            
            VStack(spacing: 12) {
                Image(systemName: "arrow.down.doc")
                    .font(.system(size: 48))
                    .foregroundColor(.accentColor)
                
                Text("Drop files to import")
                    .font(.system(size: 16, weight: .medium))
            }
        }
        .ignoresSafeArea()
    }
}

// MARK: - DaVinci Colors
struct DaVinciColors {
    static let panelBackground = Color(red: 0.157, green: 0.157, blue: 0.157)
    static let headerBackground = Color(red: 0.118, green: 0.118, blue: 0.118)
    static let sidebarBackground = Color(red: 0.137, green: 0.137, blue: 0.137)
    static let toolbarBackground = Color(red: 0.196, green: 0.196, blue: 0.196)
    static let inspectorBackground = Color(red: 0.176, green: 0.176, blue: 0.176)
}
