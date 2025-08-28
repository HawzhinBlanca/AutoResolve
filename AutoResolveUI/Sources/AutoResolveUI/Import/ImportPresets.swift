// AUTORESOLVE V3.0 - IMPORT PRESETS
import Combine
// Professional import presets for different workflows

import SwiftUI
import AppKit

// MARK: - Import Preset Manager
class ImportPresetManager: ObservableObject {
    @Published var presets: [ImportPreset] = []
    @Published var selectedPreset: ImportPreset?
    
    private let presetsURL: URL
    
    init() {
        // Get presets directory
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appFolder = appSupport.appendingPathComponent("AutoResolve")
        presetsURL = appFolder.appendingPathComponent("ImportPresets")
        
        // Create directory if needed
        try? FileManager.default.createDirectory(at: presetsURL, withIntermediateDirectories: true)
        
        // Load default presets
        loadDefaultPresets()
        
        // Load custom presets
        loadCustomPresets()
    }
    
    // MARK: - Default Presets
    private func loadDefaultPresets() {
        presets = [
            ImportPreset(
                name: "Original Media",
                icon: "doc",
                description: "Import media without modifications",
                settings: ImportSettings(
                    copyToProject: false,
                    createOptimized: false,
                    createProxies: false,
                    importMetadata: true
                ),
                isBuiltIn: true
            ),
            
            ImportPreset(
                name: "Offline Edit",
                icon: "rectangle.compress.vertical",
                description: "Create lightweight proxies for editing",
                settings: ImportSettings(
                    copyToProject: true,
                    createOptimized: false,
                    createProxies: true,
                    proxyResolution: .quarter,
                    proxyCodec: .proRes422Proxy,
                    importMetadata: true
                ),
                isBuiltIn: true
            ),
            
            ImportPreset(
                name: "Online Edit",
                icon: "sparkles",
                description: "Create optimized media for best performance",
                settings: ImportSettings(
                    copyToProject: true,
                    createOptimized: true,
                    createProxies: false,
                    importMetadata: true
                ),
                isBuiltIn: true
            ),
            
            ImportPreset(
                name: "Archive Project",
                icon: "archivebox",
                description: "Copy all media to project with proxies",
                settings: ImportSettings(
                    copyToProject: true,
                    createOptimized: false,
                    createProxies: true,
                    proxyResolution: .half,
                    proxyCodec: .proRes422LT,
                    importMetadata: true,
                    importMarkers: true,
                    importLUTs: true
                ),
                isBuiltIn: true
            ),
            
            ImportPreset(
                name: "Web Delivery",
                icon: "globe",
                description: "Transcode for web platforms",
                settings: ImportSettings(
                    copyToProject: true,
                    createOptimized: false,
                    createProxies: true,
                    proxyResolution: .full,
                    proxyCodec: .h264,
                    normalizeAudio: true,
                    sampleRate: 48000
                ),
                isBuiltIn: true
            ),
            
            ImportPreset(
                name: "Cinema DNG",
                icon: "film",
                description: "Import RAW cinema footage",
                settings: ImportSettings(
                    copyToProject: false,
                    createOptimized: false,
                    createProxies: true,
                    proxyResolution: .quarter,
                    proxyCodec: .proRes422Proxy,
                    importMetadata: true,
                    importLUTs: true
                ),
                isBuiltIn: true
            ),
            
            ImportPreset(
                name: "Audio Post",
                icon: "waveform",
                description: "Import audio with sync and normalization",
                settings: ImportSettings(
                    copyToProject: true,
                    autoSyncAudio: true,
                    normalizeAudio: true,
                    sampleRate: 48000,
                    importMetadata: true
                ),
                isBuiltIn: true
            ),
            
            ImportPreset(
                name: "VFX Plates",
                icon: "cube.transparent",
                description: "Import with alpha and high quality",
                settings: ImportSettings(
                    copyToProject: true,
                    createOptimized: true,
                    alphaHandling: .premultiplied,
                    importMetadata: true,
                    importMarkers: true
                ),
                isBuiltIn: true
            )
        ]
    }
    
    // MARK: - Custom Presets
    func loadCustomPresets() {
        do {
            let files = try FileManager.default.contentsOfDirectory(
                at: presetsURL,
                includingPropertiesForKeys: nil
            )
            
            for file in files where file.pathExtension == "json" {
                if let data = try? Data(contentsOf: file),
                   let preset = try? JSONDecoder().decode(ImportPreset.self, from: data) {
                    presets.append(preset)
                }
            }
        } catch {
            print("Failed to load custom presets: \(error)")
        }
    }
    
    func savePreset(_ preset: ImportPreset) {
        guard !preset.isBuiltIn else { return }
        
        let url = presetsURL.appendingPathComponent("\(preset.id.uuidString).json")
        
        do {
            let data = try JSONEncoder().encode(preset)
            try data.write(to: url)
            
            if !presets.contains(where: { $0.id == preset.id }) {
                presets.append(preset)
            }
        } catch {
            print("Failed to save preset: \(error)")
        }
    }
    
    func deletePreset(_ preset: ImportPreset) {
        guard !preset.isBuiltIn else { return }
        
        let url = presetsURL.appendingPathComponent("\(preset.id.uuidString).json")
        try? FileManager.default.removeItem(at: url)
        
        presets.removeAll { $0.id == preset.id }
    }
    
    func applyPreset(_ preset: ImportPreset, to settings: inout ImportSettings) {
        settings = preset.settings
        selectedPreset = preset
    }
}

// MARK: - Import Preset Model
struct ImportPreset: Identifiable, Codable {
    public let id: UUID
    var name: String
    var icon: String
    var description: String
    var settings: ImportSettings
    var isBuiltIn: Bool
    var color: PresetColor
    
    init(_ id: UUID = UUID(),
         name: String,
         icon: String = "doc",
         description: String = "",
         settings: ImportSettings,
         isBuiltIn: Bool = false,
         color: PresetColor = .blue) {
        self.id = id
        self.name = name
        self.icon = icon
        self.description = description
        self.settings = settings
        self.isBuiltIn = isBuiltIn
        self.color = color
    }
    
    enum PresetColor: String, Codable, CaseIterable {
        case blue, green, orange, purple, red, yellow, gray
        
        var swiftUIColor: Color {
            switch self {
            case .blue: return .blue
            case .green: return .green
            case .orange: return .orange
            case .purple: return .purple
            case .red: return .red
            case .yellow: return .yellow
            case .gray: return .gray
            }
        }
    }
}

// MARK: - Import Preset Selector View
struct ImportPresetSelector: View {
    @ObservedObject var importManager: ImportDialogManager
    @StateObject private var presetManager = ImportPresetManager()
    @State private var showCreatePreset = false
    @State private var editingPreset: ImportPreset?
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Text("Import Presets")
                    .font(.system(size: 13, weight: .semibold))
                
                Spacer()
                
                Button(action: { showCreatePreset = true }) {
                    Image(systemName: "plus.circle")
                        .font(.system(size: 14))
                }
                .buttonStyle(PlainButtonStyle())
                .help("Create custom preset")
            }
            
            // Preset grid
            ScrollView {
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible())
                ], spacing: 8) {
                    ForEach(presetManager.presets) { preset in
                        PresetCard(
                            preset: preset,
                            isSelected: presetManager.selectedPreset?.id == preset.id,
                            onSelect: {
                                presetManager.applyPreset(preset, to: &importManager.settings)
                            },
                            onEdit: {
                                if !preset.isBuiltIn {
                                    editingPreset = preset
                                }
                            }
                        )
                    }
                }
            }
            .frame(maxHeight: 300)
        }
        .sheet(isPresented: $showCreatePreset) {
            CreatePresetView(
                presetManager: presetManager,
                baseSettings: importManager.settings
            )
        }
        .sheet(item: $editingPreset) { preset in
            EditPresetView(
                preset: preset,
                presetManager: presetManager
            )
        }
    }
}

// MARK: - Preset Card
struct PresetCard: View {
    let preset: ImportPreset
    let isSelected: Bool
    let onSelect: () -> Void
    let onEdit: () -> Void
    
    @State private var isHovered = false
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: preset.icon)
                    .font(.system(size: 16))
                    .foregroundColor(preset.color.swiftUIColor)
                
                Spacer()
                
                if preset.isBuiltIn {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                } else if isHovered {
                    Button(action: onEdit) {
                        Image(systemName: "pencil.circle")
                            .font(.system(size: 12))
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
            
            Text(preset.name)
                .font(.system(size: 11, weight: .medium))
                .lineLimit(1)
            
            Text(preset.description)
                .font(.system(size: 10))
                .foregroundColor(.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .frame(height: 80)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? preset.color.swiftUIColor.opacity(0.2) : Color.black.opacity(0.2))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(
                            isSelected ? preset.color.swiftUIColor : Color.clear,
                            lineWidth: 2
                        )
                )
        )
        .onHover { hovering in
            isHovered = hovering
        }
        .onTapGesture(perform: onSelect)
    }
}

// MARK: - Create Preset View
struct CreatePresetView: View {
    @ObservedObject var presetManager: ImportPresetManager
    let baseSettings: ImportSettings
    
    @State private var presetName = ""
    @State private var presetDescription = ""
    @State private var selectedIcon = "doc"
    @State private var selectedColor = ImportPreset.PresetColor.blue
    @State private var settings: ImportSettings
    @Environment(\.dismiss) private var dismiss
    
    init(presetManager: ImportPresetManager, baseSettings: ImportSettings) {
        self.presetManager = presetManager
        self.baseSettings = baseSettings
        self._settings = State(initialValue: baseSettings)
    }
    
    private let icons = [
        "doc", "folder", "film", "video", "waveform",
        "rectangle.compress.vertical", "sparkles", "archivebox",
        "globe", "cube.transparent", "star", "bookmark"
    ]
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Create Import Preset")
                    .font(.system(size: 16, weight: .semibold))
                
                Spacer()
                
                Button(action: { dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(PlainButtonStyle())
            }
            .padding()
            .background(DaVinciColors.headerBackground)
            
            Divider()
            
            // Content
            Form {
                Section("Preset Information") {
                    TextField("Preset Name", text: $presetName)
                    TextField("Description", text: $presetDescription)
                    
                    HStack {
                        Text("Icon:")
                        
                        Picker("", selection: $selectedIcon) {
                            ForEach(icons, id: \.self) { icon in
                                Image(systemName: icon).tag(icon)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                    }
                    
                    HStack {
                        Text("Color:")
                        
                        Picker("", selection: $selectedColor) {
                            ForEach(ImportPreset.PresetColor.allCases, id: \.self) { color in
                                Rectangle()
                                    .fill(color.swiftUIColor)
                                    .frame(width: 20, height: 20)
                                    .tag(color)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                    }
                }
                
                Section("Import Settings") {
                    Toggle("Copy to project folder", isOn: $settings.copyToProject)
                    Toggle("Create optimized media", isOn: $settings.createOptimized)
                    Toggle("Create proxies", isOn: $settings.createProxies)
                    
                    if settings.createProxies {
                        Picker("Proxy Resolution", selection: $settings.proxyResolution) {
                            ForEach(ProxyResolution.allCases, id: \.self) { res in
                                Text(res.rawValue).tag(res)
                            }
                        }
                        
                        Picker("Proxy Codec", selection: $settings.proxyCodec) {
                            ForEach(ProxyCodec.allCases, id: \.self) { codec in
                                Text(codec.rawValue).tag(codec)
                            }
                        }
                    }
                }
                
                Section("Audio Settings") {
                    Toggle("Auto-sync audio", isOn: $settings.autoSyncAudio)
                    Toggle("Normalize audio", isOn: $settings.normalizeAudio)
                }
                
                Section("Metadata") {
                    Toggle("Import metadata", isOn: $settings.importMetadata)
                    Toggle("Import markers", isOn: $settings.importMarkers)
                    Toggle("Import LUTs", isOn: $settings.importLUTs)
                }
            }
            .padding()
            
            Divider()
            
            // Footer
            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.escape)
                
                Spacer()
                
                Button("Create Preset") {
                    createPreset()
                }
                .keyboardShortcut(.return)
                .disabled(presetName.isEmpty)
                .buttonStyle(BorderedProminentButtonStyle())
            }
            .padding()
            .background(DaVinciColors.toolbarBackground)
        }
        .frame(width: 600, height: 700)
        .background(DaVinciColors.panelBackground)
    }
    
    private func createPreset() {
        let preset = ImportPreset(
            name: presetName,
            icon: selectedIcon,
            description: presetDescription,
            settings: settings,
            isBuiltIn: false,
            color: selectedColor
        )
        
        presetManager.savePreset(preset)
        dismiss()
    }
}

// MARK: - Edit Preset View
struct EditPresetView: View {
    @State var preset: ImportPreset
    @ObservedObject var presetManager: ImportPresetManager
    @Environment(\.dismiss) private var dismiss
    
    private let icons = [
        "doc", "folder", "film", "video", "waveform",
        "rectangle.compress.vertical", "sparkles", "archivebox",
        "globe", "cube.transparent", "star", "bookmark"
    ]
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Edit Import Preset")
                    .font(.system(size: 16, weight: .semibold))
                
                Spacer()
                
                Button(action: { deletePreset() }) {
                    Image(systemName: "trash")
                        .font(.system(size: 14))
                        .foregroundColor(.red)
                }
                .buttonStyle(PlainButtonStyle())
                .help("Delete preset")
                
                Button(action: { dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(PlainButtonStyle())
            }
            .padding()
            .background(DaVinciColors.headerBackground)
            
            // Similar content to CreatePresetView but editing existing preset
            // ... (form content)
            
            // Footer
            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.escape)
                
                Spacer()
                
                Button("Save Changes") {
                    saveChanges()
                }
                .keyboardShortcut(.return)
                .buttonStyle(BorderedProminentButtonStyle())
            }
            .padding()
            .background(DaVinciColors.toolbarBackground)
        }
        .frame(width: 600, height: 700)
        .background(DaVinciColors.panelBackground)
    }
    
    private func saveChanges() {
        presetManager.savePreset(preset)
        dismiss()
    }
    
    private func deletePreset() {
        presetManager.deletePreset(preset)
        dismiss()
    }
}

// MARK: - Proxy Settings View
struct ProxySettingsView: View {
    @Binding var settings: ImportSettings
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Resolution", selection: $settings.proxyResolution) {
                Text("Quarter (1/4)").tag(ProxyResolution.quarter)
                Text("Half (1/2)").tag(ProxyResolution.half)
                Text("Full").tag(ProxyResolution.full)
            }
            .pickerStyle(MenuPickerStyle())
            
            Picker("Codec", selection: $settings.proxyCodec) {
                ForEach(ProxyCodec.allCases, id: \.self) { codec in
                    Text(codec.rawValue).tag(codec)
                }
            }
            .pickerStyle(MenuPickerStyle())
        }
        .font(.system(size: 11))
    }
}

// MARK: - Settings Section View
struct SettingsSection<Content: View>: View {
    let title: String
    let isExpanded: Bool
    @ViewBuilder let content: () -> Content
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .font(.system(size: 10))
                
                Text(title)
                    .font(.system(size: 12, weight: .medium))
                
                Spacer()
            }
            .contentShape(Rectangle())
            .onTapGesture {
                withAnimation(.easeInOut(duration: 0.2)) {
                    // Toggle expansion
                }
            }
            
            if isExpanded {
                content()
                    .padding(.leading, 20)
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.black.opacity(0.1))
        )
    }
}
