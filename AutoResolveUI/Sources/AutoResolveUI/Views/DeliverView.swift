import SwiftUI

public struct DeliverView: View {
    @EnvironmentObject var appState: AppState
    @State private var selectedFormat: ExportFormat = .fcpxml
    @State private var outputPath: String = ""
    @State private var exportPreset: String = "Default"
    
    enum ExportFormat: String, CaseIterable {
        case fcpxml = "Final Cut Pro XML"
        case edl = "EDL"
        case mp4 = "MP4 Video"
        case mov = "QuickTime Movie"
        
        var fileExtension: String {
            switch self {
            case .fcpxml: return "fcpxml"
            case .edl: return "edl"
            case .mp4: return "mp4"
            case .mov: return "mov"
            }
        }
    }
    
    public var body: some View {
        HStack(spacing: 0) {
            // Format selection sidebar
            VStack(alignment: .leading, spacing: 0) {
                Text("Export Format")
                    .font(UITheme.Typography.headline)
                    .foregroundColor(UITheme.Colors.textPrimary)
                    .padding(UITheme.Sizes.spacingM)
                
                VStack(spacing: UITheme.Sizes.spacingXS) {
                    ForEach(ExportFormat.allCases, id: \.self) { format in
                        FormatOption(
                            format: format,
                            isSelected: selectedFormat == format
                        ) {
                            selectedFormat = format
                        }
                    }
                }
                .padding(UITheme.Sizes.spacingM)
                
                Spacer()
            }
            .frame(width: UITheme.Sizes.sidebarWidth)
            .background(UITheme.Colors.surface)
            
            // Export settings
            VStack(alignment: .leading, spacing: UITheme.Sizes.spacingL) {
                Text("Export Settings")
                    .font(UITheme.Typography.title)
                    .foregroundColor(UITheme.Colors.textPrimary)
                
                // Format info
                VStack(alignment: .leading, spacing: UITheme.Sizes.spacingM) {
                    HStack {
                        Text("Format:")
                            .font(UITheme.Typography.body)
                            .foregroundColor(UITheme.Colors.textSecondary)
                        Text(selectedFormat.rawValue)
                            .font(UITheme.Typography.body)
                            .foregroundColor(UITheme.Colors.textPrimary)
                    }
                    
                    HStack {
                        Text("Extension:")
                            .font(UITheme.Typography.body)
                            .foregroundColor(UITheme.Colors.textSecondary)
                        Text(".\(selectedFormat.fileExtension)")
                            .font(UITheme.Typography.mono)
                            .foregroundColor(UITheme.Colors.textPrimary)
                    }
                }
                .panelStyle()
                
                // Output settings
                VStack(alignment: .leading, spacing: UITheme.Sizes.spacingM) {
                    Text("Output Location")
                        .font(UITheme.Typography.headline)
                        .foregroundColor(UITheme.Colors.textPrimary)
                    
                    HStack {
                        TextField("Output path (optional)", text: $outputPath)
                            .textFieldStyle(.roundedBorder)
                        
                        Button("Browse...") {
                            // File browser implementation
                        }
                    }
                    
                    Text("Leave empty to export to default artifacts folder")
                        .font(UITheme.Typography.caption)
                        .foregroundColor(UITheme.Colors.textSecondary)
                }
                .panelStyle()
                
                // Preset selection
                if selectedFormat == .mp4 || selectedFormat == .mov {
                    VStack(alignment: .leading, spacing: UITheme.Sizes.spacingM) {
                        Text("Export Preset")
                            .font(UITheme.Typography.headline)
                            .foregroundColor(UITheme.Colors.textPrimary)
                        
                        Picker("Preset", selection: $exportPreset) {
                            Text("Default").tag("Default")
                            Text("High Quality").tag("High Quality")
                            Text("Web Optimized").tag("Web Optimized")
                            Text("Proxy").tag("Proxy")
                        }
                        .pickerStyle(MenuPickerStyle())
                    }
                    .panelStyle()
                }
                
                // Timeline info
                if let timeline = appState.timeline {
                    VStack(alignment: .leading, spacing: UITheme.Sizes.spacingM) {
                        Text("Timeline Information")
                            .font(UITheme.Typography.headline)
                            .foregroundColor(UITheme.Colors.textPrimary)
                        
                        HStack {
                            Text("Name:")
                                .font(UITheme.Typography.body)
                                .foregroundColor(UITheme.Colors.textSecondary)
                            Text(timeline.name)
                                .font(UITheme.Typography.body)
                                .foregroundColor(UITheme.Colors.textPrimary)
                        }
                        
                        HStack {
                            Text("Duration:")
                                .font(UITheme.Typography.body)
                                .foregroundColor(UITheme.Colors.textSecondary)
                            Text(appState.timebase.timecodeFromTime(timeline.cmDuration))
                                .font(UITheme.Typography.timecode)
                                .foregroundColor(UITheme.Colors.textPrimary)
                        }
                        
                        HStack {
                            Text("Tracks:")
                                .font(UITheme.Typography.body)
                                .foregroundColor(UITheme.Colors.textSecondary)
                            Text("\(timeline.tracks.count)")
                                .font(UITheme.Typography.body)
                                .foregroundColor(UITheme.Colors.textPrimary)
                        }
                    }
                    .panelStyle()
                }
                
                Spacer()
                
                // Export button
                HStack {
                    Spacer()
                    
                    Button("Export") {
                        Task {
                            await performExport()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(appState.timeline == nil || appState.isProcessing)
                }
            }
            .padding(UITheme.Sizes.spacingXL)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(UITheme.Colors.background)
    }
    
    func performExport() async {
        switch selectedFormat {
        case .fcpxml:
            await appState.exportFCPXML()
        case .edl:
            await appState.exportEDL()
        case .mp4, .mov:
            // MP4/MOV export would be implemented here
            appState.statusMessage = "Video export not yet implemented"
        }
    }
}

struct FormatOption: View {
    let format: DeliverView.ExportFormat
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isSelected ? UITheme.Colors.accent : UITheme.Colors.textDisabled)
                
                Text(format.rawValue)
                    .font(UITheme.Typography.body)
                    .foregroundColor(isSelected ? UITheme.Colors.textPrimary : UITheme.Colors.textSecondary)
                
                Spacer()
            }
            .padding(UITheme.Sizes.spacingS)
            .background(isSelected ? UITheme.Colors.surfaceLight : Color.clear)
            .cornerRadius(UITheme.Sizes.cornerRadiusS)
        }
        .buttonStyle(PlainButtonStyle())
    }
}