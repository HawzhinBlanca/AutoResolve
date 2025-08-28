// AUTORESOLVE V3.0 - IMPORT SUPPORTING VIEWS
// Additional views for the professional import dialog

import SwiftUI
import AppKit
import AVFoundation

// MARK: - File Browser View
struct FileBrowserView: View {
    @Binding var currentPath: URL
    @ObservedObject var importManager: ImportDialogManager
    @State private var files: [URL] = []
    @State private var selectedFiles: Set<URL> = []
    
    public var body: some View {
        ScrollView {
            LazyVStack(spacing: 1) {
                ForEach(files, id: \.self) { file in
                    FileBrowserRow(
                        file: file,
                        isSelected: selectedFiles.contains(file),
                        onSelect: { toggleSelection(file) },
                        onDoubleClick: { handleDoubleClick(file) }
                    )
                }
            }
        }
        .onAppear {
            loadFiles()
        }
        .onChange(of: currentPath) { _, _ in
            loadFiles()
        }
    }
    
    private func loadFiles() {
        do {
            let contents = try FileManager.default.contentsOfDirectory(
                at: currentPath,
                includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey, .contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
            files = contents.sorted { (a, b) in a.lastPathComponent < b.lastPathComponent }
        } catch {
            files = []
        }
    }
    
    private func toggleSelection(_ file: URL) {
        if selectedFiles.contains(file) {
            selectedFiles.remove(file)
            importManager.removeFile(file)
        } else {
            selectedFiles.insert(file)
            importManager.addFile(file)
        }
    }
    
    private func handleDoubleClick(_ file: URL) {
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: file.path, isDirectory: &isDirectory) {
            if isDirectory.boolValue {
                currentPath = file
            } else {
                importManager.addFile(file)
            }
        }
    }
}

// MARK: - File Browser Row
struct FileBrowserRow: View {
    let file: URL
    let isSelected: Bool
    let onSelect: () -> Void
    let onDoubleClick: () -> Void
    
    @State private var fileInfo: FileInfo?
    
    struct FileInfo {
        let isDirectory: Bool
        let size: Int64
        let modifiedDate: Date
        let icon: NSImage
    }
    
    public var body: some View {
        HStack(spacing: 8) {
            if let info = fileInfo {
                Image(nsImage: info.icon)
                    .resizable()
                    .frame(width: 16, height: 16)
            }
            
            Text(file.lastPathComponent)
                .font(.system(size: 11))
                .lineLimit(1)
            
            Spacer()
            
            if let info = fileInfo, !info.isDirectory {
                Text(formatFileSize(info.size))
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture(perform: onSelect)
        .onTapGesture(count: 2, perform: onDoubleClick)
        .onAppear {
            loadFileInfo()
        }
    }
    
    private func loadFileInfo() {
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath: file.path, isDirectory: &isDirectory)
        
        let attributes = try? FileManager.default.attributesOfItem(atPath: file.path)
        
        fileInfo = FileInfo(
            isDirectory: isDirectory.boolValue,
            size: attributes?[.size] as? Int64 ?? 0,
            modifiedDate: attributes?[.modificationDate] as? Date ?? Date(),
            icon: NSWorkspace.shared.icon(forFile: file.path)
        )
    }
    
    private func formatFileSize(_ size: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: size)
    }
}

// MARK: - Path Bar
struct PathBar: View {
    @Binding var currentPath: URL
    let onNavigate: () -> Void
    
    private var pathComponents: [URL] {
        var components: [URL] = []
        var url = currentPath
        
        while url.path != "/" {
            components.insert(url, at: 0)
            url = url.deletingLastPathComponent()
        }
        
        return components
    }
    
    public var body: some View {
        HStack(spacing: 4) {
            ForEach(pathComponents, id: \.self) { component in
                HStack(spacing: 4) {
                    if component != pathComponents.first {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                    }
                    
                    Button(action: {
                        currentPath = component
                        onNavigate()
                    }) {
                        Text(component.lastPathComponent)
                            .font(.system(size: 11))
                            .lineLimit(1)
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
            
            Spacer()
        }
    }
}

// MARK: - Favorite Button
struct FavoriteButton: View {
    let url: URL
    let action: () -> Void
    
    public var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: "star.fill")
                    .font(.system(size: 10))
                    .foregroundColor(.yellow)
                
                Text(url.lastPathComponent)
                    .font(.system(size: 11))
                    .lineLimit(1)
                
                Spacer()
            }
            .padding(.vertical, 4)
            .padding(.horizontal, 12)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Video Preview View
struct VideoPreviewView: View {
    let url: URL
    @State private var player: AVPlayer?
    @State private var thumbnail: NSImage?
    
    public var body: some View {
        ZStack {
            if let thumbnail = thumbnail {
                Image(nsImage: thumbnail)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
            }
            
            // Play button overlay
            Button(action: playVideo) {
                Image(systemName: "play.circle.fill")
                    .font(.system(size: 48))
                    .foregroundColor(.white.opacity(0.8))
            }
            .buttonStyle(PlainButtonStyle())
        }
        .onAppear {
            generateThumbnail()
        }
    }
    
    private func generateThumbnail() {
        Task {
            let asset = AVAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            
            let time = CMTime(seconds: 1, preferredTimescale: 600)
            
            do {
                let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
                thumbnail = NSImage(cgImage: cgImage, size: CGSize(width: 640, height: 360))
            } catch {
                print("Failed to generate thumbnail: \(error)")
            }
        }
    }
    
    private func playVideo() {
        player = AVPlayer(url: url)
        player?.play()
    }
}

// MARK: - Preview Controls
struct PreviewControls: View {
    @Binding var isPlaying: Bool
    let file: URL
    @ObservedObject var importManager: ImportDialogManager
    
    public var body: some View {
        HStack {
            // Play/Pause
            Button(action: { isPlaying.toggle() }) {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 16))
            }
            .buttonStyle(PlainButtonStyle())
            
            // Scrubber
            Slider(value: .constant(0.5))
                .controlSize(.small)
            
            // Duration
            if let metadata = importManager.validationResults[file]?.metadata {
                if metadata.duration > 0 {
                    Text(formatDuration(metadata.duration))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.white)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.6))
        )
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.unitsStyle = .positional
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: duration) ?? "0:00"
    }
}

// MARK: - Import File Row
struct ImportFileRow: View {
    let file: URL
    let isSelected: Bool
    @ObservedObject var importManager: ImportDialogManager
    let onSelect: () -> Void
    
    public var body: some View {
        HStack(spacing: 12) {
            // Checkbox
            Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                .font(.system(size: 12))
                .foregroundColor(isSelected ? .accentColor : .secondary)
            
            // Thumbnail
            if let preview = importManager.getPreview(for: file) {
                Image(nsImage: preview)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 40, height: 30)
                    .clipped()
                    .cornerRadius(4)
            }
            
            // File info
            VStack(alignment: .leading, spacing: 2) {
                Text(file.lastPathComponent)
                    .font(.system(size: 11))
                    .lineLimit(1)
                
                HStack(spacing: 8) {
                    if let metadata = importManager.validationResults[file]?.metadata {
                        Text(formatDuration(metadata.duration))
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                        
                        if let resolution = metadata.resolution {
                            Text("\(Int(resolution.width))×\(Int(resolution.height))")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                        }
                        
                        if let fps = metadata.frameRate {
                            Text("\(String(format: "%.2f", fps)) fps")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            
            Spacer()
            
            // Validation status
            if let result = importManager.validationResults[file] {
                if !result.warnings.isEmpty {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11))
                        .foregroundColor(.yellow)
                        .help(result.warnings.joined(separator: "\n"))
                } else if result.isValid {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 11))
                        .foregroundColor(.green)
                }
            }
            
            // File size
            if let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                Text(formatFileSize(Int64(size)))
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture(perform: onSelect)
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.unitsStyle = .abbreviated
        return formatter.string(from: duration) ?? "0s"
    }
    
    private func formatFileSize(_ size: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: size)
    }
}

// MARK: - Media Pool View Model Extension
// MediaPoolViewModel is now fully implemented in MediaPool/MediaPoolViewModel.swift
