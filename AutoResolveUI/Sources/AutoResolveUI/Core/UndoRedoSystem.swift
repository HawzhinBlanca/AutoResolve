// AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
// Professional Undo/Redo System with Command Pattern

import Foundation
import SwiftUI
import Combine

// MARK: - Command Protocol
protocol UndoableCommand {
    var description: String { get }
    func execute()
    func undo()
    func canMergeWith(_ other: UndoableCommand) -> Bool
    func mergedWith(_ other: UndoableCommand) -> UndoableCommand?
}

extension UndoableCommand {
    func canMergeWith(_ other: UndoableCommand) -> Bool { false }
    func mergedWith(_ other: UndoableCommand) -> UndoableCommand? { nil }
}

// MARK: - Professional Undo Manager
@MainActor
class ProfessionalUndoManager: ObservableObject {
    @Published private(set) var undoStack: [UndoableCommand] = []
    @Published private(set) var redoStack: [UndoableCommand] = []
    @Published private(set) var isExecuting: Bool = false
    
    private let maxUndoLevels: Int = 100
    private var mergeTimeWindow: TimeInterval = 1.0
    private var lastCommandTime: Date?
    
    var canUndo: Bool {
        !undoStack.isEmpty && !isExecuting
    }
    
    var canRedo: Bool {
        !redoStack.isEmpty && !isExecuting
    }
    
    var undoDescription: String? {
        undoStack.last?.description
    }
    
    var redoDescription: String? {
        redoStack.last?.description
    }
    
    // MARK: - Command Execution
    func execute(_ command: UndoableCommand) {
        guard !isExecuting else { return }
        
        isExecuting = true
        defer { isExecuting = false }
        
        // Try to merge with the last command if within time window
        let now = Date()
        let canMerge = lastCommandTime.map { now.timeIntervalSince($0) < mergeTimeWindow } ?? false
        
        if canMerge,
           let lastCommand = undoStack.last,
           command.canMergeWith(lastCommand),
           let mergedCommand = lastCommand.mergedWith(command) {
            
            // Replace last command with merged version
            undoStack[undoStack.count - 1] = mergedCommand
            command.execute()
        } else {
            // Execute new command
            command.execute()
            undoStack.append(command)
            
            // Limit undo stack size
            if undoStack.count > maxUndoLevels {
                undoStack.removeFirst()
            }
        }
        
        // Clear redo stack when new command is executed
        redoStack.removeAll()
        lastCommandTime = now
    }
    
    func undo() {
        guard canUndo else { return }
        
        isExecuting = true
        defer { isExecuting = false }
        
        let command = undoStack.removeLast()
        command.undo()
        redoStack.append(command)
        
        lastCommandTime = Date()
    }
    
    func redo() {
        guard canRedo else { return }
        
        isExecuting = true
        defer { isExecuting = false }
        
        let command = redoStack.removeLast()
        command.execute()
        undoStack.append(command)
        
        lastCommandTime = Date()
    }
    
    func clear() {
        undoStack.removeAll()
        redoStack.removeAll()
        lastCommandTime = nil
    }
    
    // MARK: - Group Commands
    func beginGroup(_ description: String) -> CommandGroup {
        CommandGroup(description: description, undoManager: self)
    }
}

// MARK: - Command Group for Batch Operations
class CommandGroup {
    private let description: String
    private var commands: [UndoableCommand] = []
    private weak var undoManager: ProfessionalUndoManager?
    
    init(description: String, undoManager: ProfessionalUndoManager) {
        self.description = description
        self.undoManager = undoManager
    }
    
    func add(_ command: UndoableCommand) {
        commands.append(command)
    }
    
    @MainActor
    func commit() {
        guard !commands.isEmpty else { return }
        
        let groupCommand = GroupCommand(description: description, commands: commands)
        undoManager?.execute(groupCommand)
    }
}

// MARK: - Concrete Commands

// Group Command for batch operations
struct GroupCommand: UndoableCommand {
    let description: String
    private let commands: [UndoableCommand]
    
    init(description: String, commands: [UndoableCommand]) {
        self.description = description
        self.commands = commands
    }
    
    func execute() {
        commands.forEach { $0.execute() }
    }
    
    func undo() {
        commands.reversed().forEach { $0.undo() }
    }
}

// Add Video Clip Command
struct AddVideoClipCommand: UndoableCommand {
    let description = "Add Video Clip"
    private let clip: VideoClip
    private let trackId: UUID
    private var project: VideoProject?
    
    init(clip: VideoClip, trackId: UUID, project: VideoProject) {
        self.clip = clip
        self.trackId = trackId
        self.project = project
    }
    
    func execute() {
        guard let project = project else { return }
        
        if let trackIndex = project.timeline.videoTracks.firstIndex(where: { $0.id == trackId }) {
            project.timeline.videoTracks[trackIndex].clips.append(clip)
        }
    }
    
    func undo() {
        guard let project = project else { return }
        
        if let trackIndex = project.timeline.videoTracks.firstIndex(where: { $0.id == trackId }) {
            project.timeline.videoTracks[trackIndex].clips.removeAll { $0.id == clip.id }
        }
    }
}

// Remove Video Clip Command
class RemoveVideoClipCommand: UndoableCommand {
    let description = "Remove Video Clip"
    private let clipId: UUID
    private let trackId: UUID
    private var removedClip: VideoClip?
    private var clipIndex: Int?
    private var project: VideoProject?
    
    init(clipId: UUID, trackId: UUID, project: VideoProject) {
        self.clipId = clipId
        self.trackId = trackId
        self.project = project
    }
    
    func execute() {
        guard let project = project else { return }
        
        if let trackIndex = project.timeline.videoTracks.firstIndex(where: { $0.id == trackId }),
           let clipIndex = project.timeline.videoTracks[trackIndex].clips.firstIndex(where: { $0.id == clipId }) {
            
            self.clipIndex = clipIndex
            self.removedClip = project.timeline.videoTracks[trackIndex].clips.remove(at: clipIndex)
        }
    }
    
    func undo() {
        guard let project = project,
              let removedClip = removedClip,
              let clipIndex = clipIndex else { return }
        
        if let trackIndex = project.timeline.videoTracks.firstIndex(where: { $0.id == trackId }) {
            project.timeline.videoTracks[trackIndex].clips.insert(removedClip, at: clipIndex)
        }
    }
}

// Move Clip Command
struct MoveClipCommand: UndoableCommand {
    let description = "Move Clip"
    private let clipId: UUID
    private let oldTime: TimeInterval
    private let newTime: TimeInterval
    private let oldTrackId: UUID
    private let newTrackId: UUID
    private var project: VideoProject?
    
    init(clipId: UUID, oldTime: TimeInterval, newTime: TimeInterval, 
         oldTrackId: UUID, newTrackId: UUID, project: VideoProject) {
        self.clipId = clipId
        self.oldTime = oldTime
        self.newTime = newTime
        self.oldTrackId = oldTrackId
        self.newTrackId = newTrackId
        self.project = project
    }
    
    func execute() {
        moveClip(from: oldTrackId, to: newTrackId, time: newTime)
    }
    
    func undo() {
        moveClip(from: newTrackId, to: oldTrackId, time: oldTime)
    }
    
    private func moveClip(from fromTrackId: UUID, to toTrackId: UUID, time: TimeInterval) {
        guard let project = project else { return }
        
        // Find and remove clip from source track
        var clip: VideoClip?
        if let fromTrackIndex = project.timeline.videoTracks.firstIndex(where: { $0.id == fromTrackId }),
           let clipIndex = project.timeline.videoTracks[fromTrackIndex].clips.firstIndex(where: { $0.id == clipId }) {
            clip = project.timeline.videoTracks[fromTrackIndex].clips.remove(at: clipIndex)
        }
        
        // Add clip to destination track
        if var clip = clip,
           let toTrackIndex = project.timeline.videoTracks.firstIndex(where: { $0.id == toTrackId }) {
            clip.timelineStartTime = time
            project.timeline.videoTracks[toTrackIndex].clips.append(clip)
        }
    }
    
    func canMergeWith(_ other: UndoableCommand) -> Bool {
        if let otherMove = other as? MoveClipCommand {
            return clipId == otherMove.clipId
        }
        return false
    }
    
    func mergedWith(_ other: UndoableCommand) -> UndoableCommand? {
        guard let otherMove = other as? MoveClipCommand,
              clipId == otherMove.clipId else { return nil }
        
        return MoveClipCommand(
            clipId: clipId,
            oldTime: oldTime,
            newTime: otherMove.newTime,
            oldTrackId: oldTrackId,
            newTrackId: otherMove.newTrackId,
            project: project
        )
    }
}

// Trim Clip Command
struct TrimClipCommand: UndoableCommand {
    let description = "Trim Clip"
    private let clipId: UUID
    private let trackId: UUID
    private let oldStartTime: TimeInterval
    private let oldDuration: TimeInterval
    private let newStartTime: TimeInterval
    private let newDuration: TimeInterval
    private var project: VideoProject?
    
    init(clipId: UUID, trackId: UUID, 
         oldStartTime: TimeInterval, oldDuration: TimeInterval,
         newStartTime: TimeInterval, newDuration: TimeInterval,
         project: VideoProject) {
        self.clipId = clipId
        self.trackId = trackId
        self.oldStartTime = oldStartTime
        self.oldDuration = oldDuration
        self.newStartTime = newStartTime
        self.newDuration = newDuration
        self.project = project
    }
    
    func execute() {
        updateClip(startTime: newStartTime, duration: newDuration)
    }
    
    func undo() {
        updateClip(startTime: oldStartTime, duration: oldDuration)
    }
    
    private func updateClip(startTime: TimeInterval, duration: TimeInterval) {
        guard let project = project else { return }
        
        if let trackIndex = project.timeline.videoTracks.firstIndex(where: { $0.id == trackId }),
           let clipIndex = project.timeline.videoTracks[trackIndex].clips.firstIndex(where: { $0.id == clipId }) {
            
            project.timeline.videoTracks[trackIndex].clips[clipIndex].timelineStartTime = startTime
            project.timeline.videoTracks[trackIndex].clips[clipIndex].duration = duration
        }
    }
    
    func canMergeWith(_ other: UndoableCommand) -> Bool {
        if let otherTrim = other as? TrimClipCommand {
            return clipId == otherTrim.clipId
        }
        return false
    }
    
    func mergedWith(_ other: UndoableCommand) -> UndoableCommand? {
        guard let otherTrim = other as? TrimClipCommand,
              clipId == otherTrim.clipId else { return nil }
        
        return TrimClipCommand(
            clipId: clipId,
            trackId: trackId,
            oldStartTime: oldStartTime,
            oldDuration: oldDuration,
            newStartTime: otherTrim.newStartTime,
            newDuration: otherTrim.newDuration,
            project: project
        )
    }
}

// Add Audio Clip Command
struct AddAudioClipCommand: UndoableCommand {
    let description = "Add Audio Clip"
    private let clip: AudioClip
    private let trackId: UUID
    private var project: VideoProject?
    
    init(clip: AudioClip, trackId: UUID, project: VideoProject) {
        self.clip = clip
        self.trackId = trackId
        self.project = project
    }
    
    func execute() {
        guard let project = project else { return }
        
        if let trackIndex = project.timeline.audioTracks.firstIndex(where: { $0.id == trackId }) {
            project.timeline.audioTracks[trackIndex].clips.append(clip)
        }
    }
    
    func undo() {
        guard let project = project else { return }
        
        if let trackIndex = project.timeline.audioTracks.firstIndex(where: { $0.id == trackId }) {
            project.timeline.audioTracks[trackIndex].clips.removeAll { $0.id == clip.id }
        }
    }
}

// Change Project Settings Command
struct ChangeProjectSettingsCommand: UndoableCommand {
    let description = "Change Project Settings"
    private let oldSettings: ProjectSettings
    private let newSettings: ProjectSettings
    private var project: VideoProject?
    
    init(oldSettings: ProjectSettings, newSettings: ProjectSettings, project: VideoProject) {
        self.oldSettings = oldSettings
        self.newSettings = newSettings
        self.project = project
    }
    
    func execute() {
        project?.settings = newSettings
    }
    
    func undo() {
        project?.settings = oldSettings
    }
}

// Add Track Command
struct AddTrackCommand: UndoableCommand {
    let description: String
    private let trackType: TrackType
    private let track: AnyTrack
    private var project: VideoProject?
    
    enum TrackType {
        case video, audio, effect, title
    }
    
    enum AnyTrack {
        case video(VideoTrack)
        case audio(AudioTrack)
        case effect(EffectTrack)
        case title(TitleTrack)
    }
    
    init(videoTrack: VideoTrack, project: VideoProject) {
        self.description = "Add Video Track"
        self.trackType = .video
        self.track = .video(videoTrack)
        self.project = project
    }
    
    init(audioTrack: AudioTrack, project: VideoProject) {
        self.description = "Add Audio Track"
        self.trackType = .audio
        self.track = .audio(audioTrack)
        self.project = project
    }
    
    func execute() {
        guard let project = project else { return }
        
        switch track {
        case .video(let videoTrack):
            project.timeline.videoTracks.append(videoTrack)
        case .audio(let audioTrack):
            project.timeline.audioTracks.append(audioTrack)
        case .effect(let effectTrack):
            project.timeline.effectTracks.append(effectTrack)
        case .title(let titleTrack):
            project.timeline.titleTracks.append(titleTrack)
        }
    }
    
    func undo() {
        guard let project = project else { return }
        
        switch track {
        case .video(let videoTrack):
            project.timeline.videoTracks.removeAll { $0.id == videoTrack.id }
        case .audio(let audioTrack):
            project.timeline.audioTracks.removeAll { $0.id == audioTrack.id }
        case .effect(let effectTrack):
            project.timeline.effectTracks.removeAll { $0.id == effectTrack.id }
        case .title(let titleTrack):
            project.timeline.titleTracks.removeAll { $0.id == titleTrack.id }
        }
    }
}

// MARK: - Undo Manager Extensions for Common Operations
extension ProfessionalUndoManager {
    
    // Convenience methods for common timeline operations
    func addVideoClip(_ clip: VideoClip, to trackId: UUID, in project: VideoProject) {
        let command = AddVideoClipCommand(clip: clip, trackId: trackId, project: project)
        execute(command)
    }
    
    func removeVideoClip(_ clipId: UUID, from trackId: UUID, in project: VideoProject) {
        let command = RemoveVideoClipCommand(clipId: clipId, trackId: trackId, project: project)
        execute(command)
    }
    
    func moveClip(_ clipId: UUID, 
                  from oldTrackId: UUID, to newTrackId: UUID,
                  oldTime: TimeInterval, newTime: TimeInterval,
                  in project: VideoProject) {
        let command = MoveClipCommand(
            clipId: clipId,
            oldTime: oldTime,
            newTime: newTime,
            oldTrackId: oldTrackId,
            newTrackId: newTrackId,
            project: project
        )
        execute(command)
    }
    
    func trimClip(_ clipId: UUID, in trackId: UUID,
                  from oldStart: TimeInterval, oldDuration: TimeInterval,
                  to newStart: TimeInterval, newDuration: TimeInterval,
                  in project: VideoProject) {
        let command = TrimClipCommand(
            clipId: clipId,
            trackId: trackId,
            oldStartTime: oldStart,
            oldDuration: oldDuration,
            newStartTime: newStart,
            newDuration: newDuration,
            project: project
        )
        execute(command)
    }
}