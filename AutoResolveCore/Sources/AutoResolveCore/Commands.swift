import Foundation

// MARK: - Timeline Commands (Event Sourcing)

public enum Command: Codable, Hashable {
    case blade(at: Tick, trackIndex: Int)
    case trim(clipId: UUID, edge: Edge, to: Tick)
    case delete(clipId: UUID)
    case snapToggle(enabled: Bool)
    case undo
    case redo
    case rippleDelete(clipId: UUID)
    case roll(clipId: UUID, delta: Tick)
    case slip(clipId: UUID, delta: Tick)
    case slide(clipId: UUID, delta: Tick)
    
    public enum Edge: String, Codable {
        case leading, trailing
    }
}

// MARK: - Command Processor

public class CommandProcessor {
    private var history: [Command] = []
    private var undoStack: [Command] = []
    private var redoStack: [Command] = []
    
    public init() {}
    
    public func execute(_ command: Command) -> Result<Void, CommandError> {
        // Validate command
        guard validate(command) else {
            return .failure(.invalidCommand)
        }
        
        // Execute and record
        history.append(command)
        undoStack.append(command)
        redoStack.removeAll() // Clear redo on new command
        
        return .success(())
    }
    
    public func undo() -> Result<Command?, CommandError> {
        guard let command = undoStack.popLast() else {
            return .success(nil)
        }
        
        redoStack.append(command)
        history.append(.undo)
        
        return .success(command)
    }
    
    public func redo() -> Result<Command?, CommandError> {
        guard let command = redoStack.popLast() else {
            return .success(nil)
        }
        
        undoStack.append(command)
        history.append(.redo)
        
        return .success(command)
    }
    
    private func validate(_ command: Command) -> Bool {
        // Command validation logic
        switch command {
        case .blade(let tick, _):
            return tick.value >= 0
        case .trim(_, _, let tick):
            return tick.value >= 0
        default:
            return true
        }
    }
    
    public func getHistory() -> [Command] {
        return history
    }
}

// MARK: - Command Errors

public enum CommandError: Error {
    case invalidCommand
    case executionFailed(String)
    case undoFailed
    case redoFailed
}