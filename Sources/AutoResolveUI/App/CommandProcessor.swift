import Foundation
import AutoResolveCore

public class CommandProcessor {
    private var undoStack: [Command] = []
    private var redoStack: [Command] = []
    
    public init() {}
    
    public func execute(_ command: Command, on project: Project) throws {
        try project.execute(command)
        undoStack.append(command)
        redoStack.removeAll()
    }
    
    public func undo(on project: Project) throws {
        guard let command = undoStack.popLast() else { return }
        try project.undo(command)
        redoStack.append(command)
    }
    
    public func redo(on project: Project) throws {
        guard let command = redoStack.popLast() else { return }
        try project.execute(command)
        undoStack.append(command)
    }
}
