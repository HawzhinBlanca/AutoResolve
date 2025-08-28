import Foundation
import Combine
import SwiftUI

@MainActor
final class OperationState: ObservableObject {
    @Published var isLoading = false
    @Published var error: Error?
    @Published var progress: Double = 0
    
    func withErrorHandling(_ operation: @escaping () async throws -> Void) {
        Task {
            isLoading = true
            defer { isLoading = false }
            do {
                try await operation()
            } catch {
                self.error = error
                HapticFeedback.error()
            }
        }
    }
}

enum HapticFeedback {
    static func error() {
        #if os(macOS)
        NSHapticFeedbackManager.defaultPerformer.perform(.levelChange, performanceTime: .now)
        #endif
    }
}





