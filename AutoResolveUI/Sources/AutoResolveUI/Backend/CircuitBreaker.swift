// AUTORESOLVE V3.0 - CIRCUIT BREAKER PATTERN
// Advanced resilience pattern for fault tolerance

import Foundation
import SwiftUI
import Combine

// Configuration for CircuitBreaker
public struct CircuitBreakerConfiguration {
    let failureThreshold: Int      // Failures before opening
    let successThreshold: Int      // Successes to close from half-open
    let timeout: TimeInterval       // Time before half-open attempt
    let resetTimeout: TimeInterval  // Time to reset failure count
    
    public static let `default` = CircuitBreakerConfiguration(
        failureThreshold: 5,
        successThreshold: 2,
        timeout: 30,
        resetTimeout: 60
    )
    
    public static let aggressive = CircuitBreakerConfiguration(
        failureThreshold: 3,
        successThreshold: 1,
        timeout: 10,
        resetTimeout: 30
    )
    
    public static let lenient = CircuitBreakerConfiguration(
        failureThreshold: 10,
        successThreshold: 3,
        timeout: 60,
        resetTimeout: 120
    )
}

/// Circuit breaker implementation for preventing cascading failures
public class CircuitBreaker<Output>: ObservableObject {
    public enum State {
        case closed     // Normal operation
        case open       // Failing, reject requests
        case halfOpen   // Testing if service recovered
        
        var canAttemptRequest: Bool {
            self != .open
        }
    }
    
    // State management
    @Published public private(set) var state: State = .closed
    @Published public private(set) var failureCount = 0
    @Published public private(set) var successCount = 0
    @Published public private(set) var lastFailureTime: Date?
    @Published public private(set) var statistics = Statistics()
    
    private let configuration: CircuitBreakerConfiguration
    private let queue = DispatchQueue(label: "circuit.breaker", attributes: .concurrent)
    private var resetTimer: Timer?
    private var halfOpenTimer: Timer?
    
    // Statistics
    public struct Statistics {
        var totalRequests = 0
        var totalFailures = 0
        var totalSuccesses = 0
        var totalRejections = 0
        var lastStateChange = Date()
        var stateChanges = 0
        
        var successRate: Double {
            guard totalRequests > 0 else { return 0 }
            return Double(totalSuccesses) / Double(totalRequests)
        }
        
        var availability: Double {
            guard totalRequests > 0 else { return 0 }
            return Double(totalRequests - totalRejections) / Double(totalRequests)
        }
    }
    
    public init(configuration: CircuitBreakerConfiguration = .default) {
        self.configuration = configuration
    }
    
    // MARK: - Public Methods
    
    /// Execute a request through the circuit breaker
    public func execute<T>(
        request: @escaping () -> AnyPublisher<T, Error>
    ) -> AnyPublisher<T, Error> {
        Future<T, Error> { [weak self] promise in
            guard let self = self else {
                promise(.failure(CircuitBreakerError.deallocated))
                return
            }
            
            self.queue.async(flags: .barrier) {
                self.statistics.totalRequests += 1
                
                switch self.state {
                case .open:
                    // Check if we should transition to half-open
                    if let lastFailure = self.lastFailureTime,
                       Date().timeIntervalSince(lastFailure) > self.configuration.timeout {
                        self.transitionTo(.halfOpen)
                        self.attemptRequest(request: request, promise: promise)
                    } else {
                        self.statistics.totalRejections += 1
                        promise(.failure(CircuitBreakerError.circuitOpen))
                    }
                    
                case .closed, .halfOpen:
                    self.attemptRequest(request: request, promise: promise)
                }
            }
        }
        .eraseToAnyPublisher()
    }
    
    /// Reset the circuit breaker
    public func reset() {
        queue.async(flags: .barrier) {
            self.state = .closed
            self.failureCount = 0
            self.successCount = 0
            self.lastFailureTime = nil
            self.cancelTimers()
            
            logInfo("[CircuitBreaker] Reset to closed state", category: .backend)
        }
    }
    
    /// Force open the circuit
    public func trip() {
        queue.async(flags: .barrier) {
            self.transitionTo(.open)
            logWarning("[CircuitBreaker] Manually tripped", category: .backend)
        }
    }
    
    // MARK: - Private Methods
    
    private func attemptRequest<T>(
        request: @escaping () -> AnyPublisher<T, Error>,
        promise: @escaping (Result<T, Error>) -> Void
    ) {
        var cancellable: AnyCancellable?
        
        cancellable = request()
            .sink(
                receiveCompletion: { [weak self] completion in
                    switch completion {
                    case .finished:
                        self?.recordSuccess()
                    case .failure(let error):
                        self?.recordFailure()
                        promise(.failure(error))
                    }
                    cancellable?.cancel()
                },
                receiveValue: { value in
                    promise(.success(value))
                }
            )
    }
    
    private func recordSuccess() {
        queue.async(flags: .barrier) {
            self.statistics.totalSuccesses += 1
            
            switch self.state {
            case .halfOpen:
                self.successCount += 1
                if self.successCount >= self.configuration.successThreshold {
                    self.transitionTo(.closed)
                }
                
            case .closed:
                // Reset failure count on success
                if self.failureCount > 0 {
                    self.scheduleFailureReset()
                }
                
            case .open:
                break // Shouldn't happen
            }
            
            logDebug("[CircuitBreaker] Request succeeded (state: \(self.state))", category: .backend)
        }
    }
    
    private func recordFailure() {
        queue.async(flags: .barrier) {
            self.statistics.totalFailures += 1
            self.lastFailureTime = Date()
            
            switch self.state {
            case .closed:
                self.failureCount += 1
                if self.failureCount >= self.configuration.failureThreshold {
                    self.transitionTo(.open)
                }
                
            case .halfOpen:
                // Single failure in half-open returns to open
                self.transitionTo(.open)
                
            case .open:
                break // Shouldn't happen
            }
            
            logWarning("[CircuitBreaker] Request failed (failures: \(self.failureCount))", category: .backend)
        }
    }
    
    private func transitionTo(_ newState: State) {
        guard state != newState else { return }
        
        let oldState = state
        state = newState
        statistics.stateChanges += 1
        statistics.lastStateChange = Date()
        
        // Reset counters
        switch newState {
        case .closed:
            failureCount = 0
            successCount = 0
            cancelTimers()
            
        case .open:
            successCount = 0
            scheduleHalfOpenTransition()
            
        case .halfOpen:
            failureCount = 0
            successCount = 0
        }
        
        logInfo("[CircuitBreaker] State transition: \(oldState) → \(newState)", category: .backend)
    }
    
    private func scheduleHalfOpenTransition() {
        halfOpenTimer?.invalidate()
        halfOpenTimer = Timer.scheduledTimer(withTimeInterval: configuration.timeout, repeats: false) { [weak self] _ in
            self?.queue.async(flags: .barrier) {
                if self?.state == .open {
                    self?.transitionTo(.halfOpen)
                }
            }
        }
    }
    
    private func scheduleFailureReset() {
        resetTimer?.invalidate()
        resetTimer = Timer.scheduledTimer(withTimeInterval: configuration.resetTimeout, repeats: false) { [weak self] _ in
            self?.queue.async(flags: .barrier) {
                self?.failureCount = 0
                logDebug("[CircuitBreaker] Failure count reset", category: .backend)
            }
        }
    }
    
    private func cancelTimers() {
        resetTimer?.invalidate()
        halfOpenTimer?.invalidate()
        resetTimer = nil
        halfOpenTimer = nil
    }
}

// MARK: - Error Types

public enum CircuitBreakerError: LocalizedError {
    case circuitOpen
    case deallocated
    
    public var errorDescription: String? {
        switch self {
        case .circuitOpen:
            return "Circuit breaker is open - service unavailable"
        case .deallocated:
            return "Circuit breaker was deallocated"
        }
    }
}

// MARK: - Backend Service Extension

extension BackendService {
    /// Circuit breakers for different endpoints
    public static let pipelineBreaker = CircuitBreaker<Any>(configuration: .default)
    private static let mediaBreaker = CircuitBreaker<Any>(configuration: .lenient)
    private static let analysisBreaker = CircuitBreaker<Any>(configuration: .aggressive)
    
    /// Execute request with circuit breaker protection
    public func executeWithBreaker<T: Decodable>(
        breaker: CircuitBreaker<Any> = pipelineBreaker,
        request: @escaping () -> AnyPublisher<T, Error>
    ) -> AnyPublisher<T, Error> {
        breaker.execute {
            request().map { $0 as Any }.eraseToAnyPublisher()
        }
        .tryMap { output -> T in
            guard let result = output as? T else {
                throw CircuitBreakerError.deallocated
            }
            return result
        }
        .eraseToAnyPublisher()
    }
}

// MARK: - Circuit Breaker Monitor View

public struct CircuitBreakerMonitor: View {
    @ObservedObject var breaker: CircuitBreaker<Any>
    @State private var expanded = false
    
    public init(breaker: CircuitBreaker<Any>) {
        self.breaker = breaker
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack {
                Circle()
                    .fill(colorForState)
                    .frame(width: 10, height: 10)
                
                Text("Circuit Breaker")
                    .font(.system(size: 12, weight: .medium))
                
                Text(breaker.state.description)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Button(action: { expanded.toggle() }) {
                    Image(systemName: expanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
            }
            
            // Statistics
            if expanded {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Requests:")
                        Text("\(breaker.statistics.totalRequests)")
                            .font(.system(.caption, design: .monospaced))
                    }
                    .font(.caption)
                    
                    HStack {
                        Text("Success Rate:")
                        Text("\(Int(breaker.statistics.successRate * 100))%")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(breaker.statistics.successRate > 0.8 ? .green : .orange)
                    }
                    .font(.caption)
                    
                    HStack {
                        Text("Availability:")
                        Text("\(Int(breaker.statistics.availability * 100))%")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(breaker.statistics.availability > 0.9 ? .green : .orange)
                    }
                    .font(.caption)
                    
                    HStack {
                        Text("Failures:")
                        Text("\(breaker.failureCount)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(breaker.failureCount > 0 ? .red : .green)
                    }
                    .font(.caption)
                    
                    // Actions
                    HStack(spacing: 8) {
                        Button("Reset") {
                            breaker.reset()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        
                        Button("Trip") {
                            breaker.trip()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .disabled(breaker.state == .open)
                    }
                    .padding(.top, 4)
                }
                .padding(.leading, 18)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color(white: 0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(colorForState.opacity(0.3), lineWidth: 1)
                )
        )
        .animation(.smooth, value: expanded)
    }
    
    private var colorForState: Color {
        switch breaker.state {
        case .closed: return .green
        case .open: return .red
        case .halfOpen: return .orange
        }
    }
}

extension CircuitBreaker.State: CustomStringConvertible {
    public var description: String {
        switch self {
        case .closed: return "Closed"
        case .open: return "Open"
        case .halfOpen: return "Half-Open"
        }
    }
}
