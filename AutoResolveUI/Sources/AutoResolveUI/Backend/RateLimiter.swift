// AUTORESOLVE V3.0 - RATE LIMITING & THROTTLING
// Prevent API abuse and manage resource consumption

import Foundation
import SwiftUI
import Combine

/// Advanced rate limiter with token bucket algorithm
public class RateLimiter: ObservableObject {
    
    // MARK: - Configuration
    public struct Configuration {
        let maxTokens: Int              // Maximum tokens in bucket
        let refillRate: Double          // Tokens per second
        let burstCapacity: Int          // Extra capacity for bursts
        let penaltyMultiplier: Double   // Penalty for violations
        
        public static let `default` = Configuration(
            maxTokens: 100,
            refillRate: 10,
            burstCapacity: 20,
            penaltyMultiplier: 2.0
        )
        
        public static let strict = Configuration(
            maxTokens: 50,
            refillRate: 5,
            burstCapacity: 10,
            penaltyMultiplier: 3.0
        )
        
        public static let relaxed = Configuration(
            maxTokens: 200,
            refillRate: 20,
            burstCapacity: 50,
            penaltyMultiplier: 1.5
        )
    }
    
    // MARK: - State
    private var tokens: Double
    private var lastRefillTime: Date
    private var violations = 0
    private let configuration: Configuration
    private let queue = DispatchQueue(label: "rate.limiter", attributes: .concurrent)
    
    @Published public var statistics = Statistics()
    
    public struct Statistics {
        var totalRequests = 0
        var allowedRequests = 0
        var throttledRequests = 0
        var violations = 0
        var currentTokens: Double = 0
        
        var allowanceRate: Double {
            guard totalRequests > 0 else { return 1.0 }
            return Double(allowedRequests) / Double(totalRequests)
        }
    }
    
    public init(configuration: Configuration = .default) {
        self.configuration = configuration
        self.tokens = Double(configuration.maxTokens)
        self.lastRefillTime = Date()
    }
    
    // MARK: - Public Methods
    
    /// Check if request should be allowed
    public func shouldAllow(cost: Double = 1.0) -> Bool {
        queue.sync(flags: .barrier) {
            refillTokens()
            statistics.totalRequests += 1
            statistics.currentTokens = tokens
            
            if tokens >= cost {
                tokens -= cost
                statistics.allowedRequests += 1
                
                logDebug("[RateLimiter] Request allowed (tokens: \(tokens))", category: .network)
                return true
            } else {
                statistics.throttledRequests += 1
                violations += 1
                statistics.violations = violations
                
                logWarning("[RateLimiter] Request throttled (tokens: \(tokens), violations: \(violations))", category: .network)
                return false
            }
        }
    }
    
    /// Execute with rate limiting
    public func execute<T>(
        cost: Double = 1.0,
        request: @escaping () -> AnyPublisher<T, Error>
    ) -> AnyPublisher<T, Error> {
        guard shouldAllow(cost: cost) else {
            return Fail(error: RateLimitError.throttled(retryAfter: timeUntilTokensAvailable(cost)))
                .eraseToAnyPublisher()
        }
        
        return request()
    }
    
    /// Execute with automatic retry after throttling
    public func executeWithRetry<T>(
        cost: Double = 1.0,
        maxRetries: Int = 3,
        request: @escaping () -> AnyPublisher<T, Error>
    ) -> AnyPublisher<T, Error> {
        executeWithRetryInternal(
            cost: cost,
            retriesLeft: maxRetries,
            request: request
        )
    }
    
    /// Reset rate limiter
    public func reset() {
        queue.async(flags: .barrier) {
            self.tokens = Double(self.configuration.maxTokens)
            self.lastRefillTime = Date()
            self.violations = 0
            self.statistics = Statistics()
            
            logInfo("[RateLimiter] Reset", category: .network)
        }
    }
    
    /// Get time until tokens available
    public func timeUntilTokensAvailable(_ required: Double) -> TimeInterval {
        queue.sync {
            refillTokens()
            
            if tokens >= required {
                return 0
            }
            
            let tokensNeeded = required - tokens
            return tokensNeeded / configuration.refillRate
        }
    }
    
    // MARK: - Private Methods
    
    private func refillTokens() {
        let now = Date()
        let timePassed = now.timeIntervalSince(lastRefillTime)
        let tokensToAdd = timePassed * configuration.refillRate
        
        if tokensToAdd > 0 {
            let maxCapacity = Double(configuration.maxTokens + configuration.burstCapacity)
            tokens = min(tokens + tokensToAdd, maxCapacity)
            lastRefillTime = now
            
            // Decay violations over time
            if violations > 0 && timePassed > 60 {
                violations = max(0, violations - 1)
            }
        }
    }
    
    private func executeWithRetryInternal<T>(
        cost: Double,
        retriesLeft: Int,
        request: @escaping () -> AnyPublisher<T, Error>
    ) -> AnyPublisher<T, Error> {
        if shouldAllow(cost: cost) {
            return request()
                .catch { error -> AnyPublisher<T, Error> in
                    // Only retry on rate limit errors
                    if case RateLimitError.throttled = error, retriesLeft > 0 {
                        let delay = self.timeUntilTokensAvailable(cost)
                        return Just(())
                            .delay(for: .seconds(delay), scheduler: DispatchQueue.main)
                            .flatMap { _ in
                                self.executeWithRetryInternal(
                                    cost: cost,
                                    retriesLeft: retriesLeft - 1,
                                    request: request
                                )
                            }
                            .eraseToAnyPublisher()
                    }
                    return Fail(error: error).eraseToAnyPublisher()
                }
                .eraseToAnyPublisher()
        } else if retriesLeft > 0 {
            let delay = timeUntilTokensAvailable(cost)
            return Just(())
                .delay(for: .seconds(delay), scheduler: DispatchQueue.main)
                .flatMap { _ in
                    self.executeWithRetryInternal(
                        cost: cost,
                        retriesLeft: retriesLeft - 1,
                        request: request
                    )
                }
                .eraseToAnyPublisher()
        } else {
            return Fail(error: RateLimitError.throttled(retryAfter: timeUntilTokensAvailable(cost)))
                .eraseToAnyPublisher()
        }
    }
}

// MARK: - Error Types

public enum RateLimitError: LocalizedError {
    case throttled(retryAfter: TimeInterval)
    
    public var errorDescription: String? {
        switch self {
        case .throttled(let retryAfter):
            return String(format: "Rate limited. Retry after %.1f seconds", retryAfter)
        }
    }
}

// MARK: - Global Rate Limiters

public class RateLimitManager {
    public static let shared = RateLimitManager()
    
    // Different limiters for different resources
    public let apiLimiter = RateLimiter(configuration: .default)
    public let mediaLimiter = RateLimiter(configuration: .relaxed)
    public let analysisLimiter = RateLimiter(configuration: .strict)
    
    private init() {}
    
    /// Get appropriate limiter for endpoint
    public func limiter(for endpoint: String) -> RateLimiter {
        if endpoint.contains("media") || endpoint.contains("upload") {
            return mediaLimiter
        } else if endpoint.contains("analysis") || endpoint.contains("pipeline") {
            return analysisLimiter
        } else {
            return apiLimiter
        }
    }
    
    /// Reset all limiters
    public func resetAll() {
        apiLimiter.reset()
        mediaLimiter.reset()
        analysisLimiter.reset()
    }
}

// MARK: - Rate Limit Monitor View

public struct RateLimitMonitor: View {
    @ObservedObject var limiter: RateLimiter
    let name: String
    @State private var expanded = false
    
    public init(name: String, limiter: RateLimiter) {
        self.name = name
        self.limiter = limiter
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                // Status indicator
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                
                Text(name)
                    .font(.system(size: 12, weight: .medium))
                
                // Token gauge
                ProgressView(value: limiter.statistics.currentTokens, total: 100)
                    .frame(width: 60)
                    .tint(statusColor)
                
                Text("\(Int(limiter.statistics.currentTokens))")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Button(action: { expanded.toggle() }) {
                    Image(systemName: expanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
            }
            
            if expanded {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Allowance:")
                        Text("\(Int(limiter.statistics.allowanceRate * 100))%")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(limiter.statistics.allowanceRate > 0.9 ? .green : .orange)
                    }
                    .font(.caption)
                    
                    HStack {
                        Text("Throttled:")
                        Text("\(limiter.statistics.throttledRequests)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(limiter.statistics.throttledRequests > 0 ? .orange : .green)
                    }
                    .font(.caption)
                    
                    HStack {
                        Text("Violations:")
                        Text("\(limiter.statistics.violations)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(limiter.statistics.violations > 0 ? .red : .green)
                    }
                    .font(.caption)
                    
                    Button("Reset") {
                        limiter.reset()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .padding(.top, 4)
                }
                .padding(.leading, 14)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color(white: 0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(statusColor.opacity(0.3), lineWidth: 1)
                )
        )
        .animation(.smooth, value: expanded)
    }
    
    private var statusColor: Color {
        let tokens = limiter.statistics.currentTokens
        if tokens > 50 {
            return .green
        } else if tokens > 20 {
            return .orange
        } else {
            return .red
        }
    }
}
