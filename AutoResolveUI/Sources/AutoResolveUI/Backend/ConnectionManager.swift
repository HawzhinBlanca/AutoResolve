// AUTORESOLVE V3.0 - ROBUST CONNECTION MANAGER
// Handles backend connectivity with retry logic and graceful degradation

import Foundation
import Combine
import Network

/// Manages backend connection with automatic retry and health monitoring
public class ConnectionManager: ObservableObject {
    public static let shared = ConnectionManager()
    
    // Connection state
    @Published public var connectionState: ConnectionState = .disconnected
    @Published public var healthStatus: HealthStatus?
    @Published public var retryCount = 0
    @Published public var lastError: Error?
    @Published public var networkAvailable = true
    
    // Configuration
    private let maxRetries = 5
    private let baseRetryDelay: TimeInterval = 1.0
    private let maxRetryDelay: TimeInterval = 30.0
    private let healthCheckInterval: TimeInterval = 5.0
    
    // Network monitoring
    private let monitor = NWPathMonitor()
    private let monitorQueue = DispatchQueue(label: "network.monitor")
    
    // Timers and cancellables
    private var healthCheckTimer: Timer?
    private var retryTimer: Timer?
    private var cancellables = Set<AnyCancellable>()
    
    // URLs
    private let baseURL = URL(string: "http://localhost:8000")!
    private var healthURL: URL { baseURL.appendingPathComponent("health") }
    
    // Session with custom configuration
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        config.timeoutIntervalForResource = 30
        config.waitsForConnectivity = true
        config.allowsConstrainedNetworkAccess = true
        config.allowsExpensiveNetworkAccess = true
        return URLSession(configuration: config)
    }()
    
    private init() {
        setupNetworkMonitoring()
        startHealthChecking()
    }
    
    // MARK: - Public Methods
    
    /// Start connection with automatic retry
    public func connect() {
        guard networkAvailable else {
            connectionState = .error(ConnectionError.noNetwork)
            return
        }
        
        connectionState = .connecting
        retryCount = 0
        attemptConnection()
    }
    
    /// Disconnect and stop health checks
    public func disconnect() {
        connectionState = .disconnected
        healthCheckTimer?.invalidate()
        retryTimer?.invalidate()
    }
    
    /// Force reconnection
    public func reconnect() {
        disconnect()
        Thread.sleep(forTimeInterval: 0.5)
        connect()
    }
    
    // MARK: - Private Methods
    
    private func setupNetworkMonitoring() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.networkAvailable = path.status == .satisfied
                
                if path.status == .satisfied && self?.connectionState == .error(ConnectionError.noNetwork) {
                    self?.connect()
                } else if path.status != .satisfied {
                    self?.connectionState = .error(ConnectionError.noNetwork)
                }
            }
        }
        monitor.start(queue: monitorQueue)
    }
    
    private func startHealthChecking() {
        healthCheckTimer?.invalidate()
        healthCheckTimer = Timer.scheduledTimer(withTimeInterval: healthCheckInterval, repeats: true) { [weak self] _ in
            self?.checkHealth()
        }
    }
    
    private func attemptConnection() {
        guard retryCount < maxRetries else {
            connectionState = .error(ConnectionError.maxRetriesExceeded)
            return
        }
        
        checkHealth { [weak self] success in
            guard let self = self else { return }
            
            if success {
                self.connectionState = .connected
                self.retryCount = 0
                self.startHealthChecking()
            } else {
                self.scheduleRetry()
            }
        }
    }
    
    private func scheduleRetry() {
        retryCount += 1
        let delay = min(baseRetryDelay * pow(2.0, Double(retryCount - 1)), maxRetryDelay)
        
        connectionState = .reconnecting(attempt: retryCount, maxAttempts: maxRetries)
        
        retryTimer?.invalidate()
        retryTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            self?.attemptConnection()
        }
    }
    
    private func checkHealth(completion: ((Bool) -> Void)? = nil) {
        let request = URLRequest(url: healthURL, cachePolicy: .reloadIgnoringLocalCacheData)
        
        session.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                
                if let error = error {
                    self.lastError = error
                    self.healthStatus = nil
                    completion?(false)
                    
                    if self.connectionState == .connected {
                        self.connectionState = .error(ConnectionError.connectionLost)
                        self.scheduleRetry()
                    }
                    return
                }
                
                guard let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode == 200,
                      let data = data else {
                    completion?(false)
                    
                    if self.connectionState == .connected {
                        self.connectionState = .error(ConnectionError.invalidResponse)
                        self.scheduleRetry()
                    }
                    return
                }
                
                do {
                    let health = try JSONDecoder().decode(HealthStatus.self, from: data)
                    self.healthStatus = health
                    
                    if self.connectionState != .connected {
                        self.connectionState = .connected
                    }
                    completion?(true)
                } catch {
                    self.lastError = error
                    completion?(false)
                    
                    if self.connectionState == .connected {
                        self.connectionState = .error(ConnectionError.decodingError)
                        self.scheduleRetry()
                    }
                }
            }
        }.resume()
    }
}

// MARK: - Supporting Types

public enum ConnectionState: Equatable {
    case disconnected
    case connecting
    case connected
    case reconnecting(attempt: Int, maxAttempts: Int)
    case error(ConnectionError)
    
    public var isConnected: Bool {
        self == .connected
    }
    
    public var description: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .reconnecting(let attempt, let max): return "Reconnecting (\(attempt)/\(max))..."
        case .error(let error): return "Error: \(error.localizedDescription)"
        }
    }
}

public enum ConnectionError: LocalizedError, Equatable {
    case noNetwork
    case connectionLost
    case maxRetriesExceeded
    case invalidResponse
    case decodingError
    case timeout
    
    public var errorDescription: String? {
        switch self {
        case .noNetwork: return "No network connection"
        case .connectionLost: return "Connection to backend lost"
        case .maxRetriesExceeded: return "Maximum retry attempts exceeded"
        case .invalidResponse: return "Invalid response from backend"
        case .decodingError: return "Failed to decode response"
        case .timeout: return "Connection timeout"
        }
    }
}

public struct HealthStatus: Codable {
    public let status: String
    public let pipeline: String
    public let memory_mb: Int
    public let active_tasks: Int
    
    public var isHealthy: Bool {
        status == "healthy" && pipeline == "ready"
    }
}
