import Foundation
import Combine

/// Connection manager for backend connectivity
public class ConnectionManager: ObservableObject {
    @Published public var isConnected: Bool = false
    @Published public var connectionState: ConnectionState = .disconnected
    
    public static let shared = ConnectionManager()
    
    private init() {}
    
    public enum ConnectionState {
        case disconnected
        case connecting 
        case connected
        case error(String)
        
        public var isConnected: Bool {
            if case .connected = self {
                return true
            }
            return false
        }
    }
    
    public func connect() async {
        connectionState = .connecting
        // Simulate connection
        try? await 
        connectionState = .connected
        isConnected = true
    }
    
    public func disconnect() {
        connectionState = .disconnected
        isConnected = false
    }
    
    public func checkConnection() async -> Bool {
        return isConnected
    }
    
    public func reconnect() {
        Task {
            await connect()
        }
    }
}
