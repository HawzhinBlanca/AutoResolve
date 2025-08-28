// AUTORESOLVE V3.0 - HARDENED WEBSOCKET MANAGER
// Robust WebSocket connection with auto-reconnect and message queuing

import Foundation
import Combine
import SwiftUI

/// Manages WebSocket connections with automatic reconnection and message buffering
public class WebSocketManager: NSObject, ObservableObject {
    public static let shared = WebSocketManager()
    
    // Connection state
    @Published public var isConnected = false
    @Published public var connectionState: WSConnectionState = .disconnected
    @Published public var lastError: Error?
    @Published public var messageCount = 0
    
    // WebSocket
    private var webSocket: URLSessionWebSocketTask?
    private var session: URLSession!
    private let url = URL(string: "ws://localhost:8000/ws/progress")!
    
    // Reconnection
    private var reconnectTimer: Timer?
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 10
    private let baseReconnectDelay: TimeInterval = 1.0
    private let maxReconnectDelay: TimeInterval = 60.0
    
    // Message handling
    private var messageQueue: [WSMessage] = []
    private var messageHandlers: [String: (WSMessage) -> Void] = [:]
    private var pingTimer: Timer?
    private let pingInterval: TimeInterval = 30.0
    
    // Publishers
    private let messageSubject = PassthroughSubject<WSMessage, Never>()
    public var messagePublisher: AnyPublisher<WSMessage, Never> {
        messageSubject.eraseToAnyPublisher()
    }
    
    override init() {
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: .main)
    }
    
    // MARK: - Connection Management
    
    public func connect() {
        guard connectionState != .connected && connectionState != .connecting else { return }
        
        connectionState = .connecting
        reconnectAttempts = 0
        
        createWebSocket()
        startReceiving()
        startPinging()
    }
    
    public func disconnect() {
        connectionState = .disconnecting
        
        stopPinging()
        cancelReconnect()
        
        webSocket?.cancel(with: .goingAway, reason: nil)
        webSocket = nil
        
        connectionState = .disconnected
        isConnected = false
    }
    
    private func createWebSocket() {
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        
        webSocket = session.webSocketTask(with: request)
        webSocket?.resume()
        
        // Send queued messages
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.flushMessageQueue()
        }
    }
    
    // MARK: - Message Handling
    
    public func send(_ message: WSMessage) {
        guard let webSocket = webSocket else {
            // Queue message if not connected
            messageQueue.append(message)
            return
        }
        
        do {
            let data = try JSONEncoder().encode(message)
            let wsMessage = URLSessionWebSocketTask.Message.data(data)
            
            webSocket.send(wsMessage) { [weak self] error in
                if let error = error {
                    print("[WebSocket] Send error: \(error)")
                    self?.handleConnectionError(error)
                    // Re-queue message
                    self?.messageQueue.append(message)
                } else {
                    self?.messageCount += 1
                }
            }
        } catch {
            print("[WebSocket] Encoding error: \(error)")
        }
    }
    
    public func subscribe(to type: String, handler: @escaping (WSMessage) -> Void) {
        messageHandlers[type] = handler
    }
    
    private func startReceiving() {
        webSocket?.receive { [weak self] result in
            guard let self = self else { return }
            
            switch result {
            case .success(let message):
                self.handleMessage(message)
                self.startReceiving() // Continue receiving
                
            case .failure(let error):
                print("[WebSocket] Receive error: \(error)")
                self.handleConnectionError(error)
            }
        }
    }
    
    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .data(let data):
            do {
                let wsMessage = try JSONDecoder().decode(WSMessage.self, from: data)
                messageSubject.send(wsMessage)
                
                // Call specific handlers
                if let handler = messageHandlers[wsMessage.type] {
                    handler(wsMessage)
                }
                
                // Update connection state on first message
                if connectionState != .connected {
                    connectionState = .connected
                    isConnected = true
                    reconnectAttempts = 0
                }
            } catch {
                print("[WebSocket] Decode error: \(error)")
            }
            
        case .string(let text):
            if let data = text.data(using: .utf8) {
                handleMessage(.data(data))
            }
            
        @unknown default:
            break
        }
    }
    
    // MARK: - Connection Health
    
    private func startPinging() {
        stopPinging()
        
        pingTimer = Timer.scheduledTimer(withTimeInterval: pingInterval, repeats: true) { [weak self] _ in
            self?.sendPing()
        }
    }
    
    private func stopPinging() {
        pingTimer?.invalidate()
        pingTimer = nil
    }
    
    private func sendPing() {
        webSocket?.sendPing { [weak self] error in
            if let error = error {
                print("[WebSocket] Ping failed: \(error)")
                self?.handleConnectionError(error)
            }
        }
    }
    
    // MARK: - Error Handling & Reconnection
    
    private func handleConnectionError(_ error: Error) {
        lastError = error
        
        if connectionState == .connected || connectionState == .connecting {
            connectionState = .reconnecting
            isConnected = false
            scheduleReconnect()
        }
    }
    
    private func scheduleReconnect() {
        guard reconnectAttempts < maxReconnectAttempts else {
            connectionState = .failed
            return
        }
        
        cancelReconnect()
        
        let delay = min(
            baseReconnectDelay * pow(2.0, Double(reconnectAttempts)),
            maxReconnectDelay
        )
        
        reconnectAttempts += 1
        
        reconnectTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            print("[WebSocket] Reconnect attempt \(self?.reconnectAttempts ?? 0)")
            self?.connect()
        }
    }
    
    private func cancelReconnect() {
        reconnectTimer?.invalidate()
        reconnectTimer = nil
    }
    
    private func flushMessageQueue() {
        let queue = messageQueue
        messageQueue.removeAll()
        
        for message in queue {
            send(message)
        }
    }
    
    // MARK: - Cleanup
    
    deinit {
        disconnect()
    }
}

// MARK: - URLSession Delegate

extension WebSocketManager: URLSessionWebSocketDelegate {
    public func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        print("[WebSocket] Connected")
        connectionState = .connected
        isConnected = true
        reconnectAttempts = 0
        flushMessageQueue()
    }
    
    public func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        print("[WebSocket] Closed with code: \(closeCode.rawValue)")
        
        if connectionState != .disconnecting {
            // Unexpected close, try to reconnect
            connectionState = .reconnecting
            isConnected = false
            scheduleReconnect()
        }
    }
    
    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("[WebSocket] Task error: \(error)")
            handleConnectionError(error)
        }
    }
}

// MARK: - Supporting Types

public enum WSConnectionState: Equatable {
    case disconnected
    case connecting
    case connected
    case reconnecting
    case disconnecting
    case failed
    
    public var description: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .reconnecting: return "Reconnecting..."
        case .disconnecting: return "Disconnecting..."
        case .failed: return "Connection Failed"
        }
    }
    
    public var isActive: Bool {
        self == .connected || self == .connecting || self == .reconnecting
    }
}

public struct WSMessage: Codable {
    public let id: String
    public let type: String
    public let payload: [String: AnyCodable]
    public let timestamp: Date
    
    public init(type: String, payload: [String: AnyCodable]) {
        self.id = UUID().uuidString
        self.type = type
        self.payload = payload
        self.timestamp = Date()
    }
}

// MARK: - AnyCodable for flexible message payloads

public struct AnyCodable: Codable {
    public let value: Any
    
    public init(_ value: Any) {
        self.value = value
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dictionary = try? container.decode([String: AnyCodable].self) {
            value = dictionary.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        
        switch value {
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodable($0) })
        case let dictionary as [String: Any]:
            try container.encode(dictionary.mapValues { AnyCodable($0) })
        default:
            try container.encodeNil()
        }
    }
}

// MARK: - WebSocket Status View

public struct WebSocketStatusView: View {
    @ObservedObject private var wsManager = WebSocketManager.shared
    @State private var showDetails = false
    
    public var body: some View {
        HStack(spacing: 6) {
            // Status indicator
            Circle()
                .fill(colorForState)
                .frame(width: 8, height: 8)
                .overlay(
                    Circle()
                        .fill(colorForState.opacity(0.3))
                        .frame(width: 16, height: 16)
                        .scaleEffect(wsManager.connectionState == .connected ? 1.5 : 1)
                        .opacity(wsManager.connectionState == .connected ? 0 : 1)
                        .animation(.easeOut(duration: 1).repeatForever(autoreverses: false), value: wsManager.connectionState)
                )
            
            // Status text
            Text("WS: \(wsManager.connectionState.description)")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.secondary)
            
            // Message count
            if wsManager.messageCount > 0 {
                Text("(\(wsManager.messageCount))")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.secondary.opacity(0.7))
            }
        }
        .onTapGesture {
            showDetails.toggle()
        }
        .popover(isPresented: $showDetails) {
            WebSocketDetailsView()
                .frame(width: 250, height: 150)
        }
    }
    
    private var colorForState: Color {
        switch wsManager.connectionState {
        case .connected: return .green
        case .connecting, .reconnecting: return .orange
        case .disconnected, .disconnecting: return .gray
        case .failed: return .red
        }
    }
}

struct WebSocketDetailsView: View {
    @ObservedObject private var wsManager = WebSocketManager.shared
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("WebSocket Status")
                .font(.system(size: 14, weight: .semibold))
            
            HStack {
                Text("State:")
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
                Text(wsManager.connectionState.description)
                    .font(.system(size: 12, weight: .medium))
            }
            
            HStack {
                Text("Messages:")
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
                Text("\(wsManager.messageCount)")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
            }
            
            if let error = wsManager.lastError {
                Text("Error: \(error.localizedDescription)")
                    .font(.system(size: 11))
                    .foregroundColor(.red)
                    .lineLimit(2)
            }
            
            Spacer()
            
            HStack(spacing: 8) {
                Button(action: { wsManager.connect() }) {
                    Text("Connect")
                        .font(.system(size: 11, weight: .medium))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(4)
                }
                .disabled(wsManager.connectionState.isActive)
                
                Button(action: { wsManager.disconnect() }) {
                    Text("Disconnect")
                        .font(.system(size: 11, weight: .medium))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                        .background(Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(4)
                }
                .disabled(!wsManager.connectionState.isActive)
            }
        }
        .padding()
    }
}
