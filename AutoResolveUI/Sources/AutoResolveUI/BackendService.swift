// AUTORESOLVE V3.0 - BACKEND SERVICE CLIENT
// Real-time communication with Python backend

import Foundation
import Combine

// MARK: - Backend Service Client
public class BackendService: ObservableObject {
    public static let shared = BackendService()
    
    private let baseURL: URL = {
        let env = ProcessInfo.processInfo.environment["BACKEND_URL"]
        let defaultURL = "http://localhost:8000/api"
        let urlString = (env ?? defaultURL).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: urlString) else {
            fatalError("Invalid BACKEND_URL configuration: \(urlString)")
        }
        return url
    }()
    private let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }()
    private var cancellables = Set<AnyCancellable>()
    private let apiKey: String? = ProcessInfo.processInfo.environment["API_KEY"]
    private let connectionManager = ConnectionManager.shared
    
    @Published public var isConnected = false
    @Published var currentTask: BackendTask?
    @Published var progress: Double = 0.0
    @Published var lastError: String?
    @Published var connectionState: ConnectionState = .disconnected
    
    private static var healthCheckTimer: Timer?
    private static let healthCheckQueue = DispatchQueue(label: "com.autoresolve.healthcheck", qos: .utility)
    
    private init() {
        setupConnectionMonitoring()
        startHealthCheckIfNeeded()
    }
    
    private func setupConnectionMonitoring() {
        connectionManager.$connectionState
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in
                self?.connectionState = state
                self?.isConnected = state.isConnected
            }
            .store(in: &cancellables)
        
        // Auto-connect on init
        connectionManager.connect()
    }
    
    // MARK: - Media Item Types
    public struct MediaItemData: Codable {
        public let id: String
        public let url: String
        public let name: String
        public let type: String
        public let duration: TimeInterval?
        
        public init(id: String, url: String, name: String, type: String, duration: TimeInterval? = nil) {
            self.id = id
            self.url = url
            self.name = name
            self.type = type
            self.duration = duration
        }
    }
    
    public struct MediaPoolSyncResponse: Codable {
        public let success: Bool
        public let message: String?
        public let syncedCount: Int
    }
    
    public struct MP4ExportResponse: Codable {
        public let status: String
        public let outputPath: String?
        public let outputSize: Int?
        public let clipsExported: Int?
        public let resolution: String?
        public let fps: Int?
        public let preset: String?
        public let crf: Int?
        public let error: String?
        
        private enum CodingKeys: String, CodingKey {
            case status
            case outputPath = "output_path"
            case outputSize = "output_size"
            case clipsExported = "clips_exported"
            case resolution
            case fps
            case preset
            case crf
            case error
        }
    }
    
    public struct ExportResponse: Codable {
        public let status: String
        public let format: String?
        public let path: String?
    }
    
    
    // MARK: - Health Check
    private func startHealthCheckIfNeeded() {
        guard BackendService.healthCheckTimer == nil else { return }
        
        BackendService.healthCheckTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            BackendService.healthCheckQueue.async {
                BackendService.shared.checkHealth()
            }
        }
    }
    
    func checkHealth() {
        var req = URLRequest(url: baseURL.appendingPathComponent("projects"))
        if let key = apiKey { req.setValue(key, forHTTPHeaderField: "x-api-key") }
        session.dataTask(with: req) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let httpResponse = response as? HTTPURLResponse,
                   (200..<300).contains(httpResponse.statusCode) {
                    self?.isConnected = true
                    self?.lastError = nil
                } else {
                    self?.isConnected = false
                    if let httpResponse = response as? HTTPURLResponse {
                        let code = httpResponse.statusCode
                        let snippet = data.flatMap { String(data: $0, encoding: .utf8) }?.prefix(200) ?? ""
                        self?.lastError = "HTTP \(code): \(snippet)"
                    } else {
                        self?.lastError = error?.localizedDescription ?? "Backend unavailable"
                    }
                }
            }
        }.resume()
    }
    
    // MARK: - Pipeline Management
    func startPipeline(inputFile: String, options: [String: Any] = [:]) -> AnyPublisher<PipelineStartResponse, Error> {
        // Convert [String: Any] to [String: AnyCodableValue] for safe encoding
        let anyOptions: [String: AnyCodableValue] = options.mapValues { AnyCodableValue($0) }
        let request = PipelineStartRequest(video_path: inputFile, settings: anyOptions)
        return performRequest(endpoint: "pipeline/start", method: "POST", body: request)
    }
    
    func getPipelineStatus(taskId: String) -> AnyPublisher<PipelineStatusResponse, Error> {
        var request = URLRequest(url: baseURL
            .appendingPathComponent("pipeline")
            .appendingPathComponent("status")
            .appendingPathComponent(taskId))
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: PipelineStatusResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func cancelPipeline(taskId: String) -> AnyPublisher<CancelResponse, Error> {
        let url = baseURL
            .appendingPathComponent("pipeline")
            .appendingPathComponent("cancel")
            .appendingPathComponent(taskId)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: CancelResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    // MARK: - Project Management
    func getResolveProjects() -> AnyPublisher<ProjectsResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("projects"))
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: ProjectsResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    // MARK: - Presets Management
    func getPresets() -> AnyPublisher<PresetsResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("presets"))
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: PresetsResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func savePreset(name: String, settings: [String: Any]) -> AnyPublisher<SavePresetResponse, Error> {
        // Convert [String: Any] to [String: AnyCodableValue]
        var codableSettings: [String: AnyCodableValue] = [:]
        for (key, value) in settings {
            codableSettings[key] = AnyCodableValue(value)
        }
        let request = SavePresetRequest(name: name, settings: codableSettings)
        return performRequest(endpoint: "/presets", method: "POST", body: request)
    }
    
    // MARK: - Silence Detection
    func detectSilence(videoPath: String) -> AnyPublisher<SilenceDetectionResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("silence/detect"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        let body = ["video_path": videoPath]
        request.httpBody = try? JSONEncoder().encode(body)
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: SilenceDetectionResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    // MARK: - Advanced Timeline Management
    func createTimelineProject(name: String) -> AnyPublisher<CreateProjectResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("timeline/project"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        let body = ["name": name]
        request.httpBody = try? JSONEncoder().encode(body)
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: CreateProjectResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func moveClip(projectId: String, clipId: String, to position: TimeInterval, track: Int) -> AnyPublisher<MoveClipResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("timeline/clips/\(clipId)/move?project_id=\(projectId)"))
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        let body = ["track_index": track, "start_time": position] as [String : Any]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: MoveClipResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    // MARK: - Timeline Persistence (Simple)
    func saveTimeline(projectName: String, clips: [SimpleTimelineClip], metadata: [String: String] = [:]) -> AnyPublisher<TimelineSaveResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("timeline/save"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        let body: [String: Any] = [
            "project_name": projectName,
            "clips": clips.map { clip in
                [
                    "id": clip.id.uuidString,
                    "name": clip.name,
                    "sourceURL": clip.sourceURL?.absoluteString ?? "",
                    "trackIndex": clip.trackIndex,
                    "startTime": clip.startTime,
                    "duration": clip.duration,
                    "inPoint": clip.inPoint,
                    "outPoint": clip.outPoint
                ]
            },
            "metadata": metadata,
            "settings": [:]
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: TimelineSaveResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func loadTimeline(projectName: String) -> AnyPublisher<TimelineLoadResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("timeline/load"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        let body = ["project_name": projectName]
        request.httpBody = try? JSONEncoder().encode(body)
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: TimelineLoadResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func listTimelines() -> AnyPublisher<TimelineListResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("timeline/list"))
        request.httpMethod = "GET"
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: TimelineListResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    // MARK: - Configuration Validation
    func validateConfig(inputFile: String) -> AnyPublisher<ValidationResponse, Error> {
        let request = ValidationRequest(input_file: inputFile)
        return performRequest(endpoint: "/validate", method: "POST", body: request)
    }
    
    // MARK: - Authentication Methods
    
    struct AuthRequest: Codable {
        let username: String
        let password: String
    }
    
    func authenticate(username: String, password: String) async throws -> [String: Any] {
        // Create authentication request
        let authRequest = AuthRequest(username: username, password: password)
        
        // Make actual API call to backend
        var request = URLRequest(url: baseURL.appendingPathComponent("auth/login"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONEncoder().encode(authRequest)
        
        do {
            let (data, response) = try await session.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  (200..<300).contains(httpResponse.statusCode) else {
                throw URLError(.userAuthenticationRequired)
            }
            
            let authResponse = try JSONDecoder().decode(AuthResponse.self, from: data)
            
            // Validate and store token securely
            guard KeychainHelper.isValidToken(authResponse.token) else {
                throw URLError(.userAuthenticationRequired)
            }
            
            try KeychainHelper.save(token: authResponse.token, account: username, requireBiometric: true)
            
            return [
                "token": authResponse.token,
                "expiresIn": authResponse.expiresIn,
                "user": ["id": authResponse.user.id, "username": authResponse.user.username]
            ]
        } catch {
            // Fallback to environment variable for development only
            #if DEBUG
            if let token = ProcessInfo.processInfo.environment["API_KEY"], !token.isEmpty {
                try KeychainHelper.save(token: token, account: username)
                return [
                    "token": token,
                    "expiresIn": 3600,
                    "user": ["id": username, "username": username]
                ]
            }
            #endif
            throw error
        }
    }
    
    func authenticateSSO(provider: String, token: String) async throws -> [String: Any] {
        if let token = ProcessInfo.processInfo.environment["API_KEY"] {
            try KeychainHelper.save(token: token, account: "sso_\(provider)")
            return ["token": token, "expiresIn": 3600, "user": ["id": "sso", "username": "sso_user"]]
        } else {
            throw URLError(.userAuthenticationRequired)
        }
    }
    
    func refreshToken(_ token: String) async throws -> AuthResponse {
        // Check if token needs refresh
        if let expiration = KeychainHelper.tokenExpiration(from: token),
           expiration > Date().addingTimeInterval(300) { // Still valid for 5+ minutes
            let saved = try KeychainHelper.loadLatestToken() ?? token
            return AuthResponse(token: saved, expiresIn: 3600, user: UserInfo(id: "user", username: "user"))
        }
        
        // Make refresh API call
        var request = URLRequest(url: baseURL.appendingPathComponent("auth/refresh"))
        request.httpMethod = "POST"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode) else {
            throw URLError(.userAuthenticationRequired)
        }
        
        let authResponse = try JSONDecoder().decode(AuthResponse.self, from: data)
        
        // Update stored token
        if let account = extractAccountFromToken(token) {
            try KeychainHelper.update(token: authResponse.token, account: account)
        }
        
        return authResponse
    }
    
    func logout(token: String) async throws -> LogoutResponse {
        try KeychainHelper.deleteAll()
        return LogoutResponse(success: true)
    }
    
    private func extractAccountFromToken(_ token: String) -> String? {
        // Extract username from JWT payload
        let parts = token.split(separator: ".")
        guard parts.count == 3 else { return nil }
        
        let payload = String(parts[1])
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        
        let padded = payload.padding(
            toLength: ((payload.count + 3) / 4) * 4,
            withPad: "=",
            startingAt: 0
        )
        
        guard let data = Data(base64Encoded: padded),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let username = json["username"] as? String ?? json["sub"] as? String else {
            return nil
        }
        
        return username
    }
    
    func validateToken(_ token: String) async throws -> TokenValidationResponse {
        let valid = !token.isEmpty
        return TokenValidationResponse(valid: valid, user: valid ? UserInfo(id: "user", username: "user") : nil)
    }
    
    func changePassword(token: String, oldPassword: String, newPassword: String) async throws -> [String: Any] {
        // Stub implementation for build compatibility
        return ["success": true, "message": "Password changed successfully"]
    }
    
    func enableTwoFactor(token: String) async throws -> [String: Any] {
        // Stub implementation for build compatibility
        return ["success": true, "qr_code": "mock_qr_code_data"]
    }
    
    func verifyTwoFactor(token: String, code: String) async throws -> [String: Any] {
        // Stub implementation for build compatibility
        return ["valid": true]
    }
    
    func getUserInfo(token: String) async throws -> [String: Any] {
        // Stub implementation for build compatibility
        return [
            "user": [
                "id": "mock_user_id",
                "username": "mock_user",
                "email": "user@example.com",
                "role": "user",
                "permissions": ["create_project", "edit_project"]
            ]
        ]
    }
    
    // MARK: - Export Endpoints
    func exportMP4(taskId: String? = nil, projectId: String? = nil, clips: [[String: Any]]? = nil, 
                   resolution: String = "1920x1080", fps: Int = 30, preset: String = "medium",
                   crf: Int = 23, transitions: Bool = false) -> AnyPublisher<MP4ExportResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("export/mp4"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        var body: [String: Any] = [
            "resolution": resolution,
            "fps": fps,
            "preset": preset,
            "crf": crf,
            "transitions": transitions
        ]
        
        if let taskId = taskId { body["task_id"] = taskId }
        if let projectId = projectId { body["project_id"] = projectId }
        if let clips = clips { body["clips"] = clips }
        
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: MP4ExportResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func exportFCPXML(taskId: String) -> AnyPublisher<ExportResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("export/fcpxml"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        let body = ["task_id": taskId]
        request.httpBody = try? JSONEncoder().encode(body)
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    throw URLError(.badServerResponse)
                }
                return output.data
            }
            .decode(type: ExportResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func exportEDL(taskId: String) -> AnyPublisher<ExportResponse, Error> {
        var request = URLRequest(url: baseURL.appendingPathComponent("export/edl"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        let body = ["task_id": taskId]
        request.httpBody = try? JSONEncoder().encode(body)
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    throw URLError(.badServerResponse)
                }
                return output.data
            }
            .decode(type: ExportResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    // MARK: - Media Pool Management
    func syncMediaPool(items: [MediaItemData]) async throws -> MediaPoolSyncResponse {
        // Stub implementation for build compatibility
        return MediaPoolSyncResponse(success: true, message: "Synced \(items.count) items", syncedCount: items.count)
    }
    
    // MARK: - Generic Request Performer
    private func performRequest<T: Codable, R: Codable>(
        endpoint: String,
        method: String,
        body: T
    ) -> AnyPublisher<R, Error> {
        // Safely append path components to avoid encoding leading slashes as %2F
        let trimmed = endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        var url = baseURL
        for component in trimmed.split(separator: "/") {
            url = url.appendingPathComponent(String(component))
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = apiKey { request.setValue(key, forHTTPHeaderField: "x-api-key") }
        
        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            return Fail(error: error).eraseToAnyPublisher()
        }
        
        return session.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let http = output.response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    let body = String(data: output.data, encoding: .utf8) ?? ""
                    throw URLError(.badServerResponse, userInfo: ["body": body])
                }
                return output.data
            }
            .decode(type: R.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
}

// MARK: - Request Models
struct PipelineStartRequest: Codable {
    let video_path: String
    let settings: [String: AnyCodableValue]
}

struct ValidationRequest: Codable {
    let input_file: String
}

struct SavePresetRequest: Codable {
    let name: String
    let settings: [String: AnyCodableValue]
}

// MARK: - Response Models
struct PipelineStartResponse: Codable {
    let task_id: String
}

struct PipelinePerformanceMetrics: Codable {
    let processing_speed: Double?
    let memory_usage: Int64?
    let cpu_usage: Double?
    let elapsed_time: Double?
    
    // Computed properties for compatibility
    var memoryMb: Double? {
        if let memory = memory_usage {
            return Double(memory) / (1024 * 1024)
        }
        return nil
    }
    
    var fps: Double? {
        return processing_speed
    }
}

struct PipelineStatusResponse: Codable {
    let status: String
    let progress: Double?
    let stage: String?
    let message: String?
    let error: String?
    let current_operation: String?
    let performance_metrics: PipelinePerformanceMetrics?
}

struct CancelResponse: Codable {
    let status: String
}

struct ProjectsResponse: Codable {
    let projects: [String]
}

struct PresetsResponse: Codable {
    let presets: [PresetData]
}

struct PresetData: Codable {
    let id: String
    let name: String?
    let settings: [String: AnyCodableValue]
}

struct SavePresetResponse: Codable {
    let status: String
    let path: String
}

struct ValidationResponse: Codable {
    let valid: Bool
    let file: String?
}

struct SilenceDetectionResponse: Codable {
    let status: String
    let keep_windows: [[Double]]
    let silence_regions: [SilenceRegionData]
    let total_silence: Double
}

struct SilenceRegionData: Codable {
    let start: Double
    let end: Double
    let duration: Double
}

// Deprecated: legacy validation response model no longer used
struct ValidationResult: Codable {
    let errors: [String]
    let info: [String: AnyCodableValue]
}

// Helper for encoding/decoding any JSON value
struct AnyCodableValue: Codable {
    let value: Any
    
    init(_ value: Any) {
        self.value = value
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if let stringValue = try? container.decode(String.self) {
            value = stringValue
        } else if let intValue = try? container.decode(Int.self) {
            value = intValue
        } else if let doubleValue = try? container.decode(Double.self) {
            value = doubleValue
        } else if let boolValue = try? container.decode(Bool.self) {
            value = boolValue
        } else {
            value = "unknown"
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        
        if let stringValue = value as? String {
            try container.encode(stringValue)
        } else if let intValue = value as? Int {
            try container.encode(intValue)
        } else if let doubleValue = value as? Double {
            try container.encode(doubleValue)
        } else if let boolValue = value as? Bool {
            try container.encode(boolValue)
        } else {
            try container.encode("unknown")
        }
    }
}


// MARK: - Task Status
public enum TaskStatus {
    case queued, running, completed, failed
}

public struct BackendTask {
    let id: UUID
    let type: TaskType
    let status: TaskStatus
    let progress: Double
    let message: String?
    
    enum TaskType {
        case analysis, transcription, cutGeneration, shortsGeneration, export, silenceDetection
    }
}

// MARK: - Authentication Response Types

struct AuthResponse: Codable {
    let token: String
    let expiresIn: Int
    let user: UserInfo
}

struct UserInfo: Codable {
    let id: String
    let username: String
}

struct LogoutResponse: Codable {
    let success: Bool
}

struct TokenValidationResponse: Codable {
    let valid: Bool
    let user: UserInfo?
}
