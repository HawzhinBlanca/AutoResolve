// AUTORESOLVE V3.0 - BACKEND SERVICE CLIENT
// Real-time communication with Python backend

import Foundation
import Combine

// MARK: - Backend Service Client
class BackendService: ObservableObject {
    static let shared = BackendService()
    
    private let baseURL = URL(string: "http://localhost:8081/api")!
    private let session = URLSession.shared
    private var cancellables = Set<AnyCancellable>()
    
    @Published var isConnected = false
    @Published var currentTask: BackendTask?
    @Published var progress: Double = 0.0
    @Published var lastError: String?
    
    init() {
        startHealthCheck()
    }
    
    // MARK: - Health Check
    private func startHealthCheck() {
        Timer.publish(every: 5.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.checkHealth()
            }
            .store(in: &cancellables)
    }
    
    func checkHealth() {
        let url = baseURL.appendingPathComponent("/projects")
        
        session.dataTask(with: url) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode == 200 {
                    self?.isConnected = true
                    self?.lastError = nil
                } else {
                    self?.isConnected = false
                    self?.lastError = error?.localizedDescription ?? "Backend unavailable"
                }
            }
        }.resume()
    }
    
    // MARK: - Pipeline Management
    func startPipeline(inputFile: String, options: [String: Any] = [:]) -> AnyPublisher<PipelineStartResponse, Error> {
        let request = PipelineStartRequest(input_file: inputFile, options: options)
        return performRequest(endpoint: "/pipeline/start", method: "POST", body: request)
    }
    
    func getPipelineStatus(taskId: String) -> AnyPublisher<PipelineStatusResponse, Error> {
        let url = baseURL.appendingPathComponent("/pipeline/status/\(taskId)")
        
        return session.dataTaskPublisher(for: url)
            .map(\.data)
            .decode(type: PipelineStatusResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func cancelPipeline(taskId: String) -> AnyPublisher<CancelResponse, Error> {
        let url = baseURL.appendingPathComponent("/pipeline/cancel/\(taskId)")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        return session.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: CancelResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    // MARK: - Project Management
    func getResolveProjects() -> AnyPublisher<ProjectsResponse, Error> {
        let url = baseURL.appendingPathComponent("/projects")
        
        return session.dataTaskPublisher(for: url)
            .map(\.data)
            .decode(type: ProjectsResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    // MARK: - Presets Management
    func getPresets() -> AnyPublisher<PresetsResponse, Error> {
        let url = baseURL.appendingPathComponent("/presets")
        
        return session.dataTaskPublisher(for: url)
            .map(\.data)
            .decode(type: PresetsResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func savePreset(name: String, settings: [String: Any]) -> AnyPublisher<SavePresetResponse, Error> {
        let request = SavePresetRequest(name: name, settings: settings)
        return performRequest(endpoint: "/presets", method: "POST", body: request)
    }
    
    // MARK: - Configuration Validation
    func validateConfig(inputFile: String) -> AnyPublisher<ValidationResponse, Error> {
        let request = ValidationRequest(input_file: inputFile)
        return performRequest(endpoint: "/validate", method: "POST", body: request)
    }
    
    // MARK: - Generic Request Performer
    private func performRequest<T: Codable, R: Codable>(
        endpoint: String,
        method: String,
        body: T
    ) -> AnyPublisher<R, Error> {
        let url = baseURL.appendingPathComponent(endpoint)
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            return Fail(error: error).eraseToAnyPublisher()
        }
        
        return session.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: R.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
}

// MARK: - Request Models
struct PipelineStartRequest: Codable {
    let input_file: String
    let options: [String: Any]
    
    enum CodingKeys: String, CodingKey {
        case input_file, options
    }
    
    init(input_file: String, options: [String: Any]) {
        self.input_file = input_file
        self.options = options
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        input_file = try container.decode(String.self, forKey: .input_file)
        
        // Handle options as [String: Any]
        if let optionsData = try? container.decode([String: String].self, forKey: .options) {
            options = optionsData
        } else {
            options = [:]
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(input_file, forKey: .input_file)
        
        // Simplified encoding for options
        let stringOptions = options.compactMapValues { $0 as? String }
        try container.encode(stringOptions, forKey: .options)
    }
}

struct ValidationRequest: Codable {
    let input_file: String
}

struct SavePresetRequest: Codable {
    let name: String
    let settings: [String: Any]
    
    enum CodingKeys: String, CodingKey {
        case name, settings
    }
    
    init(name: String, settings: [String: Any]) {
        self.name = name
        self.settings = settings
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        
        if let settingsData = try? container.decode([String: String].self, forKey: .settings) {
            settings = settingsData
        } else {
            settings = [:]
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        
        let stringSettings = settings.compactMapValues { $0 as? String }
        try container.encode(stringSettings, forKey: .settings)
    }
}

// MARK: - Response Models
struct PipelineStartResponse: Codable {
    let task_id: String
}

struct PipelineStatusResponse: Codable {
    let status: String
    let progress: Double?
    let stage: String?
    let message: String?
    let error: String?
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
    let input: ValidationResult
    let broll: ValidationResult
    let resolve: ValidationResult
}

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
struct BackendTask {
    let id: UUID
    let type: TaskType
    let status: TaskStatus
    let progress: Double
    let message: String?
    
    enum TaskType {
        case analysis, transcription, cutGeneration, shortsGeneration, export, silenceDetection
    }
    
    enum TaskStatus {
        case queued, running, completed, failed
    }
}