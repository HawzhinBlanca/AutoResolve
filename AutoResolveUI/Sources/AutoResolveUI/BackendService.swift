// AUTORESOLVE V3.0 - BACKEND SERVICE CLIENT
// Real-time communication with Python backend

import Foundation
import Combine

// MARK: - Backend Service Client
class BackendService: ObservableObject {
    static let shared = BackendService()
    
    private let baseURL = URL(string: "http://localhost:8000/api")!
    private let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        return URLSession(configuration: config)
    }()
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
        let url = baseURL.appendingPathComponent("projects")
        
        session.dataTask(with: url) { [weak self] data, response, error in
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
        let request = PipelineStartRequest(input_file: inputFile, options: anyOptions)
        return performRequest(endpoint: "pipeline/start", method: "POST", body: request)
    }
    
    func getPipelineStatus(taskId: String) -> AnyPublisher<PipelineStatusResponse, Error> {
        let url = baseURL
            .appendingPathComponent("pipeline")
            .appendingPathComponent("status")
            .appendingPathComponent(taskId)
        
        return session.dataTaskPublisher(for: url)
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
        let url = baseURL.appendingPathComponent("projects")
        
        return session.dataTaskPublisher(for: url)
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
        let url = baseURL.appendingPathComponent("presets")
        
        return session.dataTaskPublisher(for: url)
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
        // Safely append path components to avoid encoding leading slashes as %2F
        let trimmed = endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        var url = baseURL
        for component in trimmed.split(separator: "/") {
            url = url.appendingPathComponent(String(component))
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
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
    let input_file: String
    let options: [String: AnyCodableValue]
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

struct PipelineStatusResponse: Codable {
    let status: String
    let progress: Double?
    let stage: String?
    let message: String?
    let error: String?
    let current_operation: String?
    let performance_metrics: BackendPerformanceMetrics?
}

public struct BackendPerformanceMetrics: Codable {
    let processing_time: Double?
    let realtime_factor: Double?
    let memory_usage: Double?
    let cpu_usage: Double?
    let memory_mb: Double?
    let fps: Double?
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
public struct BackendTask {
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