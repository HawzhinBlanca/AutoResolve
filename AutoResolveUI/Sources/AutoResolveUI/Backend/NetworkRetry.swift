// AUTORESOLVE V3.0 - NETWORK RETRY LOGIC
// Robust retry mechanism for API calls

import Foundation
import Combine

/// Provides retry capabilities for network requests
public extension Publisher where Failure == Error {
    
    /// Retry with exponential backoff
    func retryWithBackoff(
        retries: Int = 3,
        initialDelay: TimeInterval = 1.0,
        maxDelay: TimeInterval = 30.0
    ) -> AnyPublisher<Output, Failure> {
        self.catch { error -> AnyPublisher<Output, Failure> in
            guard retries > 0 else {
                return Fail(error: error).eraseToAnyPublisher()
            }
            
            let delay = Swift.min(initialDelay * pow(2.0, Double(3 - retries)), maxDelay)
            
            return Just(())
                .delay(for: .seconds(delay), scheduler: DispatchQueue.main)
                .flatMap { _ -> AnyPublisher<Output, Failure> in
                    self.retryWithBackoff(
                        retries: retries - 1,
                        initialDelay: initialDelay,
                        maxDelay: maxDelay
                    )
                }
                .eraseToAnyPublisher()
        }
        .eraseToAnyPublisher()
    }
    
    /// Retry only on specific error conditions
    func retryOnConnectionError(
        retries: Int = 3,
        delay: TimeInterval = 1.0
    ) -> AnyPublisher<Output, Failure> {
        self.catch { error -> AnyPublisher<Output, Failure> in
            guard retries > 0,
                  isRetryableError(error) else {
                return Fail(error: error).eraseToAnyPublisher()
            }
            
            return Just(())
                .delay(for: .seconds(delay), scheduler: DispatchQueue.main)
                .flatMap { _ -> AnyPublisher<Output, Failure> in
                    self.retryOnConnectionError(retries: retries - 1, delay: delay)
                }
                .eraseToAnyPublisher()
        }
        .eraseToAnyPublisher()
    }
}

/// Check if error is retryable
private func isRetryableError(_ error: Error) -> Bool {
    if let urlError = error as? URLError {
        switch urlError.code {
        case .timedOut,
             .cannotFindHost,
             .cannotConnectToHost,
             .networkConnectionLost,
             .dnsLookupFailed,
             .notConnectedToInternet:
            return true
        default:
            return false
        }
    }
    return false
}

/// Request wrapper with built-in retry and timeout
public class RobustRequest<T: Decodable> {
    private let url: URL
    private let method: String
    private let body: Data?
    private let headers: [String: String]
    private let session: URLSession
    private let retries: Int
    private let timeout: TimeInterval
    
    public init(
        url: URL,
        method: String = "GET",
        body: Data? = nil,
        headers: [String: String] = [:],
        session: URLSession = .shared,
        retries: Int = 3,
        timeout: TimeInterval = 30
    ) {
        self.url = url
        self.method = method
        self.body = body
        self.headers = headers
        self.session = session
        self.retries = retries
        self.timeout = timeout
    }
    
    public func execute() -> AnyPublisher<T, Error> {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = body
        request.timeoutInterval = timeout
        
        for (key, value) in headers {
            request.setValue(value, forHTTPHeaderField: key)
        }
        
        if body != nil && request.value(forHTTPHeaderField: "Content-Type") == nil {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        
        return session.dataTaskPublisher(for: request)
            .mapError { $0 as Error }
            .retryOnConnectionError(retries: retries) // retry only on network errors before decode
            .tryMap { output -> Data in
                guard let httpResponse = output.response as? HTTPURLResponse else {
                    throw NetworkError.invalidResponse
                }
                
                switch httpResponse.statusCode {
                case 200..<300:
                    return output.data
                case 401:
                    throw NetworkError.unauthorized
                case 404:
                    throw NetworkError.notFound
                case 500..<600:
                    throw NetworkError.serverError(httpResponse.statusCode)
                default:
                    throw NetworkError.httpError(httpResponse.statusCode)
                }
            }
            .decode(type: T.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
}

/// Network-specific errors
public enum NetworkError: LocalizedError {
    case invalidResponse
    case unauthorized
    case notFound
    case serverError(Int)
    case httpError(Int)
    case decodingError(Error)
    
    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from server"
        case .unauthorized:
            return "Unauthorized request"
        case .notFound:
            return "Resource not found"
        case .serverError(let code):
            return "Server error (\(code))"
        case .httpError(let code):
            return "HTTP error (\(code))"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        }
    }
}
