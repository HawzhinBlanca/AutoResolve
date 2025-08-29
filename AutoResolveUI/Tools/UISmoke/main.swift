import Foundation

@main
struct UISmoke {
    static func main() async {
        let base = ProcessInfo.processInfo.environment["BACKEND_URL_BASE"] ?? "http://localhost:8000"
        let apiKey = ProcessInfo.processInfo.environment["API_KEY"]
        let client = UIClient(baseURL: base, apiKey: apiKey)

        do {
            let ok = try await client.health()
            print("UI->Backend health:", ok)

            let started = try await client.startPipeline(videoPath: ProcessInfo.processInfo.environment["VIDEO_PATH"] ?? "/Users/hawzhin/AutoResolve/autorez/assets/test_media/test_video_5min.mp4")
            print("Pipeline started:", started)

            if let taskId = started {
                // Query status once
                let status = try await client.status(taskId: taskId)
                print("Status:", status.prefix(200))
            }

            print("UI SMOKE OK")
            exit(0)
        } catch {
            print("UI SMOKE FAIL:", error.localizedDescription)
            exit(1)
        }
    }
}

final class UIClient {
    let apiBase: String
    let apiKey: String?
    let session: URLSession = .shared

    init(baseURL: String, apiKey: String?) {
        self.apiBase = baseURL.hasSuffix("/api") ? baseURL : baseURL + "/api"
        self.apiKey = apiKey
    }

    func health() async throws -> Bool {
        let url = URL(string: apiBase.replacingOccurrences(of: "/api", with: "") + "/health")!
        var req = URLRequest(url: url)
        if let apiKey { req.setValue(apiKey, forHTTPHeaderField: "x-api-key") }
        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse else { return false }
        let body = String(data: data, encoding: .utf8) ?? ""
        print("/health ->", http.statusCode, body.prefix(200))
        return (200..<300).contains(http.statusCode)
    }

    func startPipeline(videoPath: String) async throws -> String? {
        let url = URL(string: apiBase + "/pipeline/start")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey { req.setValue(apiKey, forHTTPHeaderField: "x-api-key") }
        let body: [String: Any] = ["video_path": videoPath, "settings": [:]]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let s = String(data: data, encoding: .utf8) ?? ""
            throw NSError(domain: "UISmoke", code: 2, userInfo: [NSLocalizedDescriptionKey: "start failed: \(s)"])
        }
        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return json["task_id"] as? String
        }
        return nil
    }

    func status(taskId: String) async throws -> String {
        let url = URL(string: apiBase + "/pipeline/status/" + taskId)!
        var req = URLRequest(url: url)
        if let apiKey { req.setValue(apiKey, forHTTPHeaderField: "x-api-key") }
        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let s = String(data: data, encoding: .utf8) ?? ""
            throw NSError(domain: "UISmoke", code: 3, userInfo: [NSLocalizedDescriptionKey: "status failed: \(s)"])
        }
        return String(data: data, encoding: .utf8) ?? ""
    }
}


