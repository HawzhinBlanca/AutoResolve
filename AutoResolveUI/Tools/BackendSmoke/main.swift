import Foundation

@main
struct BackendSmoke {
    static func main() async {
        let base = ProcessInfo.processInfo.environment["BACKEND_URL_BASE"] ?? "http://localhost:8000"
        let api = base.hasSuffix("/") ? String(base.dropLast()) : base
        let apiKey = ProcessInfo.processInfo.environment["API_KEY"]
        let session = URLSession(configuration: .ephemeral)

        func get(_ path: String) async throws -> (Int, String) {
            guard let url = URL(string: api + path) else { throw URLError(.badURL) }
            var req = URLRequest(url: url)
            if let key = apiKey { req.setValue(key, forHTTPHeaderField: "x-api-key") }
            let (data, resp) = try await session.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
            let body = String(data: data, encoding: .utf8) ?? ""
            return (code, body)
        }

        do {
            let (hcCode, hcBody) = try await get("/health")
            print("/health ->", hcCode, hcBody.prefix(200))

            let (projCode, projBody) = try await get("/api/projects")
            print("/api/projects ->", projCode, projBody.prefix(200))

            if (200..<300).contains(hcCode) && (200..<300).contains(projCode) {
                print("SMOKE OK")
                exit(0)
            } else {
                print("SMOKE FAIL")
                exit(2)
            }
        } catch {
            print("SMOKE ERROR:", error.localizedDescription)
            exit(1)
        }
    }
}


