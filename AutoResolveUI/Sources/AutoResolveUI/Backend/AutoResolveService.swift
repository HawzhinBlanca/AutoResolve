import Foundation

class AutoResolveService {
    private let baseURL = URL(string: "http://localhost:8000/api")!

    func transcribe(videoURL: URL, completion: @escaping (Bool) -> Void) {
        let url = baseURL.appendingPathComponent("pipeline/start")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120 // Set a longer timeout

        let body = [
            "video_path": videoURL.path
        ]

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [])
        } catch {
            print("Error serializing request body: \(error)")
            completion(false)
            return
        }

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error starting transcription: \(error)")
                completion(false)
                return
            }

            guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                print("Invalid response from server")
                completion(false)
                return
            }

            completion(true)
        }.resume()
    }
}