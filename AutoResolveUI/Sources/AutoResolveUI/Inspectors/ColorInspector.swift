
import SwiftUI

struct ColorInspector: View {
    @State private var lift: Double = 0
    @State private var gamma: Double = 1
    @State private var gain: Double = 1

    private let backendService = BackendService()

    var body: some View {
        VStack {
            Text("Color Inspector").font(.headline)
            Slider(value: $lift, in: -1...1, step: 0.01)
            Text("Lift: \(lift, specifier: "%.2f")")
            Slider(value: $gamma, in: 0...2, step: 0.01)
            Text("Gamma: \(gamma, specifier: "%.2f")")
            Slider(value: $gain, in: 0...2, step: 0.01)
            Text("Gain: \(gain, specifier: "%.2f")")

            Button("Apply Color Grade") {
                let gradeData = [
                    "lift": ["r": lift, "g": lift, "b": lift],
                    "gamma": ["r": gamma, "g": gamma, "b": gamma],
                    "gain": ["r": gain, "g": gain, "b": gain]
                ]
                backendService.applyColorGrade(clipId: "clip_1", gradeData: gradeData) { success in
                    if success {
                        print("Color grade applied successfully")
                    } else {
                        print("Failed to apply color grade")
                    }
                }
            }
        }
        .padding()
    }
}
