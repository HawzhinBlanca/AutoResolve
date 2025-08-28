import SwiftUI

extension Binding where Value == TimeInterval {
    init(get: @escaping () -> TimeInterval, set: @escaping (TimeInterval) -> Void) {
        self.init(
            get: get,
            set: set
        )
    }
}

extension Binding {
    func toTimeInterval() -> TimeInterval where Value == Double {
        wrappedValue
    }
}
