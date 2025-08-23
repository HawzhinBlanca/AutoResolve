import Foundation
import os

enum Performance {
    static let log = OSLog(subsystem: "com.autoresolve.ui", category: "Performance")

    @discardableResult
    static func begin(_ name: StaticString) -> OSSignpostID {
        let id = OSSignpostID(log: log)
        if #available(macOS 10.14, *) {
            os_signpost(.begin, log: log, name: name, signpostID: id)
        }
        return id
    }

    static func end(_ id: OSSignpostID, _ name: StaticString) {
        if #available(macOS 10.14, *) {
            os_signpost(.end, log: log, name: name, signpostID: id)
        }
    }
}



