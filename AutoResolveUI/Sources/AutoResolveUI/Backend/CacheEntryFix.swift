import Foundation

// Lightweight CacheEntry used by RequestCache when AdvancedCachingLayer is excluded
final class CacheEntry: NSObject {
    let data: Data
    private let expirationDate: Date?
    
    init(data: Data, ttl: TimeInterval? = nil) {
        self.data = data
        self.expirationDate = ttl.map { Date().addingTimeInterval($0) }
    }
    
    var isValid: Bool {
        guard let exp = expirationDate else { return true }
        return Date() < exp
    }
    
    func getValue<T: Codable>(as type: T.Type) -> T? {
        try? JSONDecoder().decode(type, from: data)
    }
}

// Fix for RequestCacheEntry
extension RequestCacheEntry {
    var isValid: Bool { 
        // Check if cache entry is still valid
        return Date() < expiresAt
    }
}

// Fix for CacheEntry conversion
extension RequestCacheEntry {
    func toCacheEntry() -> CacheEntry {
        let ttl = expiresAt.timeIntervalSinceNow
        if ttl <= 0 {
            // Already expired; return an immediate-expiry entry to avoid caching stale data
            return CacheEntry(data: data, ttl: 0)
        }
        return CacheEntry(data: data, ttl: ttl)
    }
}
