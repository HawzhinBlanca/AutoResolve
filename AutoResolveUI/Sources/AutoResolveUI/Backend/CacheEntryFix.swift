import Foundation

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
        return CacheEntry(data: data, ttl: ttl > 0 ? ttl : nil)
    }
}
