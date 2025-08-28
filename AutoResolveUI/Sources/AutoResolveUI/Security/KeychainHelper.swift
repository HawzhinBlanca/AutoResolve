// AUTORESOLVE V3.0 - SECURE KEYCHAIN HELPER
// Production-ready token storage with biometric authentication support

import Foundation
import Security
import LocalAuthentication

/// Secure token storage using iOS/macOS Keychain
public final class KeychainHelper {
    
    // MARK: - Constants
    private static let serviceName = "com.autoresolve.app"
    private static let accessGroup = "com.autoresolve.shared"
    
    public enum KeychainError: LocalizedError {
        case itemNotFound
        case duplicateItem
        case invalidData
        case authenticationFailed
        case biometricNotAvailable
        case unexpectedError(OSStatus)
        
        public var errorDescription: String? {
            switch self {
            case .itemNotFound:
                return "Token not found in keychain"
            case .duplicateItem:
                return "Token already exists"
            case .invalidData:
                return "Invalid token data"
            case .authenticationFailed:
                return "Authentication failed"
            case .biometricNotAvailable:
                return "Biometric authentication not available"
            case .unexpectedError(let status):
                return "Keychain error: \(status)"
            }
        }
    }
    
    // MARK: - Public Methods
    
    /// Save token securely with optional biometric protection
    public static func save(token: String, account: String, requireBiometric: Bool = false) throws {
        guard let data = token.data(using: .utf8) else {
            throw KeychainError.invalidData
        }
        
        // Delete existing item if present
        try? delete(account: account)
        
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        // Add biometric protection if requested
        if requireBiometric {
            let access = try createBiometricAccess()
            query[kSecAttrAccessControl as String] = access
        }
        
        let status = SecItemAdd(query as CFDictionary, nil)
        
        guard status == errSecSuccess else {
            if status == errSecDuplicateItem {
                throw KeychainError.duplicateItem
            }
            throw KeychainError.unexpectedError(status)
        }
    }
    
    /// Load token from keychain
    public static func load(account: String) throws -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecReturnData as String: true
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess else {
            if status == errSecItemNotFound {
                throw KeychainError.itemNotFound
            }
            throw KeychainError.unexpectedError(status)
        }
        
        guard let data = result as? Data,
              let token = String(data: data, encoding: .utf8) else {
            throw KeychainError.invalidData
        }
        
        return token
    }
    
    /// Load the most recent token (for any account)
    public static func loadLatestToken() throws -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecMatchLimit as String: kSecMatchLimitAll,
            kSecReturnAttributes as String: true,
            kSecReturnData as String: true
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess,
              let items = result as? [[String: Any]] else {
            return nil
        }
        
        // Sort by modification date and get the most recent
        let sorted = items.sorted { item1, item2 in
            let date1 = item1[kSecAttrModificationDate as String] as? Date ?? Date.distantPast
            let date2 = item2[kSecAttrModificationDate as String] as? Date ?? Date.distantPast
            return date1 > date2
        }
        
        guard let mostRecent = sorted.first,
              let data = mostRecent[kSecValueData as String] as? Data,
              let token = String(data: data, encoding: .utf8) else {
            return nil
        }
        
        return token
    }
    
    /// Update existing token
    public static func update(token: String, account: String) throws {
        guard let data = token.data(using: .utf8) else {
            throw KeychainError.invalidData
        }
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account
        ]
        
        let update: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrModificationDate as String: Date()
        ]
        
        let status = SecItemUpdate(query as CFDictionary, update as CFDictionary)
        
        guard status == errSecSuccess else {
            if status == errSecItemNotFound {
                // Item doesn't exist, create it
                try save(token: token, account: account)
            } else {
                throw KeychainError.unexpectedError(status)
            }
            return
        }
    }
    
    /// Delete token from keychain
    public static func delete(account: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unexpectedError(status)
        }
    }
    
    /// Delete all tokens
    public static func deleteAll() throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unexpectedError(status)
        }
    }
    
    /// Check if biometric authentication is available
    public static func isBiometricAvailable() -> Bool {
        let context = LAContext()
        var error: NSError?
        return context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error)
    }
    
    // MARK: - Private Methods
    
    private static func createBiometricAccess() throws -> SecAccessControl {
        guard isBiometricAvailable() else {
            throw KeychainError.biometricNotAvailable
        }
        
        var error: Unmanaged<CFError>?
        guard let access = SecAccessControlCreateWithFlags(
            kCFAllocatorDefault,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            .biometryCurrentSet,
            &error
        ) else {
            if let error = error?.takeRetainedValue() {
                throw KeychainError.unexpectedError(OSStatus(error._code))
            }
            throw KeychainError.unexpectedError(-1)
        }
        
        return access
    }
    
    /// Validate token format (basic JWT validation)
    public static func isValidToken(_ token: String) -> Bool {
        // Basic JWT format validation (three base64 parts separated by dots)
        let parts = token.split(separator: ".")
        guard parts.count == 3 else { return false }
        
        // Check if each part is valid base64
        for part in parts {
            let base64 = String(part)
                .replacingOccurrences(of: "-", with: "+")
                .replacingOccurrences(of: "_", with: "/")
            
            let padded = base64.padding(
                toLength: ((base64.count + 3) / 4) * 4,
                withPad: "=",
                startingAt: 0
            )
            
            guard Data(base64Encoded: padded) != nil else {
                return false
            }
        }
        
        return true
    }
    
    /// Extract expiration from JWT token
    public static func tokenExpiration(from token: String) -> Date? {
        let parts = token.split(separator: ".")
        guard parts.count == 3 else { return nil }
        
        let payload = String(parts[1])
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        
        let padded = payload.padding(
            toLength: ((payload.count + 3) / 4) * 4,
            withPad: "=",
            startingAt: 0
        )
        
        guard let data = Data(base64Encoded: padded),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let exp = json["exp"] as? TimeInterval else {
            return nil
        }
        
        return Date(timeIntervalSince1970: exp)
    }
}

// MARK: - Token Manager Extension

extension KeychainHelper {
    
    /// Comprehensive token management
    public struct TokenManager {
        
        public struct Token {
            let value: String
            let account: String
            let expiration: Date?
            let isValid: Bool
            
            var isExpired: Bool {
                guard let expiration = expiration else { return false }
                return expiration < Date()
            }
        }
        
        /// Save and validate token
        public static func saveToken(_ token: String, for account: String) throws {
            guard KeychainHelper.isValidToken(token) else {
                throw KeychainHelper.KeychainError.invalidData
            }
            
            try KeychainHelper.save(token: token, account: account)
        }
        
        /// Load and validate token
        public static func loadToken(for account: String) throws -> Token {
            let value = try KeychainHelper.load(account: account)
            let expiration = KeychainHelper.tokenExpiration(from: value)
            let isValid = KeychainHelper.isValidToken(value)
            
            return Token(
                value: value,
                account: account,
                expiration: expiration,
                isValid: isValid
            )
        }
        
        /// Refresh token if needed
        public static func refreshIfNeeded(account: String, refreshHandler: (String) async throws -> String) async throws -> Token {
            let token = try loadToken(for: account)
            
            if token.isExpired || !token.isValid {
                let newTokenValue = try await refreshHandler(token.value)
                try saveToken(newTokenValue, for: account)
                return try loadToken(for: account)
            }
            
            return token
        }
    }
}