// AUTORESOLVE V3.0 - AUTHENTICATION MANAGER
// Enterprise-grade JWT authentication with multi-factor auth and SSO support

import Foundation
import Security
import CryptoKit
import LocalAuthentication
import Combine

// MARK: - Authentication Manager Protocol
protocol AuthenticationManagerProtocol: AnyObject {
    var isAuthenticated: Bool { get }
    var currentUser: User? { get }
    var authToken: String? { get }
    
    func authenticate(username: String, password: String) async throws -> AuthResult
    func authenticateWithBiometrics() async throws -> AuthResult
    func authenticateWithSSO(provider: SSOProvider) async throws -> AuthResult
    func refreshToken() async throws -> String
    func logout() async throws
    func validateToken(_ token: String) async throws -> Bool
    func changePassword(oldPassword: String, newPassword: String) async throws
    func enableTwoFactor() async throws -> String // Returns QR code data
    func verifyTwoFactor(code: String) async throws -> Bool
}

// MARK: - Authentication Manager Implementation
@MainActor
final class AuthenticationManager: AuthenticationManagerProtocol, ObservableObject {
    
    // MARK: - Published Properties
    @Published private(set) var isAuthenticated = false
    @Published private(set) var currentUser: User?
    @Published private(set) var authToken: String?
    @Published var authState: AuthState = .unauthenticated
    @Published var twoFactorEnabled = false
    @Published var biometricsEnabled = false
    @Published var lastAuthMethod: AuthMethod = .none
    @Published var sessionExpiresAt: Date?
    
    enum AuthState {
        case unauthenticated
        case authenticating
        case authenticated
        case expired
        case locked
        case error(String)
    }
    
    enum AuthMethod {
        case none
        case password
        case biometrics
        case sso(SSOProvider)
        case twoFactor
    }
    
    // MARK: - Private Properties
    private let keychain = KeychainManager()
    private let tokenValidator = JWTTokenValidator()
    private let biometricsManager = BiometricsManager()
    private let ssoManager = SSOManager()
    private let backendService: BackendService
    private var refreshTimer: Timer?
    private var sessionTimer: Timer?
    
    // Token refresh
    private let tokenRefreshQueue = DispatchQueue(label: "com.autoresolve.auth.refresh", qos: .userInitiated)
    private var isRefreshingToken = false
    
    // Rate limiting
    private var authAttempts: [Date] = []
    private let maxAuthAttempts = 5
    private let authAttemptWindow: TimeInterval = 900 // 15 minutes
    
    // MARK: - Initialization
    init(backendService: BackendService = BackendService.shared) {
        self.backendService = backendService
        
        // Check for existing authentication
        Task {
            await checkExistingAuth()
        }
        
        // Setup automatic token refresh
        setupTokenRefresh()
        
        // Setup session monitoring
        setupSessionMonitoring()
    }
    
    deinit {
        refreshTimer?.invalidate()
        sessionTimer?.invalidate()
    }
    
    // MARK: - Public Methods
    
    func authenticate(username: String, password: String) async throws -> AuthResult {
        // Rate limiting check
        guard !isRateLimited() else {
            throw AuthError.rateLimited
        }
        
        authState = .authenticating
        recordAuthAttempt()
        
        do {
            // Hash password before sending
            let hashedPassword = hashPassword(password)
            
            // Send authentication request
            let response = try await backendService.authenticate(
                username: username,
                password: hashedPassword
            )
            
            let result = try parseAuthResponse(response)
            
            // Handle successful authentication
            await handleSuccessfulAuth(result, method: .password)
            
            return result
            
        } catch {
            authState = .error(error.localizedDescription)
            throw error
        }
    }
    
    func authenticateWithBiometrics() async throws -> AuthResult {
        guard biometricsEnabled else {
            throw AuthError.biometricsNotEnabled
        }
        
        authState = .authenticating
        
        do {
            // Verify biometrics
            let biometricResult = try await biometricsManager.authenticate()
            guard biometricResult else {
                throw AuthError.biometricsAuthFailed
            }
            
            // Get stored credentials
            guard let storedToken = try keychain.getToken() else {
                throw AuthError.noStoredCredentials
            }
            
            // Validate stored token
            if try await validateToken(storedToken) {
                let user = try await getUserInfo(token: storedToken)
                let result = AuthResult(
                    token: storedToken,
                    refreshToken: try keychain.getRefreshToken(),
                    user: user,
                    expiresAt: tokenValidator.getExpirationDate(storedToken) ?? Date().addingTimeInterval(3600)
                )
                
                await handleSuccessfulAuth(result, method: .biometrics)
                return result
            } else {
                throw AuthError.tokenExpired
            }
            
        } catch {
            authState = .error(error.localizedDescription)
            throw error
        }
    }
    
    func authenticateWithSSO(provider: SSOProvider) async throws -> AuthResult {
        authState = .authenticating
        
        do {
            let ssoResult = try await ssoManager.authenticate(with: provider)
            
            // Exchange SSO token for our JWT
            let response = try await backendService.authenticateSSO(
                provider: provider.rawValue,
                token: ssoResult.token
            )
            
            let result = try parseAuthResponse(response)
            await handleSuccessfulAuth(result, method: .sso(provider))
            
            return result
            
        } catch {
            authState = .error(error.localizedDescription)
            throw error
        }
    }
    
    func refreshToken() async throws -> String {
        guard !isRefreshingToken else {
            // Wait for ongoing refresh
            while isRefreshingToken {
                try await Task.sleep(nanoseconds: 100_000_000) // 0.1s
            }
            return authToken ?? ""
        }
        
        isRefreshingToken = true
        defer { isRefreshingToken = false }
        
        guard let refreshToken = try keychain.getRefreshToken() else {
            throw AuthError.noRefreshToken
        }
        
        do {
            let response = try await backendService.refreshToken(refreshToken)
            let newToken = response.token
            // AuthResponse doesn't have refresh_token field
            let newRefreshToken: String? = nil
            
            // Store new tokens
            try keychain.storeToken(newToken)
            if let newRefreshToken = newRefreshToken {
                try keychain.storeRefreshToken(newRefreshToken)
            }
            
            authToken = newToken
            
            // Update expiration
            sessionExpiresAt = tokenValidator.getExpirationDate(newToken)
            
            return newToken
            
        } catch {
            // Refresh failed, logout user
            try await logout()
            throw AuthError.refreshFailed
        }
    }
    
    func logout() async throws {
        // Invalidate server session
        if let token = authToken {
            try? await backendService.logout(token: token)
        }
        
        // Clear local storage
        try keychain.clearTokens()
        
        // Reset state
        authState = .unauthenticated
        isAuthenticated = false
        currentUser = nil
        authToken = nil
        sessionExpiresAt = nil
        lastAuthMethod = .none
        
        // Invalidate timers
        refreshTimer?.invalidate()
        sessionTimer?.invalidate()
    }
    
    func validateToken(_ token: String) async throws -> Bool {
        do {
            // Local validation first
            if !tokenValidator.isValid(token) {
                return false
            }
            
            // Server validation
            let response = try await backendService.validateToken(token)
            return response.valid
            
        } catch {
            return false
        }
    }
    
    func changePassword(oldPassword: String, newPassword: String) async throws {
        guard let token = authToken else {
            throw AuthError.notAuthenticated
        }
        
        let hashedOldPassword = hashPassword(oldPassword)
        let hashedNewPassword = hashPassword(newPassword)
        
        try await backendService.changePassword(token: token, oldPassword: hashedOldPassword, newPassword: hashedNewPassword)
    }
    
    func enableTwoFactor() async throws -> String {
        guard let token = authToken else {
            throw AuthError.notAuthenticated
        }
        
        let response = try await backendService.enableTwoFactor(token: token)
        let qrCodeData = (response["qr_code"] as? String) ?? ""
        
        twoFactorEnabled = true
        return qrCodeData
    }
    
    func verifyTwoFactor(code: String) async throws -> Bool {
        guard let token = authToken else {
            throw AuthError.notAuthenticated
        }
        
        let response = try await backendService.verifyTwoFactor(token: token, code: code)
        return (response["valid"] as? Bool) ?? false
    }
    
    // MARK: - Private Methods
    
    private func checkExistingAuth() async {
        do {
            if let storedToken = try keychain.getToken() {
                if try await validateToken(storedToken) {
                    authToken = storedToken
                    currentUser = try await getUserInfo(token: storedToken)
                    isAuthenticated = true
                    authState = .authenticated
                    sessionExpiresAt = tokenValidator.getExpirationDate(storedToken)
                    
                    // Setup refresh timer
                    setupTokenRefresh()
                } else {
                    // Token invalid, clear it
                    try keychain.clearTokens()
                }
            }
        } catch {
            // Clear any corrupted stored tokens
            try? keychain.clearTokens()
        }
    }
    
    private func handleSuccessfulAuth(_ result: AuthResult, method: AuthMethod) async {
        // Store tokens securely
        do {
            try keychain.storeToken(result.token)
            if let refreshToken = result.refreshToken {
                try keychain.storeRefreshToken(refreshToken)
            }
        } catch {
            print("Failed to store tokens: \(error)")
        }
        
        // Update state
        authToken = result.token
        currentUser = result.user
        isAuthenticated = true
        authState = .authenticated
        lastAuthMethod = method
        sessionExpiresAt = result.expiresAt
        
        // Clear auth attempts
        authAttempts.removeAll()
        
        // Setup token refresh
        setupTokenRefresh()
        
        // Log authentication
        await logAuthEvent(method: method, success: true)
    }
    
    private func parseAuthResponse(_ response: [String: Any]) throws -> AuthResult {
        guard let token = response["token"] as? String else {
            throw AuthError.invalidResponse
        }
        
        let refreshToken = response["refresh_token"] as? String
        
        guard let userData = response["user"] as? [String: Any],
              let userId = userData["id"] as? String,
              let username = userData["username"] as? String else {
            throw AuthError.invalidUserData
        }
        
        let user = User(
            id: UUID(uuidString: userId) ?? UUID(),
            username: username,
            email: userData["email"] as? String ?? "",
            role: UserRole(rawValue: userData["role"] as? String ?? "user") ?? .user,
            permissions: parsePermissions(userData["permissions"] as? [String] ?? [])
        )
        
        let expiresAt = tokenValidator.getExpirationDate(token) ?? Date().addingTimeInterval(3600)
        
        return AuthResult(
            token: token,
            refreshToken: refreshToken,
            user: user,
            expiresAt: expiresAt
        )
    }
    
    private func parsePermissions(_ permissionStrings: [String]) -> Set<Permission> {
        Set(permissionStrings.compactMap { Permission(rawValue: $0) })
    }
    
    private func getUserInfo(token: String) async throws -> User {
        let response = try await backendService.getUserInfo(token: token)
        
        guard let userData = response["user"] as? [String: Any],
              let userId = userData["id"] as? String,
              let username = userData["username"] as? String else {
            throw AuthError.invalidUserData
        }
        
        return User(
            id: UUID(uuidString: userId) ?? UUID(),
            username: username,
            email: userData["email"] as? String ?? "",
            role: UserRole(rawValue: userData["role"] as? String ?? "user") ?? .user,
            permissions: parsePermissions(userData["permissions"] as? [String] ?? [])
        )
    }
    
    private func setupTokenRefresh() {
        refreshTimer?.invalidate()
        
        guard let expiresAt = sessionExpiresAt else { return }
        
        // Refresh 5 minutes before expiration
        let refreshTime = expiresAt.addingTimeInterval(-300)
        let timeUntilRefresh = refreshTime.timeIntervalSinceNow
        
        if timeUntilRefresh > 0 {
            refreshTimer = Timer.scheduledTimer(withTimeInterval: timeUntilRefresh, repeats: false) { [weak self] _ in
                Task {
                    try? await self?.refreshToken()
                }
            }
        }
    }
    
    private func setupSessionMonitoring() {
        sessionTimer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.checkSessionExpiration()
            }
        }
    }
    
    private func checkSessionExpiration() {
        guard let expiresAt = sessionExpiresAt else { return }
        
        if Date() >= expiresAt {
            authState = .expired
            Task {
                try? await logout()
            }
        }
    }
    
    private func isRateLimited() -> Bool {
        let now = Date()
        
        // Remove old attempts
        authAttempts = authAttempts.filter { now.timeIntervalSince($0) < authAttemptWindow }
        
        return authAttempts.count >= maxAuthAttempts
    }
    
    private func recordAuthAttempt() {
        authAttempts.append(Date())
    }
    
    private func hashPassword(_ password: String) -> String {
        let data = Data(password.utf8)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
    
    private func logAuthEvent(method: AuthMethod, success: Bool) async {
        // Log authentication events for audit
        print("Auth event: \(method), success: \(success)")
        
        // In production, send to audit log
        // await auditLogger.log(event: .authentication, method: method, success: success)
    }
}

// MARK: - Supporting Types

struct AuthResult {
    let token: String
    let refreshToken: String?
    let user: User
    let expiresAt: Date
}

struct User {
    let id: UUID
    let username: String
    let email: String
    let role: UserRole
    let permissions: Set<Permission>
}

enum UserRole: String, CaseIterable {
    case admin = "admin"
    case editor = "editor"
    case viewer = "viewer"
    case user = "user"
}

enum Permission: String, CaseIterable {
    case createProject = "create_project"
    case editProject = "edit_project"
    case deleteProject = "delete_project"
    case shareProject = "share_project"
    case exportProject = "export_project"
    case manageUsers = "manage_users"
    case viewAnalytics = "view_analytics"
}

enum SSOProvider: String, CaseIterable {
    case google = "google"
    case microsoft = "microsoft"
    case apple = "apple"
    case github = "github"
    case okta = "okta"
}

enum AuthError: LocalizedError {
    case invalidCredentials
    case rateLimited
    case biometricsNotEnabled
    case biometricsAuthFailed
    case noStoredCredentials
    case tokenExpired
    case noRefreshToken
    case refreshFailed
    case notAuthenticated
    case invalidResponse
    case invalidUserData
    case ssoError(String)
    case twoFactorRequired
    case networkError
    
    var errorDescription: String? {
        switch self {
        case .invalidCredentials:
            return "Invalid username or password"
        case .rateLimited:
            return "Too many authentication attempts. Please try again later."
        case .biometricsNotEnabled:
            return "Biometric authentication is not enabled"
        case .biometricsAuthFailed:
            return "Biometric authentication failed"
        case .noStoredCredentials:
            return "No stored credentials found"
        case .tokenExpired:
            return "Authentication token expired"
        case .noRefreshToken:
            return "No refresh token available"
        case .refreshFailed:
            return "Failed to refresh authentication token"
        case .notAuthenticated:
            return "User is not authenticated"
        case .invalidResponse:
            return "Invalid server response"
        case .invalidUserData:
            return "Invalid user data received"
        case .ssoError(let message):
            return "SSO authentication failed: \(message)"
        case .twoFactorRequired:
            return "Two-factor authentication required"
        case .networkError:
            return "Network error occurred"
        }
    }
}

// MARK: - JWT Token Validator
final class JWTTokenValidator {
    
    func isValid(_ token: String) -> Bool {
        let components = token.components(separatedBy: ".")
        guard components.count == 3 else { return false }
        
        // Decode payload
        guard let payload = decodeJWTPayload(components[1]) else { return false }
        
        // Check expiration
        if let exp = payload["exp"] as? TimeInterval {
            return Date().timeIntervalSince1970 < exp
        }
        
        return false
    }
    
    func getExpirationDate(_ token: String) -> Date? {
        let components = token.components(separatedBy: ".")
        guard components.count == 3,
              let payload = decodeJWTPayload(components[1]),
              let exp = payload["exp"] as? TimeInterval else {
            return nil
        }
        
        return Date(timeIntervalSince1970: exp)
    }
    
    private func decodeJWTPayload(_ base64String: String) -> [String: Any]? {
        // Add padding if needed
        var base64 = base64String
        let remainder = base64.count % 4
        if remainder > 0 {
            base64 = base64.padding(toLength: base64.count + 4 - remainder, withPad: "=", startingAt: 0)
        }
        
        guard let data = Data(base64Encoded: base64),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        
        return json
    }
}

// MARK: - Keychain Manager
final class KeychainManager {
    private let service = "com.autoresolve.auth"
    private let tokenKey = "auth_token"
    private let refreshTokenKey = "refresh_token"
    
    func storeToken(_ token: String) throws {
        try storeKeychain(key: tokenKey, value: token)
    }
    
    func getToken() throws -> String? {
        return try getKeychain(key: tokenKey)
    }
    
    func storeRefreshToken(_ token: String) throws {
        try storeKeychain(key: refreshTokenKey, value: token)
    }
    
    func getRefreshToken() throws -> String? {
        return try getKeychain(key: refreshTokenKey)
    }
    
    func clearTokens() throws {
        try deleteKeychain(key: tokenKey)
        try deleteKeychain(key: refreshTokenKey)
    }
    
    private func storeKeychain(key: String, value: String) throws {
        let data = value.data(using: .utf8)!
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        // Delete existing item first
        try? deleteKeychain(key: key)
        
        let status = SecItemAdd(query as CFDictionary, nil)
        
        guard status == errSecSuccess else {
            throw KeychainError.storeFailed(status)
        }
    }
    
    private func getKeychain(key: String) throws -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        if status == errSecItemNotFound {
            return nil
        }
        
        guard status == errSecSuccess,
              let data = result as? Data,
              let string = String(data: data, encoding: .utf8) else {
            throw KeychainError.retrieveFailed(status)
        }
        
        return string
    }
    
    private func deleteKeychain(key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        
        if status != errSecSuccess && status != errSecItemNotFound {
            throw KeychainError.deleteFailed(status)
        }
    }
}

enum KeychainError: LocalizedError {
    case storeFailed(OSStatus)
    case retrieveFailed(OSStatus)
    case deleteFailed(OSStatus)
    
    var errorDescription: String? {
        switch self {
        case .storeFailed(let status):
            return "Failed to store in keychain: \(status)"
        case .retrieveFailed(let status):
            return "Failed to retrieve from keychain: \(status)"
        case .deleteFailed(let status):
            return "Failed to delete from keychain: \(status)"
        }
    }
}

// MARK: - Biometrics Manager
final class BiometricsManager {
    private let context = LAContext()
    
    var isAvailable: Bool {
        var error: NSError?
        return context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error)
    }
    
    func authenticate() async throws -> Bool {
        let reason = "Authenticate to access AutoResolve"
        
        do {
            let result = try await context.evaluatePolicy(
                .deviceOwnerAuthenticationWithBiometrics,
                localizedReason: reason
            )
            return result
        } catch {
            throw AuthError.biometricsAuthFailed
        }
    }
}

// MARK: - SSO Manager (Placeholder)
final class SSOManager {
    
    func authenticate(with provider: SSOProvider) async throws -> SSOResult {
        // Real SSO implementation would go here
        // This is a placeholder for the pattern
        
        switch provider {
        case .google:
            return try await authenticateWithGoogle()
        case .microsoft:
            return try await authenticateWithMicrosoft()
        case .apple:
            return try await authenticateWithApple()
        case .github:
            return try await authenticateWithGitHub()
        case .okta:
            return try await authenticateWithOkta()
        }
    }
    
    private func authenticateWithGoogle() async throws -> SSOResult {
        // Google OAuth implementation
        throw AuthError.ssoError("Google SSO not implemented yet")
    }
    
    private func authenticateWithMicrosoft() async throws -> SSOResult {
        // Microsoft OAuth implementation
        throw AuthError.ssoError("Microsoft SSO not implemented yet")
    }
    
    private func authenticateWithApple() async throws -> SSOResult {
        // Apple Sign In implementation
        throw AuthError.ssoError("Apple SSO not implemented yet")
    }
    
    private func authenticateWithGitHub() async throws -> SSOResult {
        // GitHub OAuth implementation
        throw AuthError.ssoError("GitHub SSO not implemented yet")
    }
    
    private func authenticateWithOkta() async throws -> SSOResult {
        // Okta SAML implementation
        throw AuthError.ssoError("Okta SSO not implemented yet")
    }
}

struct SSOResult {
    let token: String
    let userInfo: [String: Any]
}
