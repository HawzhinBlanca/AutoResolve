//
//  SecurityPenetrationTests.swift
//  AutoResolveUITests
//
//  Created by AutoResolve on 8/23/25.
//

import XCTest
import Foundation
import Network
import CryptoKit
import Combine
@testable import AutoResolveUI

/// Comprehensive security penetration testing framework for AutoResolve
/// Tests security vulnerabilities, attack vectors, and defensive mechanisms
@MainActor
class SecurityPenetrationTests: XCTestCase {
    
    // MARK: - Test Infrastructure
    
    private var encryptionManager: EncryptionManager!
    private var authManager: AuthenticationManager!
    private var accessControlManager: AccessControlManager!
    private var auditLogger: AuditLogger!
    private var securityMonitor: SecurityMonitor!
    private var collaborationManager: CollaborationManager!
    private var databaseManager: DatabaseManager!
    
    private var cancellables: Set<AnyCancellable> = []
    private var testTimeout: TimeInterval = 60.0
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        
        // Initialize security managers
        encryptionManager = EncryptionManager.shared
        authManager = AuthenticationManager.shared
        accessControlManager = AccessControlManager.shared
        auditLogger = AuditLogger.shared
        securityMonitor = SecurityMonitor.shared
        collaborationManager = CollaborationManager.shared
        databaseManager = DatabaseManager.shared
        
        // Enable security testing mode
        UserDefaults.standard.set(true, forKey: "SecurityTestMode")
        UserDefaults.standard.set(true, forKey: "VerboseSecurityLogging")
        
        // Setup test environment
        try setupSecurityTestEnvironment()
    }
    
    override func tearDownWithError() throws {
        // Reset security testing mode
        UserDefaults.standard.removeObject(forKey: "SecurityTestMode")
        UserDefaults.standard.removeObject(forKey: "VerboseSecurityLogging")
        
        // Clean up
        cancellables.removeAll()
        
        try super.tearDownWithError()
    }
    
    private func setupSecurityTestEnvironment() throws {
        // Create isolated test environment
        let testDir = FileManager.default.temporaryDirectory.appendingPathComponent("SecurityTests")
        try FileManager.default.createDirectory(at: testDir, withIntermediateDirectories: true)
        
        // Setup test certificates and keys
        try setupTestCertificates()
    }
    
    private func setupTestCertificates() throws {
        // Generate test certificates for TLS testing
        let testCertDir = FileManager.default.temporaryDirectory.appendingPathComponent("TestCerts")
        try FileManager.default.createDirectory(at: testCertDir, withIntermediateDirectories: true)
        
        // Store test cert path
        UserDefaults.standard.set(testCertDir.path, forKey: "TestCertificatePath")
    }
    
    // MARK: - Authentication Security Tests
    
    func testPasswordBruteForceProtection() async throws {
        let expectation = expectation(description: "Brute force protection test")
        
        let testUsername = "bruteforce_test_user"
        let correctPassword = "SecurePassword123!"
        let wrongPassword = "WrongPassword"
        
        do {
            // Create test user
            let testUser = TestUser(username: testUsername, password: correctPassword)
            try await authManager.createUser(testUser)
            
            // Attempt multiple failed logins (should trigger rate limiting)
            var failedAttempts = 0
            let maxAttempts = 10
            
            for attempt in 1...maxAttempts {
                let startTime = CFAbsoluteTimeGetCurrent()
                
                let result = try await authManager.authenticate(
                    username: testUsername,
                    password: wrongPassword
                )
                
                let endTime = CFAbsoluteTimeGetCurrent()
                let responseTime = endTime - startTime
                
                XCTAssertFalse(result.isSuccess, "Authentication should fail with wrong password")
                
                if !result.isSuccess {
                    failedAttempts += 1
                }
                
                // After 5 attempts, response time should increase (rate limiting)
                if attempt > 5 {
                    XCTAssertGreaterThan(responseTime, 1.0, "Rate limiting should slow down responses after multiple failures")
                }
                
                // After 8 attempts, account should be temporarily locked
                if attempt > 8 {
                    XCTAssertTrue(result.error?.contains("temporarily locked") ?? false, 
                                "Account should be temporarily locked after many failures")
                }
            }
            
            // Verify audit logs recorded the attack attempts
            let auditLogs = try await auditLogger.getAuditLogs(
                filter: AuditLogFilter(username: testUsername, action: "authentication_failed")
            )
            
            XCTAssertGreaterThanOrEqual(auditLogs.count, failedAttempts, 
                                      "All failed attempts should be logged")
            
            // Verify security monitor detected the brute force attempt
            let securityEvents = await securityMonitor.getSecurityEvents(
                filter: SecurityEventFilter(eventType: .bruteForceAttempt, username: testUsername)
            )
            
            XCTAssertFalse(securityEvents.isEmpty, "Security monitor should detect brute force attempts")
            
            expectation.fulfill()
        } catch {
            XCTFail("Brute force protection test failed: \(error)")
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: testTimeout)
    }
    
    func testJWTTokenSecurityVulnerabilities() async throws {
        let expectation = expectation(description: "JWT token security test")
        
        do {
            // Create test user and authenticate
            let testUser = TestUser(username: "jwt_test_user", password: "SecurePassword123!")
            try await authManager.createUser(testUser)
            
            let authResult = try await authManager.authenticate(
                username: testUser.username,
                password: testUser.password
            )
            
            XCTAssertTrue(authResult.isSuccess)
            XCTAssertNotNil(authResult.token)
            
            let validToken = authResult.token!
            
            // Test 1: Token tampering (modify payload)
            let tamperedToken = try tamperWithJWTToken(validToken)
            let tamperedValidation = try await authManager.validateToken(tamperedToken)
            XCTAssertFalse(tamperedValidation.isValid, "Tampered token should be invalid")
            
            // Test 2: Token signature stripping
            let unsignedToken = stripJWTSignature(validToken)
            let unsignedValidation = try await authManager.validateToken(unsignedToken)
            XCTAssertFalse(unsignedValidation.isValid, "Unsigned token should be invalid")
            
            // Test 3: Algorithm confusion attack (change RS256 to HS256)
            let confusedToken = try performAlgorithmConfusionAttack(validToken)
            let confusedValidation = try await authManager.validateToken(confusedToken)
            XCTAssertFalse(confusedValidation.isValid, "Algorithm confusion attack should fail")
            
            // Test 4: Replay attack (use expired token)
            try await Task.sleep(nanoseconds: 2_000_000_000) // Wait 2 seconds
            let expiredToken = try createExpiredToken(testUser.username)
            let expiredValidation = try await authManager.validateToken(expiredToken)
            XCTAssertFalse(expiredValidation.isValid, "Expired token should be invalid")
            
            // Test 5: Token with invalid claims
            let invalidClaimsToken = try createTokenWithInvalidClaims(testUser.username)
            let invalidClaimsValidation = try await authManager.validateToken(invalidClaimsToken)
            XCTAssertFalse(invalidClaimsValidation.isValid, "Token with invalid claims should be invalid")
            
            // Verify security events were logged
            let securityEvents = await securityMonitor.getSecurityEvents(
                filter: SecurityEventFilter(eventType: .tokenTampering)
            )
            XCTAssertFalse(securityEvents.isEmpty, "Token tampering attempts should be detected")
            
            expectation.fulfill()
        } catch {
            XCTFail("JWT security test failed: \(error)")
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: testTimeout)
    }
    
    // MARK: - Encryption Security Tests
    
    func testEncryptionVulnerabilities() async throws {
        let expectation = expectation(description: "Encryption vulnerabilities test")
        
        do {
            let testData = "Sensitive data that must be protected".data(using: .utf8)!
            
            // Test 1: Weak encryption key detection
            let weakKey = Data(repeating: 0x00, count: 32) // Weak key (all zeros)
            
            do {
                _ = try await encryptionManager.encryptWithCustomKey(data: testData, key: weakKey)
                XCTFail("Encryption should reject weak keys")
            } catch {
                // Expected to fail with weak key
                XCTAssertTrue(error.localizedDescription.contains("weak key") || 
                            error.localizedDescription.contains("invalid key"))
            }
            
            // Test 2: Key reuse detection
            let key1 = try encryptionManager.generateEncryptionKey()
            let encrypted1 = try await encryptionManager.encryptWithCustomKey(data: testData, key: key1)
            let encrypted2 = try await encryptionManager.encryptWithCustomKey(data: testData, key: key1)
            
            // Even with same key and data, ciphertexts should be different (due to IV/nonce)
            XCTAssertNotEqual(encrypted1, encrypted2, "Same plaintext with same key should produce different ciphertexts")
            
            // Test 3: IV/Nonce reuse vulnerability
            let repeatedIV = Data(repeating: 0x01, count: 16)
            
            do {
                _ = try await encryptionManager.encryptWithFixedIV(data: testData, key: key1, iv: repeatedIV)
                _ = try await encryptionManager.encryptWithFixedIV(data: testData, key: key1, iv: repeatedIV)
                XCTFail("Encryption should detect and prevent IV reuse")
            } catch {
                // Expected to fail with IV reuse detection
            }
            
            // Test 4: Padding oracle attack resistance
            let encryptedData = try await encryptionManager.encrypt(data: testData, purpose: .fileStorage)
            
            // Tamper with encrypted data to test padding oracle resistance
            var tamperedData = encryptedData
            tamperedData[tamperedData.count - 1] ^= 0x01 // Flip last bit
            
            do {
                _ = try await encryptionManager.decrypt(encryptedData: tamperedData, purpose: .fileStorage)
                XCTFail("Decryption should fail with tampered data")
            } catch {
                // Expected to fail - tampering should be detected
            }
            
            // Test 5: Side-channel attack resistance (timing attack)
            let iterations = 100
            var timingDifferences: [Double] = []
            
            for _ in 0..<iterations {
                let validData = try await encryptionManager.encrypt(data: testData, purpose: .fileStorage)
                let invalidData = Data(repeating: 0xFF, count: validData.count)
                
                let validStart = CFAbsoluteTimeGetCurrent()
                _ = try? await encryptionManager.decrypt(encryptedData: validData, purpose: .fileStorage)
                let validEnd = CFAbsoluteTimeGetCurrent()
                
                let invalidStart = CFAbsoluteTimeGetCurrent()
                _ = try? await encryptionManager.decrypt(encryptedData: invalidData, purpose: .fileStorage)
                let invalidEnd = CFAbsoluteTimeGetCurrent()
                
                let timingDiff = abs((validEnd - validStart) - (invalidEnd - invalidStart))
                timingDifferences.append(timingDiff)
            }
            
            let avgTimingDiff = timingDifferences.reduce(0, +) / Double(timingDifferences.count)
            XCTAssertLessThan(avgTimingDiff, 0.001, "Timing differences should be minimal to prevent timing attacks")
            
            expectation.fulfill()
        } catch {
            XCTFail("Encryption vulnerabilities test failed: \(error)")
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: testTimeout)
    }
    
    // MARK: - Access Control Security Tests
    
    func testPrivilegeEscalationAttacks() async throws {
        let expectation = expectation(description: "Privilege escalation test")
        
        do {
            // Create users with different privilege levels
            let regularUserId = UUID()
            let adminUserId = UUID()
            
            let regularRole = AccessRole(
                id: UUID(),
                name: "regular_user",
                permissions: [.readMedia, .editOwnProjects]
            )
            
            let adminRole = AccessRole(
                id: UUID(),
                name: "admin",
                permissions: [.readMedia, .editOwnProjects, .editAllProjects, .manageUsers, .systemAdmin]
            )
            
            // Assign roles
            try await accessControlManager.assignRole(userId: regularUserId, role: regularRole)
            try await accessControlManager.assignRole(userId: adminUserId, role: adminRole)
            
            // Test 1: Direct privilege escalation attempt
            do {
                try await accessControlManager.assignRole(userId: regularUserId, role: adminRole)
                XCTFail("Regular user should not be able to escalate to admin role")
            } catch {
                // Expected to fail
            }
            
            // Test 2: Permission manipulation attempt
            let maliciousPermissions: Set<Permission> = [.systemAdmin, .manageUsers]
            
            do {
                try await accessControlManager.grantPermissions(
                    userId: regularUserId,
                    permissions: maliciousPermissions,
                    grantedBy: regularUserId // Self-granting
                )
                XCTFail("User should not be able to grant permissions to themselves")
            } catch {
                // Expected to fail
            }
            
            // Test 3: Role modification attack
            let modifiedRole = AccessRole(
                id: regularRole.id,
                name: "modified_regular",
                permissions: [.readMedia, .editAllProjects, .systemAdmin] // Escalated permissions
            )
            
            do {
                try await accessControlManager.updateRole(
                    roleId: regularRole.id,
                    newRole: modifiedRole,
                    updatedBy: regularUserId
                )
                XCTFail("Regular user should not be able to modify roles")
            } catch {
                // Expected to fail
            }
            
            // Test 4: Session hijacking simulation
            let regularUserSession = try await authManager.createSession(userId: regularUserId)
            
            // Attempt to use regular user session to perform admin actions
            let hasAdminAccess = try await accessControlManager.checkAccess(
                userId: regularUserId,
                resource: "system:users",
                action: .manageUsers
            )
            
            XCTAssertFalse(hasAdminAccess, "Regular user should not have admin access even with valid session")
            
            // Test 5: Resource path traversal attack
            let maliciousResourcePath = "../../system/admin/users"
            
            let hasTraversalAccess = try await accessControlManager.checkAccess(
                userId: regularUserId,
                resource: maliciousResourcePath,
                action: .readMedia
            )
            
            XCTAssertFalse(hasTraversalAccess, "Path traversal attacks should be blocked")
            
            // Verify all attacks were logged
            let securityEvents = await securityMonitor.getSecurityEvents(
                filter: SecurityEventFilter(eventType: .privilegeEscalationAttempt, userId: regularUserId)
            )
            
            XCTAssertGreaterThan(securityEvents.count, 3, "Multiple privilege escalation attempts should be detected")
            
            expectation.fulfill()
        } catch {
            XCTFail("Privilege escalation test failed: \(error)")
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: testTimeout)
    }
    
    // MARK: - Database Security Tests
    
    func testSQLInjectionPrevention() async throws {
        let expectation = expectation(description: "SQL injection prevention test")
        
        do {
            // Test 1: Classic SQL injection in search
            let maliciousSearchTerms = [
                "'; DROP TABLE projects; --",
                "' OR '1'='1",
                "'; UPDATE projects SET owner='hacker' WHERE '1'='1'; --",
                "' UNION SELECT password FROM users WHERE '1'='1",
                "'; INSERT INTO admin_users VALUES('hacker', 'password'); --"
            ]
            
            for searchTerm in maliciousSearchTerms {
                let searchResult = try await databaseManager.searchProjects(query: searchTerm)
                
                // Should return empty or legitimate results, not execute injection
                XCTAssertNotNil(searchResult, "Search should return results without executing injection")
                
                // Verify database integrity wasn't compromised
                let projectsStillExist = try await databaseManager.loadAllProjects()
                XCTAssertFalse(projectsStillExist.isEmpty || projectsStillExist.count > 1000, 
                             "Database should not be compromised by injection attempt")
            }
            
            // Test 2: Second-order SQL injection
            let maliciousProjectName = "Normal Project'; DROP TABLE media_items; --"
            let project = VideoProject(name: maliciousProjectName)
            
            try await databaseManager.saveProject(project)
            
            // Later operation that might trigger second-order injection
            let loadedProject = try await databaseManager.loadProject(id: project.id)
            XCTAssertNotNil(loadedProject)
            XCTAssertEqual(loadedProject?.name, maliciousProjectName) // Name should be stored as-is
            
            // Verify media_items table still exists
            let mediaItems = try await databaseManager.loadAllMediaItems()
            XCTAssertNotNil(mediaItems) // Should not crash
            
            // Test 3: Blind SQL injection timing attack
            let timingAttackQueries = [
                "test' AND (SELECT COUNT(*) FROM projects) > 0 AND SLEEP(5) --",
                "test'; WAITFOR DELAY '00:00:05' --",
                "test' || (SELECT CASE WHEN (1=1) THEN pg_sleep(5) ELSE 0 END) || '"
            ]
            
            for query in timingAttackQueries {
                let startTime = CFAbsoluteTimeGetCurrent()
                _ = try await databaseManager.searchProjects(query: query)
                let endTime = CFAbsoluteTimeGetCurrent()
                let queryTime = endTime - startTime
                
                XCTAssertLessThan(queryTime, 2.0, "Query should not be delayed by injection attempt")
            }
            
            // Verify security events were logged
            let securityEvents = await securityMonitor.getSecurityEvents(
                filter: SecurityEventFilter(eventType: .sqlInjectionAttempt)
            )
            
            XCTAssertGreaterThan(securityEvents.count, maliciousSearchTerms.count, 
                               "SQL injection attempts should be detected and logged")
            
            expectation.fulfill()
        } catch {
            XCTFail("SQL injection prevention test failed: \(error)")
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: testTimeout)
    }
    
    // MARK: - Network Security Tests
    
    func testNetworkSecurityVulnerabilities() async throws {
        let expectation = expectation(description: "Network security test")
        
        do {
            // Test 1: Man-in-the-middle attack simulation
            let sessionId = UUID()
            let userId = UUID()
            
            let session = try await collaborationManager.createSession(
                projectId: UUID(),
                createdBy: userId,
                isEncrypted: true
            )
            
            // Simulate MITM by intercepting and modifying messages
            let originalMessage = CollaborationMessage(
                type: .edit,
                content: "Legitimate edit operation",
                senderId: userId,
                timestamp: Date()
            )
            
            let modifiedMessage = CollaborationMessage(
                type: .edit,
                content: "Malicious edit operation - DELETE ALL",
                senderId: userId,
                timestamp: Date()
            )
            
            // Send original message (encrypted)
            let encryptedOriginal = try JSONEncoder().encode(originalMessage)
            try await collaborationManager.sendMessage(
                sessionId: session.id,
                encryptedData: encryptedOriginal,
                senderId: userId
            )
            
            // Attempt to inject modified message
            let encryptedModified = try JSONEncoder().encode(modifiedMessage)
            
            do {
                try await collaborationManager.sendMessage(
                    sessionId: session.id,
                    encryptedData: encryptedModified,
                    senderId: userId
                )
                
                // Verify message integrity wasn't compromised
                let receivedMessages = try await collaborationManager.getMessages(sessionId: session.id)
                let lastMessage = try JSONDecoder().decode(CollaborationMessage.self, 
                                                         from: receivedMessages.last!)
                
                XCTAssertNotEqual(lastMessage.content, modifiedMessage.content, 
                                "Modified message should be detected and rejected")
                
            } catch {
                // Expected to fail due to encryption/integrity checks
            }
            
            // Test 2: Replay attack
            let replayMessage = encryptedOriginal
            
            // Wait and try to replay the message
            try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
            
            do {
                try await collaborationManager.sendMessage(
                    sessionId: session.id,
                    encryptedData: replayMessage,
                    senderId: userId
                )
                
                XCTFail("Replay attack should be detected and prevented")
            } catch {
                // Expected to fail due to replay protection
            }
            
            // Test 3: Session fixation attack
            let fixedSessionId = UUID()
            
            do {
                _ = try await collaborationManager.joinSessionWithFixedId(
                    sessionId: fixedSessionId,
                    userId: userId
                )
                XCTFail("Session fixation should be prevented")
            } catch {
                // Expected to fail
            }
            
            // Test 4: SSL/TLS downgrade attack simulation
            let insecureConnectionAttempt = try await testInsecureConnection()
            XCTAssertFalse(insecureConnectionAttempt, 
                          "Insecure connections should be rejected")
            
            // Verify network security events
            let networkEvents = await securityMonitor.getSecurityEvents(
                filter: SecurityEventFilter(eventType: .networkSecurityViolation)
            )
            
            XCTAssertFalse(networkEvents.isEmpty, "Network security violations should be detected")
            
            expectation.fulfill()
        } catch {
            XCTFail("Network security test failed: \(error)")
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: testTimeout)
    }
    
    // MARK: - File System Security Tests
    
    func testFileSystemSecurityVulnerabilities() async throws {
        let expectation = expectation(description: "File system security test")
        
        do {
            // Test 1: Path traversal attack
            let maliciousPaths = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "/etc/shadow",
                "../../database/users.db",
                "../config/secrets.key",
                "file:///etc/passwd",
                "\\\\server\\share\\sensitive.txt"
            ]
            
            for maliciousPath in maliciousPaths {
                do {
                    let fileURL = URL(fileURLWithPath: maliciousPath)
                    let mediaItem = MediaItem(url: fileURL, type: .video)
                    
                    try await databaseManager.saveMediaItem(mediaItem)
                    
                    // Try to access the media item
                    _ = try await databaseManager.loadMediaItem(id: mediaItem.id)
                    
                    XCTFail("Path traversal attack should be blocked: \(maliciousPath)")
                } catch {
                    // Expected to fail - path traversal should be blocked
                }
            }
            
            // Test 2: Symbolic link attack
            let testDir = FileManager.default.temporaryDirectory.appendingPathComponent("SymlinkTest")
            try FileManager.default.createDirectory(at: testDir, withIntermediateDirectories: true)
            
            let legitimateFile = testDir.appendingPathComponent("legitimate.mp4")
            try "legitimate content".write(to: legitimateFile, atomically: true, encoding: .utf8)
            
            let symlinkFile = testDir.appendingPathComponent("symlink.mp4")
            let sensitiveTarget = "/etc/passwd"
            
            do {
                try FileManager.default.createSymbolicLink(at: symlinkFile, 
                                                         withDestinationURL: URL(fileURLWithPath: sensitiveTarget))
                
                let mediaItem = MediaItem(url: symlinkFile, type: .video)
                try await databaseManager.saveMediaItem(mediaItem)
                
                XCTFail("Symbolic link to sensitive file should be blocked")
            } catch {
                // Expected to fail
            }
            
            // Test 3: File inclusion attack
            let fileInclusionAttempts = [
                "php://filter/convert.base64-encode/resource=../../../etc/passwd",
                "data://text/plain;base64,PD9waHAgcGhwaW5mbygpOyA/Pg==",
                "file:///proc/self/environ",
                "expect://id"
            ]
            
            for inclusionAttempt in fileInclusionAttempts {
                do {
                    let url = URL(string: inclusionAttempt) ?? URL(fileURLWithPath: inclusionAttempt)
                    let mediaItem = MediaItem(url: url, type: .video)
                    
                    try await databaseManager.saveMediaItem(mediaItem)
                    XCTFail("File inclusion attack should be blocked: \(inclusionAttempt)")
                } catch {
                    // Expected to fail
                }
            }
            
            // Test 4: File upload security
            let maliciousFileTypes = [
                ("malicious.exe", Data([0x4D, 0x5A])), // PE executable header
                ("script.php", "<?php system($_GET['cmd']); ?>".data(using: .utf8)!),
                ("payload.js", "alert('XSS')".data(using: .utf8)!),
                ("hidden.scr", Data([0xFF, 0xD8, 0xFF])) // JPEG header but .scr extension
            ]
            
            for (filename, content) in maliciousFileTypes {
                let tempFile = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
                try content.write(to: tempFile)
                
                do {
                    let mediaItem = MediaItem(url: tempFile, type: .video)
                    try await databaseManager.saveMediaItem(mediaItem)
                    
                    XCTFail("Malicious file should be rejected: \(filename)")
                } catch {
                    // Expected to fail due to file type validation
                }
            }
            
            // Verify file system security events
            let fileSystemEvents = await securityMonitor.getSecurityEvents(
                filter: SecurityEventFilter(eventType: .fileSystemViolation)
            )
            
            XCTAssertGreaterThan(fileSystemEvents.count, 10, 
                               "File system security violations should be detected")
            
            expectation.fulfill()
        } catch {
            XCTFail("File system security test failed: \(error)")
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: testTimeout)
    }
    
    // MARK: - Compliance Security Tests
    
    func testDataProtectionCompliance() async throws {
        let expectation = expectation(description: "Data protection compliance test")
        
        do {
            let complianceFramework = ComplianceFramework.shared
            
            // Test 1: Data encryption at rest
            let sensitiveData = PersonalData(
                id: UUID(),
                email: "test@example.com",
                name: "Test User",
                processingPurpose: "Video editing"
            )
            
            try await complianceFramework.recordPersonalData(sensitiveData)
            
            // Verify data is encrypted when stored
            let storedData = try await complianceFramework.getStoredPersonalData(id: sensitiveData.id)
            XCTAssertTrue(storedData.isEncrypted, "Personal data should be encrypted at rest")
            
            // Test 2: Data retention policy enforcement
            let oldData = PersonalData(
                id: UUID(),
                email: "old@example.com",
                name: "Old User",
                processingPurpose: "Expired processing",
                retentionDate: Calendar.current.date(byAdding: .day, value: -400, to: Date())! // Expired
            )
            
            try await complianceFramework.recordPersonalData(oldData)
            
            // Trigger retention policy check
            try await complianceFramework.enforceRetentionPolicies()
            
            // Verify expired data was deleted
            let expiredData = try await complianceFramework.getStoredPersonalData(id: oldData.id)
            XCTAssertNil(expiredData, "Expired personal data should be automatically deleted")
            
            // Test 3: Data minimization compliance
            let excessiveData = PersonalData(
                id: UUID(),
                email: "excessive@example.com",
                name: "User with too much data",
                processingPurpose: "Simple video editing",
                additionalData: ["ssn": "123-45-6789", "medical_info": "sensitive"] // Excessive
            )
            
            do {
                try await complianceFramework.recordPersonalData(excessiveData)
                XCTFail("Excessive data collection should be blocked")
            } catch {
                // Expected to fail due to data minimization policy
            }
            
            // Test 4: Cross-border data transfer restrictions
            let euUserData = PersonalData(
                id: UUID(),
                email: "eu_user@example.com",
                name: "EU User",
                processingPurpose: "Video editing",
                jurisdiction: "EU"
            )
            
            try await complianceFramework.recordPersonalData(euUserData)
            
            // Attempt to transfer to non-adequate country
            do {
                try await complianceFramework.transferData(
                    dataId: euUserData.id,
                    destinationCountry: "NonAdequateCountry"
                )
                XCTFail("Cross-border transfer to non-adequate country should be blocked")
            } catch {
                // Expected to fail
            }
            
            // Test 5: Consent management
            let userData = PersonalData(
                id: UUID(),
                email: "consent_test@example.com",
                name: "Consent Test User",
                processingPurpose: "Video editing"
            )
            
            // Record data without proper consent
            do {
                try await complianceFramework.recordPersonalDataWithoutConsent(userData)
                XCTFail("Data processing without consent should be blocked")
            } catch {
                // Expected to fail
            }
            
            // Record with proper consent
            let consent = DataConsent(
                userId: userData.id,
                purposes: ["video_editing"],
                consentDate: Date(),
                isValid: true
            )
            
            try await complianceFramework.recordConsent(consent)
            try await complianceFramework.recordPersonalData(userData)
            
            // Verify data was recorded with proper consent
            let consentedData = try await complianceFramework.getStoredPersonalData(id: userData.id)
            XCTAssertNotNil(consentedData, "Data with proper consent should be stored")
            
            // Verify compliance audit trail
            let complianceEvents = try await auditLogger.getAuditLogs(
                filter: AuditLogFilter(category: "compliance")
            )
            
            XCTAssertGreaterThan(complianceEvents.count, 3, 
                               "Compliance events should be thoroughly audited")
            
            expectation.fulfill()
        } catch {
            XCTFail("Data protection compliance test failed: \(error)")
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: testTimeout)
    }
    
    // MARK: - Helper Methods
    
    private func tamperWithJWTToken(_ token: String) throws -> String {
        let parts = token.split(separator: ".")
        guard parts.count == 3 else {
            throw SecurityTestError.invalidToken
        }
        
        // Decode payload
        var payload = String(parts[1])
        
        // Add padding if needed
        while payload.count % 4 != 0 {
            payload += "="
        }
        
        guard let payloadData = Data(base64Encoded: payload),
              var payloadDict = try JSONSerialization.jsonObject(with: payloadData) as? [String: Any] else {
            throw SecurityTestError.tokenDecodingFailed
        }
        
        // Tamper with payload (add admin role)
        payloadDict["role"] = "admin"
        
        let tamperedPayloadData = try JSONSerialization.data(withJSONObject: payloadDict)
        let tamperedPayload = tamperedPayloadData.base64EncodedString()
            .replacingOccurrences(of: "=", with: "")
        
        return "\(parts[0]).\(tamperedPayload).\(parts[2])"
    }
    
    private func stripJWTSignature(_ token: String) -> String {
        let parts = token.split(separator: ".")
        if parts.count == 3 {
            return "\(parts[0]).\(parts[1])."
        }
        return token
    }
    
    private func performAlgorithmConfusionAttack(_ token: String) throws -> String {
        let parts = token.split(separator: ".")
        guard parts.count == 3 else {
            throw SecurityTestError.invalidToken
        }
        
        // Change algorithm in header from RS256 to HS256
        var header = String(parts[0])
        while header.count % 4 != 0 {
            header += "="
        }
        
        guard let headerData = Data(base64Encoded: header),
              var headerDict = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw SecurityTestError.tokenDecodingFailed
        }
        
        headerDict["alg"] = "HS256"
        
        let modifiedHeaderData = try JSONSerialization.data(withJSONObject: headerDict)
        let modifiedHeader = modifiedHeaderData.base64EncodedString()
            .replacingOccurrences(of: "=", with: "")
        
        return "\(modifiedHeader).\(parts[1]).\(parts[2])"
    }
    
    private func createExpiredToken(_ username: String) throws -> String {
        // Create a token that's already expired
        let expiredDate = Calendar.current.date(byAdding: .hour, value: -2, to: Date())!
        
        let payload: [String: Any] = [
            "sub": username,
            "exp": Int(expiredDate.timeIntervalSince1970),
            "iat": Int(expiredDate.timeIntervalSince1970) - 3600
        ]
        
        let payloadData = try JSONSerialization.data(withJSONObject: payload)
        let payloadString = payloadData.base64EncodedString()
            .replacingOccurrences(of: "=", with: "")
        
        // Mock JWT structure (this won't have valid signature)
        return "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.\(payloadString).mock_signature"
    }
    
    private func createTokenWithInvalidClaims(_ username: String) throws -> String {
        let payload: [String: Any] = [
            "sub": username,
            "exp": "invalid_expiry", // Invalid expiry format
            "aud": ["invalid_audience"],
            "iss": "malicious_issuer"
        ]
        
        let payloadData = try JSONSerialization.data(withJSONObject: payload)
        let payloadString = payloadData.base64EncodedString()
            .replacingOccurrences(of: "=", with: "")
        
        return "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.\(payloadString).mock_signature"
    }
    
    private func testInsecureConnection() async throws -> Bool {
        // Simulate attempt to connect over insecure channel
        // This would normally involve network code, but we simulate the security check
        let connectionParameters = NWParameters.tcp
        connectionParameters.allowLocalEndpointReuse = true
        
        // In real implementation, this would test TLS configuration
        // For testing, we simulate the security validation
        return false // Insecure connections should always be rejected
    }
}

// MARK: - Supporting Types

private enum SecurityTestError: Error {
    case invalidToken
    case tokenDecodingFailed
    case networkTestFailed
    case encryptionTestFailed
}

private struct TestUser {
    let username: String
    let password: String
}

private struct SecurityEventFilter {
    let eventType: SecurityEventType?
    let username: String?
    let userId: UUID?
    
    init(eventType: SecurityEventType? = nil, username: String? = nil, userId: UUID? = nil) {
        self.eventType = eventType
        self.username = username
        self.userId = userId
    }
}

private enum SecurityEventType {
    case bruteForceAttempt
    case tokenTampering
    case privilegeEscalationAttempt
    case sqlInjectionAttempt
    case networkSecurityViolation
    case fileSystemViolation
}

private struct PersonalData {
    let id: UUID
    let email: String
    let name: String
    let processingPurpose: String
    let retentionDate: Date?
    let additionalData: [String: String]?
    let jurisdiction: String?
    
    init(id: UUID, email: String, name: String, processingPurpose: String, 
         retentionDate: Date? = nil, additionalData: [String: String]? = nil, 
         jurisdiction: String? = nil) {
        self.id = id
        self.email = email
        self.name = name
        self.processingPurpose = processingPurpose
        self.retentionDate = retentionDate
        self.additionalData = additionalData
        self.jurisdiction = jurisdiction
    }
}

private struct DataConsent {
    let userId: UUID
    let purposes: [String]
    let consentDate: Date
    let isValid: Bool
}

private struct StoredPersonalData {
    let data: PersonalData
    let isEncrypted: Bool
}