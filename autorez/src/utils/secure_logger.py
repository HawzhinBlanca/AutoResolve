"""
Secure logging with automatic PII sanitization
Blueprint.md compliant logging module
"""
import logging
import re
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

# Patterns to sanitize
SENSITIVE_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
    (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CREDIT_CARD]'),  # Credit card
    (r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', 'Bearer [TOKEN]'),  # Bearer tokens
    (r'"password"\s*:\s*"[^"]*"', '"password":"[REDACTED]"'),  # Password fields
    (r'"token"\s*:\s*"[^"]*"', '"token":"[REDACTED]"'),  # Token fields
    (r'"api_key"\s*:\s*"[^"]*"', '"api_key":"[REDACTED]"'),  # API keys
    (r'/Users/[^/\s]+', '/Users/[USER]'),  # User paths
    (r'\/home\/[^\/\s]+', '/home/[USER]'),  # Linux home paths
]

# Fields to always redact in structured logs
REDACTED_FIELDS = {
    'password', 'token', 'api_key', 'secret', 'credential',
    'authorization', 'auth', 'private_key', 'access_token',
    'refresh_token', 'client_secret'
}


class SecureFormatter(logging.Formatter):
    """Custom formatter that sanitizes sensitive data"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Format the base message
        msg = super().format(record)
        
        # Sanitize the message
        msg = self.sanitize_message(msg)
        
        # Sanitize any extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key.lower() in REDACTED_FIELDS:
                    setattr(record, key, '[REDACTED]')
                elif isinstance(value, str):
                    setattr(record, key, self.sanitize_message(value))
        
        return msg
    
    @staticmethod
    def sanitize_message(msg: str) -> str:
        """Remove sensitive information from log messages"""
        for pattern, replacement in SENSITIVE_PATTERNS:
            msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
        return msg


class StructuredLogger:
    """Structured logging with automatic sanitization"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler with secure formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(SecureFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        # File handler for audit trail (if configured)
        log_dir = Path(os.getenv("LOG_DIR", "logs"))
        if log_dir.exists():
            file_handler = logging.FileHandler(
                log_dir / f"{name.replace('.', '_')}.log"
            )
            file_handler.setFormatter(SecureFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            ))
            self.logger.addHandler(file_handler)
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary data"""
        sanitized = {}
        for key, value in data.items():
            if key.lower() in REDACTED_FIELDS:
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, str):
                sanitized[key] = SecureFormatter.sanitize_message(value)
            else:
                sanitized[key] = value
        return sanitized
    
    def debug(self, msg: str, **kwargs):
        """Debug level logging with sanitization"""
        extra = self._sanitize_dict(kwargs) if kwargs else {}
        self.logger.debug(msg, extra=extra)
    
    def info(self, msg: str, **kwargs):
        """Info level logging with sanitization"""
        extra = self._sanitize_dict(kwargs) if kwargs else {}
        self.logger.info(msg, extra=extra)
    
    def warning(self, msg: str, **kwargs):
        """Warning level logging with sanitization"""
        extra = self._sanitize_dict(kwargs) if kwargs else {}
        self.logger.warning(msg, extra=extra)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs):
        """Error level logging with sanitization"""
        extra = self._sanitize_dict(kwargs) if kwargs else {}
        self.logger.error(msg, exc_info=exc_info, extra=extra)
    
    def critical(self, msg: str, exc_info: bool = False, **kwargs):
        """Critical level logging with sanitization"""
        extra = self._sanitize_dict(kwargs) if kwargs else {}
        self.logger.critical(msg, exc_info=exc_info, extra=extra)
    
    def audit(self, event: str, user: Optional[str] = None, **details):
        """Special audit logging for security events"""
        audit_data = {
            'event': event,
            'user': user or 'anonymous',
            'timestamp': datetime.now().isoformat(),
            'details': self._sanitize_dict(details)
        }
        self.logger.info(f"AUDIT: {json.dumps(audit_data)}")


# Factory function for easy import
def get_logger(name: str) -> StructuredLogger:
    """Get a configured secure logger instance"""
    return StructuredLogger(name)


# For backward compatibility
import os
from datetime import datetime

secure_logger = get_logger(__name__)