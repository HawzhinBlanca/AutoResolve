"""
Standardized exception handling for AutoResolve
Blueprint.md compliant error hierarchy
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class AutoResolveError(Exception):
    """Base exception for all AutoResolve errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
        
    def to_http_exception(self, status_code: int = 500) -> HTTPException:
        """Convert to FastAPI HTTPException"""
        return HTTPException(
            status_code=status_code,
            detail={
                "error": self.__class__.__name__,
                "message": self.message,
                "details": self.details
            }
        )


class ValidationError(AutoResolveError):
    """Input validation errors"""
    pass


class SecurityError(AutoResolveError):
    """Security-related errors"""
    def __init__(self, message: str = "Security violation detected", details: Optional[Dict[str, Any]] = None):
        # Don't expose sensitive details in production
        safe_message = "Security check failed" if details else message
        super().__init__(safe_message, details)
        # Log full details securely
        logger.error(f"Security error: {message} - Details: {details}")


class ProcessingError(AutoResolveError):
    """Video processing errors"""
    pass


class BackendError(AutoResolveError):
    """Backend service errors"""
    pass


class RateLimitError(AutoResolveError):
    """Rate limiting errors"""
    def __init__(self, retry_after: Optional[int] = None):
        message = "Rate limit exceeded"
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(message, details)


class AuthenticationError(AutoResolveError):
    """Authentication failures"""
    def __init__(self, message: str = "Authentication failed"):
        # Never expose auth details
        super().__init__("Authentication failed", {})
        logger.warning(f"Auth error: {message}")


class AuthorizationError(AutoResolveError):
    """Authorization failures"""
    def __init__(self, resource: Optional[str] = None):
        message = "Access denied"
        details = {"resource": resource} if resource else {}
        super().__init__(message, details)


# Error handler decorator
def handle_errors(default_status: int = 500):
    """Decorator for consistent error handling"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except AutoResolveError as e:
                raise e.to_http_exception(default_status)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail="An unexpected error occurred"
                )
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator