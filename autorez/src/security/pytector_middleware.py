#!/usr/bin/env python3
"""
Pytector Security Middleware - Centralized Input Validation
AutoResolve Enterprise Security Module
"""

import logging
from typing import Any
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
import json

logger = logging.getLogger(__name__)

class PytectorSecurityMiddleware(BaseHTTPMiddleware):
    """Centralized security scanning for all external inputs"""
    
    def __init__(self, app, strict_mode: bool = True):
        super().__init__(app)
        self.strict_mode = strict_mode
        
        # Initialize pytector
        try:
            import pytector
            self.pytector = pytector
            self.enabled = True
            logger.info("Pytector security middleware initialized in strict mode")
        except ImportError:
            self.pytector = None
            self.enabled = False
            logger.warning("Pytector not available - security middleware disabled")
            logger.warning("Pytector not available - security middleware disabled")
    
    async def dispatch(self, request: Request, call_next):
        """Scan all incoming requests for security threats"""
        
        if not self.enabled:
            return await call_next(request)
        
        # Skip security scan for health checks and static content
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        try:
            # Scan URL parameters
            if request.query_params:
                self._scan_data(dict(request.query_params), "query_params")
            
            # Scan request body for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        try:
                            json_body = json.loads(body.decode())
                            self._scan_data(json_body, "request_body")
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # Scan raw body if not JSON
                            self._scan_data({"raw_body": body.decode(errors='ignore')}, "raw_body")
                        
                        # Re-create request with body for downstream processing
                        request._body = body
                except Exception as e:
                    logger.error(f"Body scanning failed: {e}")
                    # Continue - don't block on scanning errors unless strict mode
                    if self.strict_mode:
                        raise HTTPException(
                            status_code=422, 
                            detail=f"Request body security scan failed: {e}"
                        )
            
            # Scan headers for suspicious content
            suspicious_headers = {
                k: v for k, v in request.headers.items() 
                if k.lower() not in ['authorization', 'content-type', 'user-agent', 'accept']
            }
            if suspicious_headers:
                self._scan_data(suspicious_headers, "headers")
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            if self.strict_mode:
                raise HTTPException(
                    status_code=500,
                    detail="Security scanning failed"
                )
        
        return await call_next(request)
    
    def _scan_data(self, data: Any, data_type: str) -> None:
        """Perform pytector security scan on data"""
        if not self.enabled or not self.pytector:
            logger.debug(f"Pytector disabled - skipping scan for {data_type}")
            return
            
        try:
            # Check if pytector has scan method
            if not hasattr(self.pytector, 'scan'):
                logger.warning("Pytector module missing scan method - disabling security middleware")
                self.enabled = False
                return
                
            scan_result = self.pytector.scan(data, mode="strict")
            
            # Log scan details
            if scan_result and hasattr(scan_result, 'flagged') and scan_result.flagged:
                logger.warning(f"Security scan flagged {data_type}: {scan_result}")
                
                if self.strict_mode:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Security scan rejected {data_type}: potentially malicious content detected"
                    )
                    
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Pytector scan failed for {data_type}: {e}")
            # Disable on repeated failures
            self.enabled = False
            logger.warning("Disabling Pytector due to scan failures")
            return

def create_security_middleware(strict_mode: bool = True):
    """Factory function to create security middleware"""
    return lambda app: PytectorSecurityMiddleware(app, strict_mode=strict_mode)