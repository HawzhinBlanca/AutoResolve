#!/usr/bin/env python3
"""
Bedrock Guardrails Integration - Enterprise Content Safety
AutoResolve Security Module for AWS Bedrock Guardrails
"""

import logging
import os
from typing import Any, Dict, List, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

class BedrockGuardrails:
    """AWS Bedrock Guardrails integration for content safety"""
    
    def __init__(self, guardrail_id: Optional[str] = None, guardrail_version: str = "DRAFT"):
        self.guardrail_id = guardrail_id or os.getenv("BEDROCK_GUARDRAIL_ID")
        self.guardrail_version = guardrail_version
        self.enabled = bool(self.guardrail_id)
        
        if self.enabled:
            try:
                self.client = boto3.client('bedrock-runtime')
                logger.info(f"Bedrock Guardrails initialized: {self.guardrail_id}")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"Bedrock client initialization failed: {e}")
                self.enabled = False
        else:
            logger.info("Bedrock Guardrails disabled - no BEDROCK_GUARDRAIL_ID set")
    
    def scan_text_content(self, text: str, source: str = "user_input") -> Dict[str, Any]:
        """Scan text content using Bedrock Guardrails"""
        if not self.enabled or not text.strip():
            return {"safe": True, "source": source, "guardrails_used": False}
        
        try:
            response = self.client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source="INPUT",
                content=[{
                    "text": {
                        "text": text,
                        "qualifiers": ["grounding_source"]
                    }
                }]
            )
            
            # Parse guardrails response
            action = response.get('action', 'NONE')
            assessments = response.get('assessments', [])
            
            is_safe = action == "NONE"
            
            result = {
                "safe": is_safe,
                "source": source,
                "guardrails_used": True,
                "action": action,
                "assessments": assessments,
                "guardrail_id": self.guardrail_id
            }
            
            if not is_safe:
                logger.warning(f"Bedrock Guardrails blocked content from {source}: {action}")
                
            return result
            
        except ClientError as e:
            logger.error(f"Bedrock Guardrails API error: {e}")
            return {
                "safe": False,  # Fail secure
                "source": source,
                "guardrails_used": False,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Bedrock Guardrails unexpected error: {e}")
            return {
                "safe": False,  # Fail secure
                "source": source, 
                "guardrails_used": False,
                "error": str(e)
            }
    
    def scan_request_data(self, data: Dict[str, Any], source: str = "api_request") -> Dict[str, Any]:
        """Scan structured request data for security threats"""
        if not self.enabled:
            return {"safe": True, "source": source, "guardrails_used": False}
        
        # Extract text content from structured data
        text_content = self._extract_text_content(data)
        
        if not text_content:
            return {"safe": True, "source": source, "guardrails_used": False}
        
        # Combine all text and scan
        combined_text = " ".join(text_content)
        return self.scan_text_content(combined_text, source)
    
    def _extract_text_content(self, obj: Any) -> List[str]:
        """Recursively extract text content from data structures"""
        text_content = []
        
        if isinstance(obj, str):
            text_content.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                text_content.extend(self._extract_text_content(value))
        elif isinstance(obj, list):
            for item in obj:
                text_content.extend(self._extract_text_content(item))
        
        return [t for t in text_content if isinstance(t, str) and t.strip()]

# Global instance for use across application
bedrock_guardrails = BedrockGuardrails()

def validate_content_safety(data: Any, source: str = "user_input") -> Dict[str, Any]:
    """Convenience function for content safety validation"""
    if isinstance(data, str):
        return bedrock_guardrails.scan_text_content(data, source)
    else:
        return bedrock_guardrails.scan_request_data(data, source)