"""
OpenRouter client for AutoResolve v3.0
ADR: Augments local processing with LLM intelligence. Never replaces core embeddings.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from openai import OpenAI
import tiktoken

class OpenRouterClient:
    def __init__(self, config):
        self.enabled = config.get('openrouter', 'enabled', fallback='false').lower() == 'true'
        if not self.enabled:
            return
            
        self.api_key = os.getenv(config.get('openrouter', 'api_key_env', fallback='OPENROUTER_API_KEY'))
        if not self.api_key:
            self.enabled = False
            return
            
        self.client = OpenAI(
            base_url=config.get('openrouter', 'base_url'),
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": config.get('openrouter', 'app_referrer', fallback=''),
                "X-Title": config.get('openrouter', 'app_title', fallback='AutoResolve')
            }
        )
        
        self.narrative_model = config.get('openrouter', 'narrative_model')
        self.reasoning_model = config.get('openrouter', 'reasoning_model')
        self.vision_model = config.get('openrouter', 'vision_model')
        
        self.max_input_tokens = int(config.get('openrouter', 'max_input_tokens', fallback='3500'))
        self.max_output_tokens = int(config.get('openrouter', 'max_output_tokens', fallback='800'))
        self.timeout = float(config.get('openrouter', 'request_timeout_s', fallback='20'))
        self.daily_cap = float(config.get('openrouter', 'daily_usd_cap', fallback='2.50'))
        self.max_calls = int(config.get('openrouter', 'max_calls_per_video', fallback='6'))
        
        self.cache_dir = Path('artifacts/cache/openrouter')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.daily_spend = self._load_daily_spend()
        self.call_count = 0
        
    def json_chat(self, model: str, system: str, user: str, images: Optional[List[str]] = None) -> Dict:
        """Call OpenRouter with strict JSON response"""
        if not self.enabled or self.call_count >= self.max_calls or self.daily_spend >= self.daily_cap:
            return {"_skipped": True, "reason": "limits"}
            
        # Cache key
        content_hash = hashlib.sha256(f"{model}{system}{user}{images}".encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"{model.replace('/', '_')}_{content_hash}.json"
        
        if cache_file.exists():
            return json.loads(cache_file.read_text())
            
        # Build messages
        messages = [{"role": "system", "content": system + " Respond with valid JSON only."}]
        
        if images:
            content = [{"type": "text", "text": user}]
            for img in images[:3]:  # Max 3 images
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user})
            
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.max_output_tokens,
                temperature=0,
                timeout=self.timeout,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Track costs
            self._track_cost(response.usage)
            self.call_count += 1
            
            # Cache result
            cache_file.write_text(json.dumps(result))
            
            return result
            
        except Exception as e:
            return {"_error": str(e), "_skipped": True}
            
    def json_reason(self, model: str, payload: Dict) -> Dict:
        """Reasoning-specific call with structured input"""
        system = "You are a video editing assistant. Analyze the provided data and return structured JSON."
        user = json.dumps(payload)
        return self.json_chat(model, system, user)
        
    def _track_cost(self, usage):
        """Track API costs"""
        # Rough estimates based on OpenRouter pricing
        input_cost = (usage.prompt_tokens / 1000000) * 0.15  # $0.15/M tokens
        output_cost = (usage.completion_tokens / 1000000) * 0.60  # $0.60/M tokens
        self.daily_spend += (input_cost + output_cost)
        
    def _load_daily_spend(self) -> float:
        """Load today's spend from disk"""
        spend_file = self.cache_dir / f"spend_{time.strftime('%Y%m%d')}.json"
        if spend_file.exists():
            return json.loads(spend_file.read_text()).get('total', 0.0)
        return 0.0
        
    def sanity_check(self) -> Dict:
        """Test connectivity and model availability"""
        if not self.enabled:
            return {"status": "disabled"}
        try:
            resp = self.json_chat(
                self.narrative_model,
                "Test connection",
                "Return {'status': 'ok'}"
            )
            return resp
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Global singleton
_client = None

def get_client(config):
    global _client
    if _client is None:
        _client = OpenRouterClient(config)
    return _client