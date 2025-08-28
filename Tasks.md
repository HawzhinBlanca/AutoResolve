# OpenRouter Integration Prompt for AutoResolve v3.0 Blueprint Update

Execute this integration plan to add OpenRouter capabilities to AutoResolve v3.0. No deviations. Every change must preserve existing guarantees.

## PHASE 1: Blueprint Update

Update `/Users/hawzhin/AutoResolve/Blueprint.md` with the following additions:

### Section 2.5: OpenRouter Dependencies (NEW)
```
openai>=1.0.0         # OpenRouter client via OpenAI SDK
tiktoken>=0.5.0       # Token counting for cost estimation
```

### Section 3: Configuration - Add OpenRouter Section
```ini
# conf/ops.ini - ADD this section after [broll]

[openrouter]
enabled = false                          # Default OFF - local only
base_url = https://openrouter.ai/api/v1
api_key_env = OPENROUTER_API_KEY
app_referrer = https://autoresolve.app
app_title = AutoResolve v3.0

# Model routing
narrative_model = cohere/command-r7b-12-2024
reasoning_model = qwen/qwq-32b
vision_model = openai/gpt-4o-mini

# Budgets & timeouts
max_input_tokens = 3500
max_output_tokens = 800
request_timeout_s = 20
daily_usd_cap = 2.50
max_calls_per_video = 6
target_api_sec_per_min = 3.0
```

## PHASE 2: Create OpenRouter Client

Create `/Users/hawzhin/AutoResolve/autorez/src/ops/openrouter.py`:

```python
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
```

## PHASE 3: Create Hybrid Evaluator

Create `/Users/hawzhin/AutoResolve/autorez/src/eval/hybrid_eval.py`:

```python
"""
Hybrid evaluation: measures quality/cost/latency with OpenRouter augmentation
"""

import time
import json
from pathlib import Path
import numpy as np
from src.ops.openrouter import get_client

def evaluate_hybrid(video_path: str, config) -> Dict:
    """Compare local-only vs OpenRouter-augmented pipeline"""
    
    results = {
        'local_only': {},
        'with_openrouter': {},
        'gates': {}
    }
    
    # Phase 1: Local-only baseline
    start = time.time()
    from src.director.creative_director import analyze_footage
    local_analysis = analyze_footage(video_path)
    local_time = time.time() - start
    
    results['local_only'] = {
        'time_seconds': local_time,
        'beats_found': len(local_analysis.get('beats', [])),
        'quality_score': compute_quality(local_analysis)
    }
    
    # Phase 2: With OpenRouter
    orc = get_client(config)
    if orc.enabled:
        start = time.time()
        
        # Augment narrative
        summary = {
            'duration': local_analysis['duration'],
            'energy_peaks': downsample(local_analysis['energy'], 1.0)[:100],
            'momentum_changes': local_analysis['momentum_changes'][:20]
        }
        
        enhanced = orc.json_reason(
            orc.narrative_model,
            {
                'task': 'identify_story_beats',
                'data': summary,
                'constraints': {
                    'max_beats': 10,
                    'min_confidence': 0.7
                }
            }
        )
        
        api_time = time.time() - start
        
        results['with_openrouter'] = {
            'time_seconds': local_time + api_time,
            'api_calls': orc.call_count,
            'api_cost_usd': orc.daily_spend,
            'beats_found': len(enhanced.get('beats', [])),
            'quality_score': compute_quality(enhanced)
        }
        
    # Phase 3: Gate checks
    video_duration_min = local_analysis['duration'] / 60
    
    results['gates'] = {
        'local_speed': (video_duration_min / local_time) > 30,  # 30x realtime
        'api_latency': api_time / video_duration_min < 3.0 if orc.enabled else True,  # <3s/min
        'quality_uplift': (results.get('with_openrouter', {}).get('quality_score', 0) / 
                          results['local_only']['quality_score']) > 1.3 if orc.enabled else False,
        'cost_per_min': (orc.daily_spend / video_duration_min) < 0.05 if orc.enabled else True,
        'all_pass': True  # Set after computing
    }
    
    results['gates']['all_pass'] = all([
        results['gates']['local_speed'],
        results['gates']['api_latency'],
        results['gates']['cost_per_min']
    ])
    
    return results

def compute_quality(analysis: Dict) -> float:
    """Score analysis quality (0-100)"""
    score = 0
    score += min(20, len(analysis.get('beats', [])) * 2)  # Story beats
    score += min(20, len(analysis.get('cuts', [])))  # Cut suggestions
    score += 20 if 'climax' in analysis else 0
    score += 20 if 'tension_curve' in analysis else 0
    score += 20 if analysis.get('confidence', 0) > 0.7 else 0
    return score

def downsample(data: List, target_fps: float) -> List:
    """Downsample to target FPS"""
    if not data:
        return []
    step = max(1, int(len(data) / (len(data) * target_fps / 30)))
    return data[::step]
```

## PHASE 4: Update Director Integration

Modify `/Users/hawzhin/AutoResolve/autorez/src/director/creative_director.py` at line 145:

```python
# ADD after local analysis complete (around line 145)
def analyze_footage(video_path: str, config=None) -> Dict:
    # ... existing local analysis code ...
    
    # NEW: OpenRouter augmentation (if enabled)
    if config and config.get('openrouter', 'enabled', fallback='false').lower() == 'true':
        from src.ops.openrouter import get_client
        orc = get_client(config)
        
        # Prepare compact summary
        summary = {
            'duration_s': analysis['duration'],
            'energy_curve': analysis['energy'][:600],  # Max 600 points
            'momentum_peaks': analysis['momentum_peaks'][:10],
            'novelty_peaks': analysis['novelty_peaks'][:10]
        }
        
        # Get narrative beats
        enhanced_beats = orc.json_chat(
            model=orc.narrative_model,
            system="Label narrative beats using three-act structure",
            user=json.dumps(summary)
        )
        
        if not enhanced_beats.get('_skipped'):
            analysis['narrative_beats'] = enhanced_beats.get('beats', [])
            analysis['openrouter_enhanced'] = True
    
    return analysis
```

## PHASE 5: Update Makefile

Add to `/Users/hawzhin/AutoResolve/autorez/Makefile` after line 85:

```makefile
# OpenRouter Integration
openrouter-setup:
	$(PY) -m pip install openai tiktoken
	@echo "Set OPENROUTER_API_KEY in your environment"

openrouter-test:
	$(PY) - << 'PY'
from src.ops.openrouter import OpenRouterClient
import configparser
cfg = configparser.ConfigParser()
cfg.read('conf/ops.ini')
client = OpenRouterClient(cfg)
print(client.sanity_check())
PY

hybrid-bench:
	$(PY) -m src.eval.hybrid_eval --video assets/pilots/scene.mp4
```

## PHASE 6: Testing Protocol

```bash
# 1. Install dependencies
cd /Users/hawzhin/AutoResolve/autorez
make openrouter-setup

# 2. Set API key
export OPENROUTER_API_KEY="your-key-here"

# 3. Test connection
make openrouter-test
# Expected: {"status": "ok"}

# 4. Run hybrid benchmark
make hybrid-bench
# Must show: all gates PASS

# 5. Enable in config
sed -i '' 's/enabled = false/enabled = true/' conf/ops.ini

# 6. Process with augmentation
python autoresolve_cli.py process test_video.mp4 --openrouter
```

## VALIDATION CHECKLIST

- [ ] V-JEPA/CLIP embeddings unchanged (local only)
- [ ] OpenRouter disabled by default
- [ ] All API calls have timeout < 20s
- [ ] Daily spend cap enforced ($2.50)
- [ ] Cache directory created and working
- [ ] JSON validation on all responses
- [ ] Local processing continues if OpenRouter fails
- [ ] Performance gates: 30x realtime maintained
- [ ] API latency < 3s per video minute
- [ ] Cost < $0.05 per video minute

## ADR Statement

"Added optional OpenRouter augmentation for narrative labeling and cut reasoning. Core embeddings remain local (V-JEPA/CLIP). Cloud assist is fail-closed with strict budgets. No network dependency in critical path."

Execute this plan exactly. Report gate results after implementation.