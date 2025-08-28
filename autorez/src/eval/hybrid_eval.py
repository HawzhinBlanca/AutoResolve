"""
Hybrid evaluation: measures quality/cost/latency with OpenRouter augmentation
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any
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
    api_time = results.get('with_openrouter', {}).get('time_seconds', 0) - local_time if orc.enabled else 0
    
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