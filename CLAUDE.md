# CLAUDE.md - AI Agent Guide for AutoResolve v3.2

## Project Context

AutoResolve is a production video editing pipeline combining Swift UI frontend with Python backend for automated video processing. The system performs silence detection, transcription, narrative analysis, and exports to DaVinci Resolve.

## Critical Rules for AI Agents

### 1. File Protection Hierarchy
```
PROTECTED-CRITICAL: Never modify without explicit user approval
PROTECTED: Modify only with clear justification
SAFE-TO-MODIFY: Can update for optimization
```

### 2. Response Principles
- No verbose explanations unless requested
- Show exact commands and file paths
- Verify changes with concrete evidence
- Never guess - check actual files first

### 3. Architecture Boundaries
```
Frontend: /Users/hawzhin/AutoResolve/AutoResolveUI/
Backend:  /Users/hawzhin/AutoResolve/autorez/
Config:   /Users/hawzhin/AutoResolve/autorez/conf/
DO NOT create duplicate GUI folders
```

## Common Tasks

### Task: Debug Build Errors
```bash
# Check Swift errors
cd AutoResolveUI && swift build 2>&1 | grep -E "error:|warning:"

# Check Python syntax
python -m py_compile autorez/src/**/*.py

# Verify dependencies
pip list | grep -E "torch|transformers|fastapi"
```

### Task: Update Configuration
```python
# WRONG - breaks format
config = configparser.ConfigParser()
config['DEFAULT']['key'] = 'value'

# CORRECT - preserves format
with open('conf/embeddings.ini', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.startswith('default_model'):
        lines[i] = 'default_model = clip\n'
```

### Task: Add New Endpoint
1. Update `backend_service_final.py`:
```python
@app.post("/api/new_endpoint")
def new_endpoint(req: VideoRequest):
    # Implementation
    return {"ok": True}
```

2. Update Blueprint.md endpoints section
3. Add test to Makefile
4. Document in README.md

### Task: Performance Optimization
Check gates first:
```bash
python -m src.eval.gates --verify
```

If failing, identify bottleneck:
```python
# Add timing to suspicious module
import time
start = time.time()
# ... operation ...
elapsed = time.time() - start
print(f"Operation took {elapsed:.2f}s")
```

## Performance Requirements

### Hard Gates (Must Pass)
```python
GATES = {
    'processing_speed_x': ('gte', 30),      # 30x realtime minimum
    'peak_rss_gb': ('lte', 16.0),          # 16GB max
    'ui_memory_mb': ('lte', 200),          # 200MB UI max
    'silence_sec_per_min': ('lte', 0.5),   # 0.5s/min max
    'transcription_rtf': ('lte', 1.5),     # 1.5x realtime max
    'vjepa_sec_per_min': ('lte', 5.0),     # 5s/min max
    'export_time_s': ('lte', 2.0),         # 2s max
}
```

### Current Benchmarks
```
Processing: 51x realtime ✓
Memory: 3.2GB peak ✓
Silence: 0.18s/min ✓
Transcription: 0.9x RTF ✓
Export: 0.3s ✓
```

## Module-Specific Guidelines

### Embedders (src/embedders/)
- V-JEPA: Local only, never send to API
- CLIP: Default embedder
- Promotion: Only if V-JEPA beats CLIP by >15% with CI>0

### Operations (src/ops/)
- transcribe.py: Uses faster-whisper, NOT whisper
- silence.py: NumPy RMS only, NO librosa
- openrouter.py: Lives in ops/, not embedders/

### Director (src/director/)
- Returns JSON with beats, tension, emphasis
- All modules must be deterministic (seeded)
- Failures should not crash pipeline

## Error Patterns & Fixes

### Pattern: "Module not found"
```bash
# Check PYTHONPATH
export PYTHONPATH=/Users/hawzhin/AutoResolve/autorez/src:$PYTHONPATH
```

### Pattern: "Config value has comment"
```python
# Fix inline comments
sed -i '' 's/^default_model.*/default_model = clip/' conf/embeddings.ini
```

### Pattern: "WebSocket mismatch"
```swift
// Frontend expects /ws/status not /ws/progress
let wsURL = URL(string: "ws://localhost:8000/ws/status")!
```

## Update Procedures

### Adding Dependencies
1. Add to requirements.txt
2. Update Blueprint.md
3. Test gates still pass
4. Document in PR

### Modifying Protected Files
1. Create backup: `cp file file.backup`
2. Make changes
3. Run tests: `make test-pipeline`
4. Verify gates: `make verify-gates`

### OpenRouter Integration
```ini
# conf/ops.ini
[openrouter]
enabled = false  # Keep false by default
```
Only enable after verifying API key and budget caps.

## Testing Requirements

### Before Any PR
```bash
make test-pipeline  # Full pipeline test
make verify-gates   # Performance gates
make proof-pack     # Generate evidence
```

### Smoke Tests
```bash
curl -s http://localhost:8000/health | jq
python autoresolve_cli.py process test.mp4
```

## Security Considerations

- Never commit API keys
- OpenRouter disabled by default
- No telemetry or external calls without explicit enable
- Cache stays local in artifacts/cache/

## Communication Style

### When Reporting Issues
```
ISSUE: [Component] Brief description
EVIDENCE: Exact error or metric
FIX: Proposed solution with code
```

### When Suggesting Changes
```
CHANGE: [File] Line numbers
REASON: Performance/Security/Correctness
IMPACT: What improves
RISK: What could break
```

## Quick Reference

### Critical Paths
```
Config: autorez/conf/*.ini
Backend: autorez/backend_service_final.py
CLI: autorez/autoresolve_cli.py
Frontend: AutoResolveUI/Sources/AutoResolveUI/BackendService.swift
```

### Key Commands
```bash
# Start backend
cd autorez && uvicorn backend_service_final:app --port 8000

# Process video
python autoresolve_cli.py process video.mp4

# Check health
curl http://localhost:8000/health

# Run evaluation
make eval-ablate

# Verify system
make verify-gates
```

## Debugging Checklist

- [ ] Backend running on port 8000?
- [ ] Config files have clean values (no inline comments)?
- [ ] PYTHONPATH includes src/?
- [ ] All dependencies installed?
- [ ] Gates passing?
- [ ] WebSocket URL matches (/ws/status)?
- [ ] Artifacts directory exists?
- [ ] Test video available?

## Final Verification

Always end sessions by running:
```bash
make verify-gates && echo "System healthy" || echo "GATES FAILED"
```

---

This guide is the source of truth for AI agents working on AutoResolve. Follow these patterns exactly. When in doubt, check the Blueprint.md for specifications.