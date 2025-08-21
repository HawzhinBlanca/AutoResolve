# AutoResolve v3.0 Proof Pack Run Log

Generated: 2025-08-21T00:21:51.525923

## System Information
- Platform: macOS-15.6-arm64-arm-64bit
- Python: 3.11.5
- Working Directory: /Users/hawzhin/AutoResolve/autorez

## Component Status
- Structure Compliance: ❌ `verify_blueprint3_structure.py`
- Memory Management: ✅ `src/utils/memory.py`
- Caching System: ✅ `src/utils/cache.py`
- CLIP Embedder: ✅ `src/embedders/clip_embedder.py`
- V-JEPA Embedder: ✅ `src/embedders/vjepa_embedder.py`
- Alignment Module: ✅ `src/align/align_vjepa_to_clip.py`
- Scoring System: ✅ `src/scoring/broll_scoring.py`
- Evaluation Harness: ✅ `src/eval/ablate_vjepa_vs_clip.py`
- Director Stack: ✅ `src/director/creative_director.py`

## Test Results

## Quality Gates Summary

### Retrieval
- vjepa_improvement: ✅ (target: ≥15%, actual: 666.7%)
- ci_lower_bound: ✅ (target: >0, actual: Top3: 0.291, MRR: 0.170)
- vjepa_speed: ✅ (target: ≤5.0 sec/min, actual: 2.01 sec/min)
- clip_speed: ❌ (target: ≤2.0 sec/min, actual: not_tested)

### System
- memory: ✅ (target: <16GB, actual: 0.64GB)
- determinism: ✅ (target: yes, actual: verified)
- cache_hit: ❌ (target: >90%, actual: not_tested)

### Director
- f1_iou: ❌ (target: ≥0.60, actual: not_tested)
- pr_auc: ❌ (target: ≥0.65, actual: not_tested)
- speed: ❌ (target: ≤7.5 sec/min, actual: not_tested)

## Overall Compliance
- Gates Passed: 5/10 (50.0%)
- Blueprint3 Compliance: ⏳ 50.0%