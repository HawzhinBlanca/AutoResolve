#!/usr/bin/env python3
"""
Analyze V-JEPA vs CLIP Results
"""

import json

results = {
  "top3": {
    "vjepa": 0.41818181818181815,
    "clip": 0.05454545454545454,
    "vjepa_ci": [0.290909081697464, 0.4218909442424774, 0.5636363625526428],
    "clip_ci": [0.0, 0.05501818284392357, 0.12727272510528564]
  },
  "mrr": {
    "vjepa": 0.2606060606060606,
    "clip": 0.02121212121212121,
    "vjepa_ci": [0.16969698667526245, 0.2629515528678894, 0.36363640427589417],
    "clip_ci": [0.0, 0.02139091119170189, 0.04848485440015793]
  },
  "perf": {
    "vjepa_sec_per_min": 2.007782300313314,
    "vjepa_peak_rss_gb": 0.6443328857421875
  }
}

print("🎯 V-JEPA-2 vs CLIP Evaluation Results")
print("=" * 70)

# Top-3 Accuracy
print("\n📊 Top-3 Accuracy (Higher is better):")
print(f"   V-JEPA-2: {results['top3']['vjepa']:.1%}")
print(f"   CLIP:     {results['top3']['clip']:.1%}")
improvement_top3 = (results['top3']['vjepa'] - results['top3']['clip']) / results['top3']['clip'] * 100
print(f"   Improvement: {improvement_top3:.1f}%")

# Check if CI intervals don't overlap (significant difference)
vjepa_ci_lower = results['top3']['vjepa_ci'][0]
clip_ci_upper = results['top3']['clip_ci'][2]
if vjepa_ci_lower > clip_ci_upper:
    print("   ✅ Statistically significant (95% CI)")

# MRR (Mean Reciprocal Rank)
print("\n📊 Mean Reciprocal Rank (Higher is better):")
print(f"   V-JEPA-2: {results['mrr']['vjepa']:.3f}")
print(f"   CLIP:     {results['mrr']['clip']:.3f}")
improvement_mrr = (results['mrr']['vjepa'] - results['mrr']['clip']) / results['mrr']['clip'] * 100
print(f"   Improvement: {improvement_mrr:.1f}%")

# Check if CI intervals don't overlap
vjepa_mrr_ci_lower = results['mrr']['vjepa_ci'][0]
clip_mrr_ci_upper = results['mrr']['clip_ci'][2]
if vjepa_mrr_ci_lower > clip_mrr_ci_upper:
    print("   ✅ Statistically significant (95% CI)")

# Performance metrics
print("\n⚡ Performance Metrics:")
print(f"   Processing speed: {results['perf']['vjepa_sec_per_min']:.2f} sec/min")
print(f"   Peak memory:      {results['perf']['vjepa_peak_rss_gb']:.2f} GB")

# Blueprint3 Quality Gates
print("\n✅ Blueprint3 Quality Gates:")

# 1. Quality: V-JEPA ≥ +15% vs CLIP on Top-3 AND MRR
quality_pass = improvement_top3 >= 15 and improvement_mrr >= 15
print(f"   Quality (≥15% improvement): {'✅ PASS' if quality_pass else '❌ FAIL'}")
print(f"      Top-3: {improvement_top3:.1f}% {'✅' if improvement_top3 >= 15 else '❌'}")
print(f"      MRR:   {improvement_mrr:.1f}% {'✅' if improvement_mrr >= 15 else '❌'}")

# 2. CI Bounds: 95% CI lower bound > 0
ci_pass = vjepa_ci_lower > 0 and vjepa_mrr_ci_lower > 0
print(f"   CI Bounds (lower > 0): {'✅ PASS' if ci_pass else '❌ FAIL'}")

# 3. Performance: ≤5.0 sec/min
perf_pass = results['perf']['vjepa_sec_per_min'] <= 5.0
print(f"   Performance (≤5.0 sec/min): {'✅ PASS' if perf_pass else '❌ FAIL'} ({results['perf']['vjepa_sec_per_min']:.2f})")

# 4. Memory: Peak RSS < 16 GB
mem_pass = results['perf']['vjepa_peak_rss_gb'] < 16
print(f"   Memory (<16 GB): {'✅ PASS' if mem_pass else '❌ FAIL'} ({results['perf']['vjepa_peak_rss_gb']:.2f} GB)")

# Promotion decision
all_gates_pass = quality_pass and ci_pass and perf_pass and mem_pass

print("\n" + "=" * 70)
print("🎯 PROMOTION DECISION:")
if all_gates_pass:
    print("   ✅ V-JEPA-2 PROMOTED as primary embedder!")
    print("   All quality gates passed successfully.")
else:
    print("   ❌ V-JEPA-2 NOT promoted (keeping CLIP baseline)")
    print("   Some quality gates failed.")

print("=" * 70)

# Save decision
decision = {
    "promote_vjepa": all_gates_pass,
    "improvements": {
        "top3": improvement_top3,
        "mrr": improvement_mrr
    },
    "gates": {
        "quality": quality_pass,
        "ci_bounds": ci_pass,
        "performance": perf_pass,
        "memory": mem_pass
    },
    "raw_results": results
}

with open("artifacts/promotion_decision.json", "w") as f:
    json.dump(decision, f, indent=2)

print(f"\n💾 Decision saved to artifacts/promotion_decision.json")