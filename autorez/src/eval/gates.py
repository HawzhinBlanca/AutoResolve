#!/usr/bin/env python3
"""
Performance Gates System - SLA Enforcement
AutoResolve Enterprise Compliance Module
"""

OPS = {
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
}

PERFORMANCE_GATES = {
    "processing_speed_x":    ("gte", 30,   "pipeline speed (× realtime)"),
    "peak_rss_gb":           ("lte", 16.0, "peak RAM during embeddings"),
    "ui_memory_mb":          ("lte", 200,  "UI memory ceiling"),
    "silence_sec_per_min":   ("lte", 0.5,  "silence detector runtime"),
    "transcription_rtf":     ("lte", 1.5,  "transcribe real-time factor"),
    "vjepa_sec_per_min":     ("lte", 5.0,  "V-JEPA time per video minute"),
    "api_sec_per_min":       ("lte", 3.0,  "OpenRouter time per video minute"),
    "api_cost_per_min":      ("lte", 0.05, "OpenRouter cost per minute"),
    "export_time_s":         ("lte", 2.0,  "EDL export time (s)"),
}

def verify_gates(metrics: dict):
    import math
    failures = []
    for name, (op, thr, desc) in PERFORMANCE_GATES.items():
        val = metrics.get(name)
        if val is None:
            failures.append(f"{name} missing")
            continue
        # Type and NaN validation
        if not isinstance(val, (int, float)) or (isinstance(val, float) and math.isnan(val)):
            failures.append(f"{name} invalid value: {val}")
            continue
        if op not in OPS:
            failures.append(f"unsupported op {op} for {name}")
            continue
        if not OPS[op](float(val), float(thr)):
            failures.append(f"{name}({val}) !{op} {thr}  ← {desc}")
    if failures:
        raise ValueError("Gates failed: " + "; ".join(failures))
    return True

if __name__ == "__main__":
    import argparse, json, sys, os
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="Verify gates")
    p.add_argument("--metrics", type=str, default=os.getenv("METRICS_PATH", "autoresz/artifacts/metrics.jsonl"))
    args = p.parse_args()

    if args.verify:
        # Load metrics from JSON or JSONL (use last record)
        try:
            with open(args.metrics, "r") as f:
                if args.metrics.endswith(".jsonl"):
                    last = None
                    for line in f:
                        if line.strip():
                            last = line
                    if not last:
                        raise ValueError("Empty metrics file")
                    metrics = json.loads(last)
                else:
                    metrics = json.load(f)
        except FileNotFoundError:
            print(f"Metrics file not found: {args.metrics}", file=sys.stderr)
            sys.exit(2)
        except json.JSONDecodeError as e:
            print(f"Invalid metrics JSON: {e}", file=sys.stderr)
            sys.exit(3)

        verify_gates(metrics)
        print("Gates: PASS")