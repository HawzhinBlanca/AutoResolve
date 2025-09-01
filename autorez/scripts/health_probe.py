#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
import urllib.request

def main():
    base = os.getenv("BACKEND_BASE", "http://localhost:8000")
    url = f"{base.rstrip('/')}/health"
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=1.5) as resp:
        body = resp.read()
    dur_ms = int((time.time() - t0) * 1000)
    data = json.loads(body.decode("utf-8"))
    out = {
        "ok": bool(data.get("ok")),
        "ver": data.get("ver"),
        "latency_ms": dur_ms
    }
    artifacts = Path("Artifacts")
    artifacts.mkdir(exist_ok=True)
    (artifacts / "backend_health.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))

if __name__ == "__main__":
    sys.exit(main())


