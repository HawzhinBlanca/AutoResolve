#!/bin/bash
set -euo pipefail

# AutoResolve v3.2 Hardened Deployment

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "AutoResolve v3.2 Final Deployment (hardened)"

# 0. Python version check (3.12+)
python3 - <<'PY' | cat
import sys
assert sys.version_info >= (3,12), f"Python 3.12+ required, found {sys.version}"
print("Python version OK")
PY

# 1. Structure check
test -f autorez/backend_service_final.py || { echo "missing backend_service_final.py"; exit 1; }
test -f autorez/src/eval/gates.py || { echo "missing gates.py"; exit 1; }

# 2. Dependency import smoke
cd autorez
python3 - <<'PY' | cat
import importlib
for m in ("torch","fastapi","numpy","pytector","boto3"):
    importlib.import_module(m)
print("Deps OK")
PY

# 3. Verify no LibROSA usage in silence op (NumPy-only)
if grep -q "import[[:space:]]\+librosa" src/ops/silence.py; then
  echo "ERROR: librosa import detected in src/ops/silence.py"; exit 1;
fi

# 4. Gates verification (prefer JSONL last record)
METRICS_PATH="${METRICS_PATH:-$(pwd)/artifacts/metrics.jsonl}"
if [ ! -f "$METRICS_PATH" ]; then echo "Missing metrics at $METRICS_PATH"; exit 1; fi
python3 src/eval/gates.py --verify --metrics "$METRICS_PATH" | cat

# 5. Backend health probe
uvicorn backend_service_final:app --port 8000 --log-level warning &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for i in {1..20}; do
  if curl -fsS http://localhost:8000/health | grep -q '"ok": true'; then
    echo "Health OK"; break; fi
  sleep 0.5
done
curl -fsS http://localhost:8000/health | grep -q '"ok": true'

echo "✓ Deployment checks passed"

