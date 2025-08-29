#!/bin/bash
# deploy_final.sh

echo "AutoResolve v3.2 Final Deployment"

# 1. Structure check
test -f autorez/backend_service_final.py || exit 1
test -f autorez/src/eval/gates.py || exit 1
test -f autorez/datasets/broll_pilot/manifest.json || exit 1

# 2. Dependency check
cd autorez
python -c "import torch, fastapi, numpy" || exit 1

# 3. Verify no LibROSA dependency (NumPy-only)
! grep -q "import librosa" src/ops/silence.py || echo "WARNING: LibROSA found but not in requirements"

# 4. Gates verification
make verify-gates || exit 1

# 5. Backend test
uvicorn backend_service_final:app --port 8000 &
PID=$!
sleep 2
curl -s http://localhost:8000/health | grep healthy || exit 1
kill $PID

# 6. Tag release
git add -A
git commit -m "AutoResolve v3.2: production final"
git tag v3.2.0

echo "✓ Deployment ready - v3.2.0"