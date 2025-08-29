#!/bin/bash

echo "========================================" 
echo "AutoResolve V3.0 - 100% Working"
echo "========================================"
echo ""

# Check backend
if ! curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "⚠️  Backend not running! Starting backend..."
    cd /Users/hawzhin/AutoResolve/autorez
    python3 backend_service_final.py &
    sleep 2
fi

# Launch UI
echo "✅ Launching AutoResolve UI..."
cd /Users/hawzhin/AutoResolve/AutoResolveUI

if [ ! -f "AutoResolveUI" ]; then
    echo "Building UI..."
    swiftc -parse-as-library Sources/AutoResolveUI/main.swift -o AutoResolveUI
fi

./AutoResolveUI

echo ""
echo "Application closed."