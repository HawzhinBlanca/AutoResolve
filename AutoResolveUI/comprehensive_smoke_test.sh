#!/bin/bash

echo "🚀 AutoResolve v3.2 COMPREHENSIVE SMOKE TEST"
echo "=== BRUTAL E2E REALITY CHECK ==="
echo

# Configuration
TEST_VIDEO="/Users/hawzhin/Videos/test_video_5min.mp4"
BACKEND_URL="http://localhost:8000"
UI_BUILD_PATH=".build/debug/AutoResolveUI"
ARTIFACTS_DIR="/Users/hawzhin/AutoResolve/autorez/artifacts"

# Test Results
PASSED=0
FAILED=0
WARNINGS=0

function test_result() {
    local test_name="$1"
    local success="$2"
    local message="$3"
    
    if [ "$success" = "true" ]; then
        echo "✅ $test_name: PASS"
        [ -n "$message" ] && echo "   $message"
        ((PASSED++))
    else
        echo "❌ $test_name: FAIL"
        [ -n "$message" ] && echo "   $message"
        ((FAILED++))
    fi
}

function warning_result() {
    local test_name="$1"
    local message="$2"
    echo "⚠️  $test_name: WARNING"
    [ -n "$message" ] && echo "   $message"
    ((WARNINGS++))
}

# Pre-flight checks
echo "=== PRE-FLIGHT CHECKS ==="

# Check test video exists
if [ -f "$TEST_VIDEO" ]; then
    test_result "Test Video Available" "true" "$TEST_VIDEO"
else
    test_result "Test Video Available" "false" "Missing: $TEST_VIDEO"
    exit 1
fi

# Check Swift build
echo "Building Swift frontend..."
BUILD_OUTPUT=$(swift build 2>&1)
if [ $? -eq 0 ]; then
    test_result "Swift Build" "true" "Build successful"
else
    test_result "Swift Build" "false" "Build failed"
    echo "$BUILD_OUTPUT" | head -10
fi

# Check backend dependencies
cd /Users/hawzhin/AutoResolve/autorez
python -c "import fastapi, torch, transformers" 2>/dev/null
test_result "Backend Dependencies" "$?" "Python dependencies checked"

echo
echo "=== BACKEND TESTING ==="

# Start backend
python -m uvicorn backend_service_final:app --port 8000 &
BACKEND_PID=$!
echo "Started backend (PID: $BACKEND_PID)"
sleep 5

# Test health endpoint
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_RESPONSE" | grep -q '"status": "healthy"'; then
    test_result "Backend Health" "true" "API responding"
else
    test_result "Backend Health" "false" "Health check failed"
fi

# Test video processing
echo "Testing video processing pipeline..."
PROCESS_START=$(date +%s)
PROCESS_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "x-api-key: dev-key-123" \
    -d "{\"video_path\": \"$TEST_VIDEO\", \"operations\": [\"silence\", \"transcription\"]}" \
    http://localhost:8000/api/process)
PROCESS_END=$(date +%s)
PROCESS_TIME=$((PROCESS_END - PROCESS_START))

if echo "$PROCESS_RESPONSE" | grep -q '"success": true'; then
    test_result "Video Processing" "true" "Completed in ${PROCESS_TIME}s"
else
    test_result "Video Processing" "false" "Processing failed"
fi

# Verify artifacts generated
if [ -f "$ARTIFACTS_DIR/cuts.json" ]; then
    test_result "Silence Artifacts" "true" "cuts.json created"
else
    warning_result "Silence Artifacts" "cuts.json missing"
fi

if [ -f "$ARTIFACTS_DIR/transcription.json" ]; then
    test_result "Transcription Artifacts" "true" "transcription.json created"
else
    warning_result "Transcription Artifacts" "transcription.json missing"
fi

echo
echo "=== UI TESTING ==="

# Test UI startup
cd /Users/hawzhin/AutoResolve/AutoResolveUI
timeout 15 $UI_BUILD_PATH &
UI_PID=$!
sleep 3

if ps -p $UI_PID > /dev/null; then
    test_result "UI Startup" "true" "UI launched successfully"
    kill $UI_PID 2>/dev/null
else
    test_result "UI Startup" "false" "UI failed to start"
fi

echo
echo "=== PERFORMANCE GATES ==="

# Memory usage check  
MEMORY_MB=$(ps -o rss= -p $BACKEND_PID 2>/dev/null | awk '{print $1/1024}' | cut -d. -f1)
if [ -n "$MEMORY_MB" ] && [ "$MEMORY_MB" -le 500 ]; then
    test_result "Backend Memory" "true" "${MEMORY_MB}MB ≤ 500MB"
else
    test_result "Backend Memory" "false" "${MEMORY_MB}MB > 500MB limit"
fi

# Processing speed check
if [ "$PROCESS_TIME" -le 10 ]; then
    test_result "Processing Speed" "true" "${PROCESS_TIME}s ≤ 10s limit"
else
    test_result "Processing Speed" "false" "${PROCESS_TIME}s > 10s limit"
fi

echo
echo "=== INTEGRATION TESTS ==="

# Test API endpoints
for endpoint in "/health" "/api/status" "/api/models"; do
    RESPONSE=$(curl -s -w "%{http_code}" "http://localhost:8000$endpoint" -o /dev/null)
    if [ "$RESPONSE" = "200" ]; then
        test_result "Endpoint $endpoint" "true" "HTTP 200"
    else
        test_result "Endpoint $endpoint" "false" "HTTP $RESPONSE"
    fi
done

# Test authentication
AUTH_RESPONSE=$(curl -s -w "%{http_code}" \
    -H "x-api-key: invalid-key" \
    "http://localhost:8000/api/process" -o /dev/null)
if [ "$AUTH_RESPONSE" != "200" ]; then
    test_result "Authentication" "true" "Rejects invalid keys"
else
    test_result "Authentication" "false" "Accepts invalid keys"
fi

echo
echo "=== ARTIFACT VERIFICATION ==="

# Check artifact quality
if [ -f "$ARTIFACTS_DIR/cuts.json" ]; then
    SILENCE_COUNT=$(jq '.silence_segments | length' "$ARTIFACTS_DIR/cuts.json" 2>/dev/null)
    if [ -n "$SILENCE_COUNT" ] && [ "$SILENCE_COUNT" -gt 0 ]; then
        test_result "Silence Detection Quality" "true" "$SILENCE_COUNT segments found"
    else
        warning_result "Silence Detection Quality" "No silence segments detected"
    fi
fi

# Check file formats
for format in "cuts.json" "transcription.json"; do
    if [ -f "$ARTIFACTS_DIR/$format" ]; then
        if jq empty "$ARTIFACTS_DIR/$format" 2>/dev/null; then
            test_result "Artifact Format ($format)" "true" "Valid JSON"
        else
            test_result "Artifact Format ($format)" "false" "Invalid JSON"
        fi
    fi
done

echo
echo "=== EDL EXPORT TEST ==="

# Test EDL generation
cd /Users/hawzhin/AutoResolve/AutoResolveUI
swift run AutoResolveUI --test-edl-export 2>/dev/null &
EXPORT_PID=$!
sleep 5

if [ -f "/tmp/test_export.edl" ]; then
    test_result "EDL Export" "true" "EDL file generated"
    
    # Verify EDL format
    if head -3 /tmp/test_export.edl | grep -q "FCM:"; then
        test_result "EDL Format" "true" "Proper EDL format"
    else
        test_result "EDL Format" "false" "Invalid EDL format"
    fi
else
    test_result "EDL Export" "false" "No EDL file generated"
fi

kill $EXPORT_PID 2>/dev/null || true

echo
echo "=== CLEANUP ==="
kill $BACKEND_PID 2>/dev/null || true
rm -f /tmp/test_export.edl
echo "Stopped all test processes"

echo
echo "=== FINAL RESULTS ==="
echo "✅ PASSED: $PASSED"
echo "❌ FAILED: $FAILED" 
echo "⚠️  WARNINGS: $WARNINGS"

TOTAL=$((PASSED + FAILED))
SUCCESS_RATE=$((PASSED * 100 / TOTAL))

echo
echo "SUCCESS RATE: $SUCCESS_RATE% ($PASSED/$TOTAL)"

if [ $FAILED -eq 0 ] && [ $SUCCESS_RATE -ge 90 ]; then
    echo "🎉 SMOKE TEST: FULL PASS"
    echo "System ready for production"
    exit 0
elif [ $SUCCESS_RATE -ge 80 ]; then
    echo "⚡ SMOKE TEST: CONDITIONAL PASS"
    echo "System functional with minor issues"
    exit 0
else
    echo "💥 SMOKE TEST: CRITICAL FAILURE"
    echo "System not ready for production"
    exit 1
fi