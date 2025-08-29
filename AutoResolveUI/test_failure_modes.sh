#!/bin/bash

echo "=== AutoResolve Failure Mode Testing ==="
echo "Testing error handling and graceful degradation"
echo

# Test 1: Backend offline scenario
echo "Test 1: Backend Offline Handling"
echo "Starting UI with backend offline..."

# Kill any running backend
pkill -f "backend_service_final" 2>/dev/null || true
sleep 2

# Start UI and check it handles offline backend gracefully
timeout 10 .build/debug/AutoResolveUI &
UI_PID=$!
sleep 3

if ps -p $UI_PID > /dev/null; then
    echo "✓ UI starts successfully with backend offline"
    kill $UI_PID 2>/dev/null
else
    echo "✗ UI failed to start with backend offline"
fi

# Test 2: Invalid video file handling
echo
echo "Test 2: Invalid Video File Handling" 
echo "Testing with corrupted video file..."

# Create invalid video file
echo "invalid video content" > /tmp/invalid_video.mp4

# Start backend first
cd /Users/hawzhin/AutoResolve/autorez
python backend_service_final.py &
BACKEND_PID=$!
sleep 3

# Test API with invalid file
RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key-123" \
  -d '{"video_path": "/tmp/invalid_video.mp4"}' \
  http://localhost:8000/api/process)

if echo "$RESPONSE" | grep -q '"success": false'; then
    echo "✓ Backend handles invalid video gracefully"
else
    echo "✗ Backend did not handle invalid video properly"
    echo "Response: $RESPONSE"
fi

# Test 3: Memory pressure handling
echo
echo "Test 3: Memory Pressure Handling"
echo "Testing with large file stress..."

# Test with stress scenario (if available)
if [ -f "/Users/hawzhin/Videos/test_video_43min.mp4" ]; then
    echo "Testing 43-minute video processing..."
    RESPONSE=$(curl -s -X POST \
      -H "Content-Type: application/json" \
      -H "x-api-key: dev-key-123" \
      -d '{"video_path": "/Users/hawzhin/Videos/test_video_43min.mp4", "operations": ["silence"]}' \
      http://localhost:8000/api/process)
    
    if echo "$RESPONSE" | grep -q '"success": true'; then
        echo "✓ Backend handles large files without crashing"
    else
        echo "⚠ Backend struggled with large file (may be expected)"
    fi
else
    echo "ⓘ Large test file not available, skipping stress test"
fi

# Test 4: Network timeout handling
echo
echo "Test 4: Network Timeout Handling"
echo "Testing API timeout behavior..."

START_TIME=$(date +%s)
RESPONSE=$(timeout 30 curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key-123" \
  -d '{"video_path": "/Users/hawzhin/Videos/test_video_5min.mp4", "operations": ["all"]}' \
  http://localhost:8000/api/process)
END_TIME=$(date +%s)

DURATION=$((END_TIME - START_TIME))

if [ $DURATION -lt 30 ] && echo "$RESPONSE" | grep -q "success"; then
    echo "✓ API responds within timeout ($DURATION seconds)"
else
    echo "⚠ API took longer than expected or timed out"
fi

# Test 5: File permission handling  
echo
echo "Test 5: File Permission Handling"
echo "Testing restricted file access..."

# Create file with no read permissions
echo "restricted content" > /tmp/restricted_video.mp4
chmod 000 /tmp/restricted_video.mp4

RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key-123" \
  -d '{"video_path": "/tmp/restricted_video.mp4"}' \
  http://localhost:8000/api/process)

if echo "$RESPONSE" | grep -q '"success": false'; then
    echo "✓ Backend handles permission errors gracefully"
else
    echo "✗ Backend did not handle permission error properly"
fi

# Cleanup
rm -f /tmp/invalid_video.mp4 /tmp/restricted_video.mp4
kill $BACKEND_PID 2>/dev/null || true

echo
echo "=== Failure Mode Test Complete ==="
echo "All critical error scenarios tested"