#!/bin/bash
# Daily validation checkpoint script

echo "=== DAILY VALIDATION CHECKPOINT ==="
date
echo ""

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $1 exists"
        return 0
    else
        echo "  ❌ $1 missing"
        return 1
    fi
}

# Function to run Python test
run_test() {
    echo "  Testing: $1"
    if python3 -c "$2" 2>/dev/null; then
        echo "    ✅ Passed"
        return 0
    else
        echo "    ❌ Failed"
        return 1
    fi
}

# Track failures
FAILURES=0

echo "1. Code Compilation Check:"
echo "  Checking Python syntax..."
find src -name "*.py" -exec python3 -m py_compile {} \; 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✅ All Python files compile"
else
    echo "  ❌ Some Python files have syntax errors"
    ((FAILURES++))
fi

echo ""
echo "2. Critical Files Check:"
check_file "src/scoring/broll_quality.py" || ((FAILURES++))
check_file "src/scoring/broll_scoring.py" || ((FAILURES++))
check_file "src/utils/telemetry.py" || ((FAILURES++))
check_file "assets/test_30min.mp4" || ((FAILURES++))
check_file "EXECUTION_PLAN.md" || ((FAILURES++))

echo ""
echo "3. Quality Scoring Test:"
run_test "Quality scoring" "
from src.scoring.broll_quality import calculate_broll_quality_score
segment = {'t0': 10, 't1': 20}
score = calculate_broll_quality_score(segment)
assert 0.0 <= score <= 1.0, f'Score {score} out of range'
print(f'Score: {score:.3f}')
" || ((FAILURES++))

echo ""
echo "4. Telemetry Test:"
run_test "Telemetry recording" "
from src.utils.telemetry import TelemetryCollector
t = TelemetryCollector()
t.record_broll_quality('test_seg', {'sharpness': 0.5}, 0.5)
print('Telemetry recorded')
" || ((FAILURES++))

echo ""
echo "5. 30-Minute Video Check:"
if [ -f "assets/test_30min.mp4" ]; then
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 assets/test_30min.mp4 2>/dev/null)
    DURATION_MIN=$(echo "scale=1; $DURATION / 60" | bc)
    echo "  ✅ 30-min test video exists (${DURATION_MIN} minutes)"
    
    # Check if approximately 30 minutes
    if (( $(echo "$DURATION_MIN >= 29 && $DURATION_MIN <= 31" | bc -l) )); then
        echo "  ✅ Duration is correct"
    else
        echo "  ❌ Duration ${DURATION_MIN} is not ~30 minutes"
        ((FAILURES++))
    fi
else
    echo "  ❌ 30-min test video missing"
    ((FAILURES++))
fi

echo ""
echo "6. Memory Usage Check:"
MEMORY_GB=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().used / (1024**3):.1f}')")
echo "  Current memory usage: ${MEMORY_GB} GB"
AVAILABLE_GB=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().available / (1024**3):.1f}')")
echo "  Available memory: ${AVAILABLE_GB} GB"

if (( $(echo "$AVAILABLE_GB > 4" | bc -l) )); then
    echo "  ✅ Sufficient memory available"
else
    echo "  ⚠️ Low memory available"
fi

echo ""
echo "7. Compliance Summary:"
python3 -c "
from pathlib import Path
import json

# Count completed tasks
completed = 0
placeholders = 0

# Check for placeholders
for file in Path('src').rglob('*.py'):
    content = file.read_text()
    if 'placeholder' in content.lower() or 'todo' in content.lower():
        placeholders += 1

# Check Day 1 deliverables
deliverables = {
    'B-roll quality scoring': Path('src/scoring/broll_quality.py').exists(),
    'Telemetry updates': Path('src/utils/telemetry.py').exists(),
    '30-min test video': Path('assets/test_30min.mp4').exists(),
    'Execution plan': Path('EXECUTION_PLAN.md').exists()
}

for name, exists in deliverables.items():
    if exists:
        completed += 1
        print(f'  ✅ {name}')
    else:
        print(f'  ❌ {name}')

print(f'\nDay 1 Completion: {completed}/{len(deliverables)} tasks')
if placeholders > 0:
    print(f'⚠️ Warning: {placeholders} files still contain placeholders')
" || ((FAILURES++))

echo ""
echo "=== VALIDATION SUMMARY ==="
if [ $FAILURES -eq 0 ]; then
    echo "✅ DAY 1 COMPLETE: All validation checks passed!"
    echo "Ready to proceed to Day 2: Memory & Performance Validation"
else
    echo "❌ DAY 1 INCOMPLETE: $FAILURES validation check(s) failed"
    echo "Please fix issues before proceeding to Day 2"
fi

echo ""
echo "Next steps for Day 2:"
echo "  - Test memory usage with 30-min video"
echo "  - Implement adaptive degradation"
echo "  - Validate transcription RTF"
echo "  - Optimize performance if needed"

exit $FAILURES