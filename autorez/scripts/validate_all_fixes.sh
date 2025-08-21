#!/bin/bash
# Validation script for all bug fixes
# Runs comprehensive tests to ensure all fixes are working

set -e  # Exit on error

echo "=== AutoResolve Bug Fix Validation Suite ==="
echo "Date: $(date)"
echo "============================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local command="$2"
    
    echo -n "Testing $test_name... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((TESTS_FAILED++))
        echo "  Command: $command"
    fi
}

# 1. Test promotion logic
echo ""
echo "=== Phase 1: Math & Logic Fixes ==="
run_test "Promotion logic" "python -m pytest tests/test_promotion_logic.py -v --tb=short"
run_test "Score normalizer import" "python -c 'from src.scoring.score_normalizer import ScoreNormalizer; s=ScoreNormalizer(); print(s.get_weights_info())'"

# 2. Test memory management
echo ""
echo "=== Phase 2: Memory Management ==="
run_test "Memory guard" "python -m pytest tests/test_memory_guard.py -v --tb=short"
run_test "Thread-safe seeds" "python -c 'from src.utils.memory import set_seeds; set_seeds(1234)'"

# 3. Test input validation
echo ""
echo "=== Phase 3: Input Validation ==="
run_test "Validators" "python -m pytest tests/test_validators.py -v --tb=short"
run_test "Segment validator import" "python -c 'from src.validators.segment_validator import SegmentValidator'"
run_test "Duration validator import" "python -c 'from src.validators.duration_validator import DurationValidator'"

# 4. Test config validation
echo ""
echo "=== Phase 4: Configuration Validation ==="
run_test "Config validator" "python -m pytest tests/test_config_validator.py -v --tb=short"
run_test "Schema generation" "python -c 'from src.config.schema_validator import ConfigValidator; print(ConfigValidator.generate_default_config(\"embeddings\")[:50])'"

# 5. Test performance scripts
echo ""
echo "=== Phase 5: Performance Scripts ==="
run_test "Benchmark script syntax" "python scripts/benchmark_vjepa.py --help"
run_test "CLIP benchmark syntax" "python scripts/benchmark_clip.py --help"

# 6. Test JSON schemas
echo ""
echo "=== Phase 6: Output Schemas ==="
run_test "Schema imports" "python -c 'from src.schemas.output_schemas import TRANSCRIPT_SCHEMA, CUTS_SCHEMA'"

# 7. Integration tests
echo ""
echo "=== Integration Tests ==="

# Test config validation with actual file
if [ -f "conf/embeddings.ini" ]; then
    run_test "Real config validation" "python -c 'from src.config.schema_validator import ConfigValidator; ConfigValidator.validate_config(\"conf/embeddings.ini\", \"embeddings\")'"
else
    echo -e "${YELLOW}⚠ Skipping real config test (file not found)${NC}"
fi

# Test memory guard in action
run_test "Memory guard execution" "python -c '
from src.utils.memory_guard import MemoryGuard
guard = MemoryGuard(max_gb=16)
with guard.protected_execution(\"test\"):
    params = guard.get_current_params()
    assert params[\"fps\"] == 1.0
print(\"Memory guard working\")
'"

# Test promotion logic with real data
run_test "Promotion decision" "python -c '
from src.utils.promotion import promote_vjepa
results = {
    \"top3\": {\"vjepa\": 0.8, \"clip\": 0.6, \"vjepa_ci\": [0.75, 0.85], \"clip_ci\": [0.55, 0.65]},
    \"mrr\": {\"vjepa\": 0.7, \"clip\": 0.55, \"vjepa_ci\": [0.65, 0.75], \"clip_ci\": [0.50, 0.60]}
}
assert promote_vjepa(results, 4.0) == True
print(\"Promotion logic correct\")
'"

# 8. Check all Python files for syntax errors
echo ""
echo "=== Syntax Validation ==="
for file in src/utils/*.py src/validators/*.py src/config/*.py src/scoring/*.py src/schemas/*.py; do
    if [ -f "$file" ]; then
        run_test "Syntax check: $(basename $file)" "python -m py_compile $file"
    fi
done

# 9. Run all unit tests together
echo ""
echo "=== Full Test Suite ==="
if command -v pytest &> /dev/null; then
    run_test "All unit tests" "python -m pytest tests/ -v --tb=short"
else
    echo -e "${YELLOW}⚠ pytest not installed, skipping full suite${NC}"
fi

# Summary
echo ""
echo "============================================"
echo "=== VALIDATION SUMMARY ==="
echo "============================================"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All validations passed successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some validations failed. Please review the errors above.${NC}"
    exit 1
fi