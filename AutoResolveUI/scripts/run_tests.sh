#!/bin/bash

#
# run_tests.sh - Comprehensive test runner for AutoResolve
# Usage: ./scripts/run_tests.sh [test-type] [options]
# Test types: unit, integration, ui, load, security, all
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AUTOREZ_ROOT="$(dirname "$PROJECT_ROOT")/autorez"

# Test configuration
UNIT_TEST_TIMEOUT=300      # 5 minutes
INTEGRATION_TEST_TIMEOUT=600  # 10 minutes
UI_TEST_TIMEOUT=1800      # 30 minutes
LOAD_TEST_TIMEOUT=3600    # 60 minutes
SECURITY_TEST_TIMEOUT=900 # 15 minutes

# Logging
LOG_DIR="$PROJECT_ROOT/test-logs"
mkdir -p "$LOG_DIR"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_DIR/test-run.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_DIR/test-run.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_DIR/test-run.log" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_DIR/test-run.log"
}

# Portable capitalization (uppercase first letter)
cap() {
    local s="$1"
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$s" <<'PY'
import sys
s = sys.argv[1] if len(sys.argv) > 1 else ""
print((s[:1].upper() + s[1:]) if s else "")
PY
    else
        # Fallback without python: return as-is
        echo "$s"
    fi
}

# Timeout wrapper: prefers GNU timeout, falls back to `python3 -m timeout`, else runs without timeout
run_with_timeout() {
    local seconds="$1"; shift
    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$seconds" "$@"
    elif command -v timeout >/dev/null 2>&1 && timeout --help 2>&1 | grep -qi -e "coreutils" -e "busybox"; then
        timeout "$seconds" "$@"
    elif python3 -c 'import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec("timeout") else 1)' >/dev/null 2>&1; then
        python3 -m timeout "$seconds" -- "$@"
    else
        log_warning "No compatible timeout utility found; running without timeout: $*"
        "$@"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Swift
    if ! command -v swift &> /dev/null; then
        log_error "Swift not found. Please install Xcode and command line tools."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.10+."
        exit 1
    fi
    
    # Check ffmpeg for test media generation
    if ! command -v ffmpeg &> /dev/null; then
        log_warning "FFmpeg not found. Some tests may not work properly."
    fi
    
    # Verify project structure
    if [[ ! -f "$PROJECT_ROOT/Package.swift" ]]; then
        log_error "Package.swift not found in $PROJECT_ROOT"
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Setup test environment
setup_test_environment() {
    log "Setting up test environment..."
    
    # Create test directories
    mkdir -p "$LOG_DIR"
    mkdir -p "/tmp/AutoResolveTests"
    mkdir -p "$HOME/AutoResolveTestMedia"
    
    # Generate test media files if ffmpeg is available
    if command -v ffmpeg &> /dev/null; then
        log "Generating test media files..."
        
        # Test video file (10 seconds, 1280x720, 30fps)
        if [[ ! -f "$HOME/AutoResolveTestMedia/test_video.mp4" ]]; then
            ffmpeg -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 \
                -c:v libx264 -preset ultrafast \
                "$HOME/AutoResolveTestMedia/test_video.mp4" \
                -y -loglevel quiet
        fi
        
        # Test audio file (10 seconds, 1kHz tone)
        if [[ ! -f "$HOME/AutoResolveTestMedia/test_audio.wav" ]]; then
            ffmpeg -f lavfi -i sine=frequency=1000:duration=10 \
                -c:a pcm_s16le \
                "$HOME/AutoResolveTestMedia/test_audio.wav" \
                -y -loglevel quiet
        fi
        
        # High resolution test video for performance tests
        if [[ ! -f "$HOME/AutoResolveTestMedia/test_video_4k.mp4" ]]; then
            ffmpeg -f lavfi -i testsrc=duration=30:size=3840x2160:rate=30 \
                -c:v libx264 -preset ultrafast \
                "$HOME/AutoResolveTestMedia/test_video_4k.mp4" \
                -y -loglevel quiet
        fi
    fi
    
    # Set environment variables
    export UITEST_MODE=1
    export AR_TEST_MODE=1
    export AR_MEDIA_ROOT="$HOME/AutoResolveTestMedia"
    
    log_success "Test environment setup completed"
}

# Build project
build_project() {
    local configuration=${1:-debug}
    log "Building project in $configuration configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Resolve dependencies
    swift package resolve
    
    # Build project
    if ! swift build --configuration "$configuration" -v 2>&1 | tee "$LOG_DIR/build-$configuration.log"; then
        log_error "Build failed in $configuration configuration"
        return 1
    fi
    
    log_success "Build completed successfully in $configuration configuration"
}

# Start backend service
start_backend() {
    log "Starting AutoResolve backend service..."
    
    if [[ ! -f "$AUTOREZ_ROOT/backend_service_final.py" ]]; then
        log_error "Backend service not found at $AUTOREZ_ROOT/backend_service_final.py"
        return 1
    fi
    
    cd "$AUTOREZ_ROOT"
    
    # Install Python dependencies if needed
    if [[ ! -f "venv/bin/activate" ]]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Start backend service in background
    python3 backend_service_final.py > "$LOG_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$LOG_DIR/backend.pid"
    
    # Wait for backend to start
    log "Waiting for backend service to start..."
    for i in {1..30}; do
        if curl --fail --silent http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Backend service started successfully (PID: $BACKEND_PID)"
            return 0
        fi
        sleep 1
    done
    
    log_error "Backend service failed to start within 30 seconds"
    return 1
}

# Stop backend service
stop_backend() {
    if [[ -f "$LOG_DIR/backend.pid" ]]; then
        local pid=$(cat "$LOG_DIR/backend.pid")
        log "Stopping backend service (PID: $pid)..."
        
        if kill "$pid" 2>/dev/null; then
            log_success "Backend service stopped"
        else
            log_warning "Backend service was not running or already stopped"
        fi
        
        rm -f "$LOG_DIR/backend.pid"
    fi
}

# Run unit tests
run_unit_tests() {
    log "Running unit tests..."
    
    cd "$PROJECT_ROOT"
    
    local start_time=$(date +%s)
    
    if run_with_timeout "$UNIT_TEST_TIMEOUT" swift test \
        --filter "AutoResolveUITests.TestFramework" \
        --configuration debug \
        --parallel \
        --enable-code-coverage \
        2>&1 | tee "$LOG_DIR/unit-tests.log"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Unit tests completed in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "Unit tests failed after ${duration}s"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    log "Running integration tests..."
    
    # Start backend service
    if ! start_backend; then
        log_error "Failed to start backend service for integration tests"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    local start_time=$(date +%s)
    local result=0
    
    if run_with_timeout "$INTEGRATION_TEST_TIMEOUT" swift test \
        --filter "AutoResolveUITests.IntegrationTestFramework" \
        --configuration debug \
        --parallel \
        --enable-code-coverage \
        2>&1 | tee "$LOG_DIR/integration-tests.log"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Integration tests completed in ${duration}s"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "Integration tests failed after ${duration}s"
        result=1
    fi
    
    stop_backend
    return $result
}

# Run UI tests
run_ui_tests() {
    log "Running UI tests..."
    
    # Build in debug configuration for UI testing
    if ! build_project "debug"; then
        log_error "Failed to build project for UI testing"
        return 1
    fi
    
    # Start backend service
    if ! start_backend; then
        log_error "Failed to start backend service for UI tests"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    local start_time=$(date +%s)
    local result=0
    
    if run_with_timeout "$UI_TEST_TIMEOUT" swift test \
        --filter "AutoResolveUITests.UITestFramework" \
        --configuration debug \
        --enable-code-coverage \
        2>&1 | tee "$LOG_DIR/ui-tests.log"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "UI tests completed in ${duration}s"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "UI tests failed after ${duration}s"
        result=1
    fi
    
    stop_backend
    return $result
}

# Run load tests
run_load_tests() {
    log "Running load tests..."
    
    # Build in release configuration for accurate performance testing
    if ! build_project "release"; then
        log_error "Failed to build project for load testing"
        return 1
    fi
    
    # Start backend service
    if ! start_backend; then
        log_error "Failed to start backend service for load tests"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    local start_time=$(date +%s)
    local result=0
    
    # Set performance test environment
    export AR_PERFORMANCE_MODE=1
    
    if run_with_timeout "$LOAD_TEST_TIMEOUT" swift test \
        --filter "AutoResolveUITests.LoadTestFramework" \
        --configuration release \
        2>&1 | tee "$LOG_DIR/load-tests.log"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Load tests completed in ${duration}s"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "Load tests failed after ${duration}s"
        result=1
    fi
    
    unset AR_PERFORMANCE_MODE
    stop_backend
    return $result
}

# Run security tests
run_security_tests() {
    log "Running security tests..."
    
    cd "$PROJECT_ROOT"
    
    local start_time=$(date +%s)
    local result=0
    
    # Security-specific environment
    export AR_SECURITY_TEST_MODE=1
    
    if run_with_timeout "$SECURITY_TEST_TIMEOUT" swift test \
        --filter "SecurityTest" \
        --configuration debug \
        --parallel \
        2>&1 | tee "$LOG_DIR/security-tests.log"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Security tests completed in ${duration}s"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "Security tests failed after ${duration}s"
        result=1
    fi
    
    unset AR_SECURITY_TEST_MODE
    return $result
}

# Generate code coverage report
generate_coverage_report() {
    log "Generating code coverage report..."
    
    cd "$PROJECT_ROOT"
    
    # Generate LCOV coverage report
    if xcrun llvm-cov export \
        .build/debug/AutoResolveUIPackageTests.xctest/Contents/MacOS/AutoResolveUIPackageTests \
        --instr-profile .build/debug/codecov/default.profdata \
        --format="lcov" > "$LOG_DIR/coverage.lcov" 2>/dev/null; then
        
        log_success "Coverage report generated: $LOG_DIR/coverage.lcov"
        
        # Generate HTML coverage report if lcov tools are available
        if command -v genhtml &> /dev/null; then
            genhtml "$LOG_DIR/coverage.lcov" \
                --output-directory "$LOG_DIR/coverage-html" \
                --title "AutoResolve Test Coverage" \
                --show-details --legend 2>/dev/null || true
                
            log_success "HTML coverage report: $LOG_DIR/coverage-html/index.html"
        fi
    else
        log_warning "Could not generate coverage report"
    fi
}

# Generate test report
generate_test_report() {
    local total_time=$1
    
    log "Generating test report..."
    
    local report_file="$LOG_DIR/test-report.md"
    
    cat > "$report_file" << EOF
# AutoResolve Test Report

**Generated:** $(date)
**Total Execution Time:** ${total_time}s
**Test Environment:** $(uname -a)

## Test Results Summary

EOF
    
    # Check each test type result
    for test_type in unit integration ui load security; do
        local log_file="$LOG_DIR/${test_type}-tests.log"
        if [[ -f "$log_file" ]]; then
            if grep -q "Test run completed successfully" "$log_file" 2>/dev/null; then
                echo "- ✅ $(cap "$test_type") Tests: **PASSED**" >> "$report_file"
            else
                echo "- ❌ $(cap "$test_type") Tests: **FAILED**" >> "$report_file"
            fi
        else
            echo "- ⏭️ $(cap "$test_type") Tests: **SKIPPED**" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

## Performance Metrics

EOF
    
    # Add system info
    echo "### System Information" >> "$report_file"
    echo '```' >> "$report_file"
    system_profiler SPHardwareDataType 2>/dev/null | head -10 >> "$report_file" || echo "System info not available" >> "$report_file"
    echo '```' >> "$report_file"
    
    # Add memory usage if available
    if [[ -f "$LOG_DIR/load-tests.log" ]]; then
        echo "### Load Test Results" >> "$report_file"
        echo '```' >> "$report_file"
        grep -i "memory\|cpu\|performance" "$LOG_DIR/load-tests.log" | tail -10 >> "$report_file" || true
        echo '```' >> "$report_file"
    fi
    
    # Add coverage info if available
    if [[ -f "$LOG_DIR/coverage.lcov" ]]; then
        echo "### Code Coverage" >> "$report_file"
        local coverage_percent=$(grep -o "lines\.\.\.\.\..*%" "$LOG_DIR/coverage.lcov" | tail -1 || echo "Coverage data not available")
        echo "**Code Coverage:** $coverage_percent" >> "$report_file"
    fi
    
    log_success "Test report generated: $report_file"
}

# Main function
main() {
    local test_type=${1:-all}
    local build_config=${2:-debug}
    
    local script_start=$(date +%s)
    
    log "Starting AutoResolve test suite (type: $test_type, config: $build_config)"
    
    # Setup
    check_prerequisites
    setup_test_environment
    
    # Build project
    if ! build_project "$build_config"; then
        log_error "Build failed. Aborting tests."
        exit 1
    fi
    
    # Run tests based on type
    local test_results=()
    
    case "$test_type" in
        "unit")
            run_unit_tests && test_results+=(0) || test_results+=(1)
            ;;
        "integration")
            run_integration_tests && test_results+=(0) || test_results+=(1)
            ;;
        "ui")
            run_ui_tests && test_results+=(0) || test_results+=(1)
            ;;
        "load")
            run_load_tests && test_results+=(0) || test_results+=(1)
            ;;
        "security")
            run_security_tests && test_results+=(0) || test_results+=(1)
            ;;
        "all")
            run_unit_tests && test_results+=(0) || test_results+=(1)
            run_integration_tests && test_results+=(0) || test_results+=(1)
            run_ui_tests && test_results+=(0) || test_results+=(1)
            run_load_tests && test_results+=(0) || test_results+=(1)
            run_security_tests && test_results+=(0) || test_results+=(1)
            ;;
        *)
            log_error "Unknown test type: $test_type"
            log "Available test types: unit, integration, ui, load, security, all"
            exit 1
            ;;
    esac
    
    # Generate coverage report
    generate_coverage_report
    
    # Calculate total time and generate report
    local script_end=$(date +%s)
    local total_time=$((script_end - script_start))
    
    generate_test_report "$total_time"
    
    # Check results
    local failed_tests=0
    for result in "${test_results[@]}"; do
        ((failed_tests += result))
    done
    
    if [[ $failed_tests -eq 0 ]]; then
        log_success "All tests completed successfully! Total time: ${total_time}s"
        log_success "Test logs available in: $LOG_DIR"
        exit 0
    else
        log_error "$failed_tests test suite(s) failed. Total time: ${total_time}s"
        log_error "Check test logs in: $LOG_DIR"
        exit 1
    fi
}

# Cleanup on exit
cleanup() {
    stop_backend
    log "Cleanup completed"
}

trap cleanup EXIT

# Run main function with all arguments
main "$@"