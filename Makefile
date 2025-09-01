# AutoResolve v7.5 - Production Makefile
# Contract: All gates must pass for ship

.PHONY: all build test perf release clean artifacts verify-gates

# Configuration
SWIFT_FLAGS = -Xswiftc -warnings-as-errors
XCODE_FLAGS = -scheme AutoResolveUI -configuration Release SWIFT_TREAT_WARNINGS_AS_ERRORS=YES

all: build test perf

# Phase 0: Build
build:
	@echo "🔨 Building AutoResolve..."
	swift build $(SWIFT_FLAGS)
	@echo "✅ Build complete"

# Phase 1: Tests
test:
	@echo "🧪 Running tests..."
	swift test $(SWIFT_FLAGS)
	@echo "✅ Tests passed"

# Phase 2: Performance
perf: build
	@echo "📊 Running performance tests..."
	swift test --filter PerfTests
	@./CI/generate_perf_report.sh > Artifacts/perf_report.json
	@echo "✅ Performance gates passed"

# Phase 3: Round-trip
round-trip: build
	@echo "🔄 Testing round-trip..."
	swift test --filter RoundTripTests
	@./CI/generate_round_trip.sh > Artifacts/round_trip_proof.json
	@echo "✅ Round-trip verified"

# Phase 4: Backend health
backend-health:
	@echo "🏥 Checking backend health..."
	@curl -s http://localhost:8000/health > Artifacts/backend_health.json || echo '{"ok":false}' > Artifacts/backend_health.json
	@echo "✅ Backend checked"

# Phase 5: Artifacts
artifacts: perf round-trip backend-health
	@echo "📦 Generating artifacts..."
	@./CI/generate_diff.sh > Artifacts/diff.json
	@touch Artifacts/notarization.log  # Placeholder until notarization
	@echo "✅ Artifacts complete"

# Verification
verify-gates: artifacts
	@echo "🚦 Verifying gates..."
	@swift run VerifyGates Artifacts/
	@echo "✅ All gates passed - ready to ship!"

# Xcode build
xcode:
	xcodebuild build $(XCODE_FLAGS)

# Notarization
notarize: xcode
	@echo "🔏 Notarizing app..."
	xcrun notarytool submit AutoResolveUI.app --wait
	xcrun stapler staple AutoResolveUI.app
	@echo "✅ App notarized"

# Clean
clean:
	swift package clean
	rm -rf .build
	rm -rf Artifacts/*

# Install backend (optional)
install-backend:
	cd autorez && pip install -r requirements.txt

# Run backend (optional)
run-backend:
	cd autorez && python backend_service_final.py

# Development
dev: build
	./.build/debug/AutoResolveUI

# Release build
release: clean xcode notarize artifacts verify-gates
	@echo "🚀 Release build complete!"