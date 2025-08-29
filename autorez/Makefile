# AutoResolve v3.2 - Production Makefile
# Complete operational commands for development, testing, and production

.PHONY: help install test clean run stop health benchmark verify-all setup eval-ablate bench-vjepa bench-clip director-analyze transcribe silence-cut shortsify

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

PY := python3

help: ## Show this help message
	@echo "$(GREEN)AutoResolve v3.2 - Production Commands$(NC)"
	@echo "========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# === SETUP & INSTALLATION ===

install: setup ## Install all dependencies
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

setup: ## Install dependencies (legacy compatibility)
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PY) -m pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest black pylint mypy

clean: clean-cache ## Clean all artifacts and caches
	@echo "$(YELLOW)Cleaning artifacts...$(NC)"
	rm -rf __pycache__ .pytest_cache *.pyc
	rm -rf artifacts/*.json artifacts/cache/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-cache: ## Clean cache only
	rm -rf artifacts/cache/*
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

# === BACKEND OPERATIONS ===

run: ## Start backend server
	@echo "$(GREEN)Starting AutoResolve backend...$(NC)"
	uvicorn backend_service_final:app --host 0.0.0.0 --port 8000 --reload

run-prod: ## Start production server with workers
	uvicorn backend_service_final:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info

run-safe: check-memory ## Run with memory protection
	@echo "Running with memory protection..."
	$(PY) run_safe.py

stop: ## Stop all AutoResolve processes
	@echo "$(YELLOW)Stopping services...$(NC)"
	-pkill -f "uvicorn backend_service_final"
	-lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@echo "$(GREEN)✓ Services stopped$(NC)"

health: ## Check system health
	@echo "$(GREEN)Health Check:$(NC)"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "$(RED)✗ Backend not responding$(NC)"

# === TESTING ===

test-pipeline: ## Test full pipeline with 30min video
	@echo "$(GREEN)Testing full pipeline...$(NC)"
	$(PY) autoresolve_cli.py process assets/test_30min.mp4
	@echo "$(GREEN)✓ Pipeline test complete$(NC)"

test-gates: verify-gates ## Alias for verify-gates

verify-gates: ## Verify performance gates
	@echo "$(GREEN)Verifying performance gates...$(NC)"
	@$(PY) -m src.eval.gates --verify && echo "$(GREEN)✓ All gates PASS$(NC)" || echo "$(RED)✗ Gates FAILED$(NC)"

# === PERFORMANCE ===

benchmark: bench-vjepa bench-clip ## Run all benchmarks
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

bench-vjepa: ## Benchmark V-JEPA embedder
	@echo "$(GREEN)Running V-JEPA benchmark...$(NC)"
	@$(PY) -c "from src.embedders.vjepa_embedder import VJEPAEmbedder; E=VJEPAEmbedder(); segs, meta = E.embed_segments('assets/test_30min.mp4', fps=1.0, window=16, strategy='temp_attn', max_segments=10); print({'segments': len(segs), **meta})"

bench-clip: ## Benchmark CLIP embedder
	@echo "$(GREEN)Running CLIP benchmark...$(NC)"
	@$(PY) -c "from src.embedders.clip_embedder import CLIPEmbedder; E=CLIPEmbedder(); segs, meta = E.embed_segments('assets/test_30min.mp4', fps=1.0, window=16, strategy='temp_attn', max_segments=10); print({'segments': len(segs), **meta})"

check-memory: ## Check system memory
	@echo "Checking system memory..."
	@$(PY) -c "import psutil; m=psutil.virtual_memory(); print(f'Available: {m.available/(1024**3):.1f}GB / {m.total/(1024**3):.1f}GB ({m.percent:.1f}% used)')"

# === OPERATIONS ===

eval-ablate: ## Run V-JEPA vs CLIP ablation
	$(PY) -m src.eval.ablate_vjepa_vs_clip datasets/broll_pilot/manifest.json

director-analyze: ## Analyze video with creative director
	$(PY) -m src.director.creative_director --video $(VIDEO) --out artifacts/creative_director.json

transcribe: ## Transcribe video audio
	$(PY) -m src.ops.transcribe --audio $(VIDEO) --out artifacts/transcript.json

silence-cut: ## Detect and cut silence
	$(PY) -m src.ops.silence $(VIDEO) artifacts/cuts.json

shortsify: ## Create short clips
	$(PY) -m src.ops.shortsify --video $(VIDEO) --out_dir artifacts/shorts

broll-select: ## Select B-roll suggestions
	$(PY) -m src.broll.selector $(VIDEO) --out artifacts/broll_selection.json

proof-pack: ## Generate proof pack
	$(PY) -m src.eval.generate_proof_pack

# === MONITORING ===

logs: ## Tail application logs
	@tail -f logs/autoresolve.log 2>/dev/null || echo "$(YELLOW)No logs available$(NC)"

metrics: metrics-tail ## Show current metrics
	@cat artifacts/metrics.json | python -m json.tool

metrics-tail: ## Tail metrics file
	@tail -f artifacts/metrics.jsonl 2>/dev/null || cat artifacts/metrics.json | python -m json.tool

status: ## Show system status
	@echo "$(GREEN)=== AutoResolve Status ===$(NC)"
	@echo -n "Backend: "; curl -s http://localhost:8000/health >/dev/null 2>&1 && echo "$(GREEN)✓ Running$(NC)" || echo "$(RED)✗ Stopped$(NC)"
	@echo -n "Port 8000: "; lsof -i:8000 >/dev/null 2>&1 && echo "$(GREEN)✓ Listening$(NC)" || echo "$(RED)✗ Free$(NC)"
	@echo -n "Gates: "; $(PY) -m src.eval.gates --verify >/dev/null 2>&1 && echo "$(GREEN)✓ Passing$(NC)" || echo "$(RED)✗ Failing$(NC)"

# === VALIDATION ===

verify-all: ## Complete system verification
	@echo "$(GREEN)=== COMPLETE SYSTEM VERIFICATION ===$(NC)"
	@echo -n "1. Dependencies: "; $(PY) -c "import torch, fastapi, numpy" 2>/dev/null && echo "$(GREEN)✓$(NC)" || echo "$(RED)✗$(NC)"
	@echo -n "2. Structure: "; test -f backend_service_final.py && echo "$(GREEN)✓$(NC)" || echo "$(RED)✗$(NC)"
	@echo -n "3. No LibROSA: "; ! grep -q "import librosa" src/ops/silence.py && echo "$(GREEN)✓$(NC)" || echo "$(RED)✗$(NC)"
	@echo -n "4. Test asset: "; test -f assets/test_30min.mp4 && echo "$(GREEN)✓$(NC)" || echo "$(RED)✗$(NC)"
	@echo -n "5. Gates pass: "; $(PY) -m src.eval.gates --verify >/dev/null 2>&1 && echo "$(GREEN)✓$(NC)" || echo "$(RED)✗$(NC)"
	@echo "$(GREEN)====================================$(NC)"

# === DEPLOYMENT ===

deploy: ## Deploy to production
	@echo "$(GREEN)Deploying to production...$(NC)"
	./deploy_final.sh

# === QUICK COMMANDS ===

quick-test: ## Quick smoke test
	@curl -s http://localhost:8000/health | grep healthy >/dev/null && echo "$(GREEN)✓ System operational$(NC)" || echo "$(RED)✗ System down$(NC)"

all: install verify-all ## Complete setup and verification