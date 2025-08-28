.PHONY: setup eval-ablate bench-vjepa bench-clip director-analyze director-eval-narrative transcribe silence-cut shortsify resolve-build broll-select broll-place proof-pack metrics-tail clean-cache run-safe check-memory

PY=python3

setup:
	$(PY) -m pip install -r requirements.txt

check-memory:
	@echo "Checking system memory..."
	@$(PY) -c "import psutil; m=psutil.virtual_memory(); print(f'Available: {m.available/(1024**3):.1f}GB / {m.total/(1024**3):.1f}GB ({m.percent:.1f}% used)')"

run-safe: check-memory
	@echo "Running with memory protection..."
	$(PY) run_safe.py

# Retrieval A/B
eval-ablate:
	$(PY) -m src.eval.ablate_vjepa_vs_clip datasets/broll_pilot/manifest.json

bench-vjepa:
	@echo "Running V-JEPA benchmark..."
	$(PY) -c "from src.embedders.vjepa_embedder import VJEPAEmbedder; E=VJEPAEmbedder(); segs, meta = E.embed_segments('assets/pilots/clip_5m.mp4', fps=1.0, window=16, strategy='temp_attn'); print({'segments': len(segs), **meta})"

bench-clip:
	@echo "Running CLIP benchmark..."
	$(PY) -c "from src.embedders.clip_embedder import CLIPEmbedder; E=CLIPEmbedder(); segs, meta = E.embed_segments('assets/pilots/clip_5m.mp4', fps=1.0, window=16, strategy='temp_attn'); print({'segments': len(segs), **meta})"

# Director
director-analyze:
	$(PY) -m src.director.creative_director --video $(VIDEO) --out artifacts/creative_director.json

director-eval-narrative:
	$(PY) -m src.eval.eval_director --video $(VIDEO) \
		--gt_incidents datasets/annotations/incidents.jsonl \
		--gt_climax datasets/annotations/climax.jsonl \
		--gt_resolution datasets/annotations/resolution.jsonl

eval-director:
	$(PY) -m src.eval.eval_director datasets/annotations/

# Ops
transcribe:
	$(PY) -m src.ops.transcribe --audio $(VIDEO) --out artifacts/transcript.json

silence-cut:
	$(PY) -m src.ops.silence --video $(VIDEO) --out artifacts/cuts.json

shortsify:
	$(PY) -m src.ops.shortsify --video $(VIDEO) --out_dir artifacts/shorts

resolve-build:
	$(PY) -m src.ops.resolve_api --video $(VIDEO) --cuts artifacts/cuts.json --transcript artifacts/transcript.json --timeline "AutoResolve v3"

# B-roll
broll-select:
	$(PY) -m src.broll.selector --video $(VIDEO) --library datasets/library/stock_manifest.json --out artifacts/broll/select.json

broll-place:
	$(PY) -m src.broll.placer --video $(VIDEO) --select artifacts/broll/select.json --out artifacts/broll/overlay.json

# Proofs & Metrics
proof-pack:
	mkdir -p proof_pack
	$(PY) -m src.eval.ablate_vjepa_vs_clip datasets/broll_pilot/manifest.json > proof_pack/ablation_results.json
	cp -f artifacts/creative_director.json proof_pack/creative_director.json 2>/dev/null || true
	cp -f artifacts/transcript.json       proof_pack/transcript.json       2>/dev/null || true
	cp -f artifacts/cuts.json             proof_pack/cuts.json             2>/dev/null || true
	cp -f artifacts/shorts/index.json     proof_pack/shorts_index.json     2>/dev/null || true
	cp -f artifacts/broll/overlay.json    proof_pack/broll_overlay.json    2>/dev/null || true
	cp -f artifacts/VERSIONS.json         proof_pack/VERSIONS.json         2>/dev/null || true
	$(PY) -c "import json,platform; print(json.dumps({'python': platform.python_version()}))" > proof_pack/environment.json
	$(PY) -m src.eval.generate_proof_pack

metrics-tail:
	@tail -n 50 artifacts/metrics.jsonl 2>/dev/null || echo "no metrics yet"

clean-cache:
	rm -rf artifacts/cache/*.npz artifacts/cache/*.json 2>/dev/null || true

# Bug fix validation commands
validate-fixes:
	bash scripts/validate_all_fixes.sh

test-unit:
	$(PY) -m pytest tests/ -v --tb=short

test-integration:
	$(PY) tests/integration/test_full_pipeline.py

bench-vjepa-new:
	$(PY) scripts/benchmark_vjepa.py --video assets/pilots/clip_5m.mp4 --iterations 3

bench-clip-new:
	$(PY) scripts/benchmark_clip.py --video assets/pilots/clip_5m.mp4 --iterations 3

# OpenRouter Integration
openrouter-setup:
	$(PY) -m pip install openai tiktoken
	@echo "Set OPENROUTER_API_KEY in your environment"

openrouter-test:
	@$(PY) -c "from src.ops.openrouter import OpenRouterClient; import configparser; cfg = configparser.ConfigParser(); cfg.read('conf/ops.ini'); client = OpenRouterClient(cfg); print(client.sanity_check())"

hybrid-bench:
	$(PY) -m src.eval.hybrid_eval --video assets/pilots/scene.mp4