#!/usr/bin/env python3
import sys, os, re, json, configparser, glob, hashlib, subprocess, shlex
from pathlib import Path

REQ_FILES = [
  "Blueprint.md","README.md",
  "autoresolve_cli.py","backend_service_final.py",
  "requirements.txt","Makefile",
  "conf/embeddings.ini","conf/director.ini","conf/ops.ini",
  # core dirs
  "src/embedders/vjepa_embedder.py","src/embedders/clip_embedder.py",
  "src/director/narrative.py","src/director/emotion.py","src/director/rhythm.py",
  "src/director/continuity.py","src/director/emphasis.py","src/director/creative_director.py",
  "src/ops/transcribe.py","src/ops/silence.py","src/ops/shortsify.py",
  "src/ops/resolve_api.py","src/ops/edl.py","src/ops/media.py",
  "src/broll/selector.py","src/broll/placer.py",
  "src/align/align_vjepa_to_clip.py","src/scoring/broll_scoring.py",
  "src/eval/ablate_vjepa_vs_clip.py","src/eval/bootstrap_ci.py","src/eval/eval_director.py",
  "src/utils/memory.py","src/utils/cache.py"
]

# minimal deps we expect for v3.1.1
EXPECTED_DEPS = {
  "torch","transformers","open-clip-torch","pillow","av","psutil","numpy",
  "faster-whisper","ffmpeg-python","fastapi","uvicorn[standard]","pydantic",
  "openai","tiktoken"
}

PATH_HINTS = {
  "openrouter_client_should_be": "src/ops/openrouter.py",
  "openrouter_wrong_place": "src/embedders/openrouter.py",
}

ENDPOINTS = {
  "health": r'GET[^"\']*/health',
  "process": r'POST[^"\']*/api/process',
  "silence": r'POST[^"\']*/api/silence',
  "transcribe": r'POST[^"\']*/api/transcribe',
  "export_edl": r'POST[^"\']*/api/export/edl',
  "export_fcpxml": r'POST[^"\']*/api/export/fcpxml',  # should NOT exist in v3.1.1
}

def read_text(p):
  try:
    return Path(p).read_text(encoding="utf-8", errors="ignore")
  except Exception:
    return ""

def grep(pattern, text):
  return re.search(pattern, text, re.IGNORECASE|re.MULTILINE) is not None

def parse_ini(path):
  ini = configparser.ConfigParser()
  ini.read(path)
  return ini

def main():
  root = Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve()
  report = {"root": str(root), "missing_files": [], "present_files": [], "warnings": [], "errors": [], "info": {}}

  # 1) Protected files/dirs
  for rel in REQ_FILES:
    p = root/"autoresz"/rel if (root/"autoresz").exists() else root/"autorez"/rel  # support both spellings just in case
    if p.exists():
      report["present_files"].append(str(p))
    else:
      report["missing_files"].append(str(p))

  # Clarify base dir (support autorez vs autorez)
  base = root/"autorez"
  if not base.exists():
    base = root/"autoresz"
  report["info"]["base"] = str(base)

  # 2) requirements.txt sanity
  reqp = base/"requirements.txt"
  missing_deps = []
  if reqp.exists():
    req_text = read_text(reqp)
    normalized = set([ln.strip() for ln in req_text.splitlines() if ln.strip() and not ln.strip().startswith("#")])
    # uvicorn installed as uvicorn[standard] OR uvicorn -> accept either
    has_uvicorn = any(x.startswith("uvicorn") for x in normalized)
    for dep in EXPECTED_DEPS:
      if dep == "uvicorn[standard]":
        if not has_uvicorn:
          missing_deps.append("uvicorn or uvicorn[standard]")
      else:
        if not any(d.lower().startswith(dep.lower()) for d in normalized):
          missing_deps.append(dep)
  else:
    report["errors"].append("requirements.txt is missing")

  # 3) config checks
  emb = base/"conf/embeddings.ini"
  dirc = base/"conf/director.ini"
  ops = base/"conf/ops.ini"
  cfg = {}
  if emb.exists():
    e = parse_ini(emb)
    dflt = e["DEFAULT"] if "DEFAULT" in e else {}
    cfg["embeddings.max_rss_gb"] = float(dflt.get("max_rss_gb", "16"))
    cfg["embeddings.default_model"] = dflt.get("default_model","clip")
  if ops.exists():
    o = parse_ini(ops)
    if o.has_section("openrouter"):
      cfg["openrouter.enabled"] = o.getboolean("openrouter","enabled", fallback=False)
    else:
      cfg["openrouter.enabled"] = False

  # 4) backend endpoints & server
  backend = base/"backend_service_final.py"
  endpoints = {"export_fcpxml": False}
  uses_fastapi = False
  if backend.exists():
    text = read_text(backend)
    uses_fastapi = ("fastapi" in text.lower())
    for key, patt in ENDPOINTS.items():
      endpoints[key] = grep(patt, text)
  else:
    report["errors"].append("backend_service_final.py missing")

  # 5) openrouter placement
  wrong_or = base/PATH_HINTS["openrouter_wrong_place"]
  right_or = base/PATH_HINTS["openrouter_client_should_be"]
  if wrong_or.exists():
    report["warnings"].append(f"OpenRouter client found at {wrong_or} (should be {right_or})")
  if right_or.exists():
    report["info"]["openrouter_client"] = str(right_or)

  # 6) vjepa reality & silence implementation hints
  vjepa = base/"src/embedders/vjepa_embedder.py"
  vjepa_text = read_text(vjepa) if vjepa.exists() else ""
  timesformer_fallback = "timesformer" in vjepa_text.lower() or "TimeSformer" in vjepa_text
  true_vjepa_refs = ("vjepa" in vjepa_text.lower())
  silence_py = base/"src/ops/silence.py"
  silence_text = read_text(silence_py) if silence_py.exists() else ""
  uses_librosa = "librosa" in silence_text.lower()
  uses_rms_gate = ("rms" in silence_text.lower()) or ("power" in silence_text.lower())

  # 7) scene detection ghosts (should not exist in v3.1.1)
  scene_files = list((base/"src").rglob("scene*.py"))
  has_cv2 = False
  if reqp.exists():
    has_cv2 = any("opencv" in ln.lower() or "cv2" in ln.lower() for ln in read_text(reqp).splitlines())
  scene_mentioned = "scene" in read_text(base/"Makefile") if (base/"Makefile").exists() else False

  # 8) dataset manifests
  broll_manifest = base/"datasets/broll_pilot/manifest.json"
  stock_manifest = base/"datasets/library/stock_manifest.json"
  report["info"]["has_broll_manifest"] = broll_manifest.exists()
  report["info"]["has_stock_manifest"] = stock_manifest.exists()

  # 9) LOC + swift count (best-effort)
  def run(cmd):
    try:
      out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT, timeout=10).decode("utf-8","ignore")
      return out.strip()
    except Exception as e:
      return f"ERR:{e}"
  loc_py = run(f"bash -lc \"find {shlex.quote(str(base))} -type f -name '*.py' -exec wc -l {{}} + | tail -1\"")
  swift_dir = root/"AutoResolveUI"
  swift_count = run(f"bash -lc \"find {shlex.quote(str(swift_dir))} -name '*.swift' | wc -l\"") if swift_dir.exists() else "0"

  # Compose findings
  report["info"].update({
    "missing_deps": sorted(set(missing_deps)),
    "endpoints": endpoints,
    "uses_fastapi": uses_fastapi,
    "vjepa_timesformer_fallback": timesformer_fallback,
    "vjepa_string_present": true_vjepa_refs,
    "silence_uses_librosa": uses_librosa,
    "silence_has_rms": uses_rms_gate,
    "scene_files_present": [str(p) for p in scene_files],
    "opencv_in_requirements": has_cv2,
    "scene_mentioned_makefile": scene_mentioned,
    "loc_py_tail": loc_py,
    "swift_files": swift_count,
    "config": cfg
  })

  # 10) derive actionable errors/warnings
  # a) endpoints: fcpxml must NOT be present in v3.1.1
  if endpoints.get("export_fcpxml", False):
    report["errors"].append("Found /api/export/fcpxml in backend_service_final.py but v3.1.1 exports EDL only.")

  if not endpoints.get("export_edl", False):
    report["errors"].append("Missing /api/export/edl endpoint.")

  if not (endpoints.get("health",False) and endpoints.get("process",False)):
    report["warnings"].append("Core endpoints (/health and/or /api/process) not detected via regex; verify routes.")

  if not uses_fastapi:
    report["errors"].append("backend_service_final.py does not import/use FastAPI.")

  # b) deps
  if report["info"]["missing_deps"]:
    report["errors"].append(f"requirements.txt is missing: {', '.join(report['info']['missing_deps'])}")

  # c) openrouter wrong placement
  if wrong_or.exists() and not right_or.exists():
    report["errors"].append(f"Move OpenRouter client to {PATH_HINTS['openrouter_client_should_be']} and fix imports.")

  # d) memory budget
  if "embeddings.max_rss_gb" in cfg and cfg["embeddings.max_rss_gb"] < 8:
    report["warnings"].append(f"embeddings.ini:max_rss_gb={cfg['embeddings.max_rss_gb']} is low for V-JEPA; set to 16.")

  # e) silence implementation mismatch
  if uses_librosa and not uses_rms_gate:
    report["warnings"].append("silence.py uses librosa only; add RMS gate and pad/merge guards per v3.1.1.")

  # f) scene detection ghosts
  if scene_files or has_cv2 or scene_mentioned:
    report["warnings"].append("Scene detection code/deps present but not in v3.1.1 scope; remove or disable.")

  # g) manifests
  if not report["info"]["has_broll_manifest"]:
    report["warnings"].append("datasets/broll_pilot/manifest.json missing (A/B eval won’t run).")
  if not report["info"]["has_stock_manifest"]:
    report["warnings"].append("datasets/library/stock_manifest.json missing (B-roll selector won’t run).")

  # h) V-JEPA reality
  if timesformer_fallback:
    report["warnings"].append("vjepa_embedder.py contains TimeSformer fallback; ensure gates demote V-JEPA if slow/weak.")

  # i) OpenRouter default
  if "openrouter.enabled" in cfg and cfg["openrouter.enabled"]:
    report["warnings"].append("OpenRouter is enabled by default; set enabled=false for local-by-default behavior.")

  # 11) Human summary
  summary = []
  if report["errors"]:
    summary.append("❌ BLOCKERS:")
    for e in report["errors"]:
      summary.append(f"  - {e}")
  if report["warnings"]:
    summary.append("⚠️  WARNINGS:")
    for w in report["warnings"]:
      summary.append(f"  - {w}")
  summary.append("ℹ️  INFO:")
  summary.append(f"  - Base: {report['info'].get('base')}")
  summary.append(f"  - Missing protected files: {len(report['missing_files'])}")
  summary.append(f"  - Swift files: {report['info'].get('swift_files')}")
  summary.append(f"  - Py LOC tail: {report['info'].get('loc_py_tail')}")
  summary.append(f"  - Endpoints: {report['info'].get('endpoints')}")
  summary.append(f"  - Config: {report['info'].get('config')}")

  print("\n".join(summary))
  print("\n=== JSON REPORT ===")
  print(json.dumps(report, indent=2))

if __name__ == "__main__":
  main()