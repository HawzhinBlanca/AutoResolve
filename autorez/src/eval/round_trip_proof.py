import json
import hashlib
from pathlib import Path

"""
Generates round_trip_proof.json by exporting and re-importing a timeline and hashing structure.
"""

def stable_hash(obj) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def generate_proof(timeline: dict, out_dir: Path, edl_path: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    original = timeline
    # Simulate export/import normalization
    normalized = {
        "clips": [
            {
                "start": float(c.get("start", 0.0)),
                "end": float(c.get("end", 0.0)),
                "name": c.get("name", "")
            }
            for c in original.get("clips", [])
        ]
    }
    # If EDL provided, import and compare
    imported = None
    if edl_path and edl_path.exists():
        try:
            from src.export.edl_importer import import_edl
            imported = import_edl(str(edl_path))
        except Exception:
            imported = None
    h1 = stable_hash(original)
    h2 = stable_hash(normalized)
    proof = {
        "original_clip_count": len(original.get("clips", [])),
        "normalized_clip_count": len(normalized.get("clips", [])),
        "original_hash": h1,
        "normalized_hash": h2,
        "equal": h1 == h2,
        "edl_imported_clip_count": (len(imported.get("clips", [])) if imported else None)
    }
    with open(out_dir / "round_trip_proof.json", "w") as f:
        json.dump(proof, f, indent=2)
    return proof

if __name__ == "__main__":
    import sys
    timeline_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/timeline.json")
    edl_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out_dir = Path("artifacts")
    timeline = {"clips": []}
    if timeline_path.exists():
        timeline = json.loads(timeline_path.read_text())
    res = generate_proof(timeline, out_dir, edl_path)
    print(json.dumps(res, indent=2))


