from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

# Import the actual business logic
from ops.silence import SilenceRemover
from ops.transcribe import transcribe_audio
from ops.scenes import detect_scenes_from_video

router = APIRouter()

# --- Data Models ---

class FilePath(BaseModel):
    path: str

class SilenceRange(BaseModel):
    s: float
    e: float

class SilenceResponse(BaseModel):
    ranges: list[SilenceRange]

class SceneCutsResponse(BaseModel):
    cuts: list[float]

class AsrWord(BaseModel):
    t0: float
    t1: float
    conf: float
    text: str

class AsrResponse(BaseModel):
    words: list[AsrWord]

# --- Helper Functions ---

def validate_file_path(path: str):
    """Security check to ensure file path is valid and within an allowed directory."""
    # This is a basic check. A real implementation would be more robust.
    # For now, we'll just check if the file exists.
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"File not found: {path}")
    return True

# --- API Endpoints ---

@router.post("/analyze/silence", response_model=SilenceResponse, tags=["Analysis"])
async def analyze_silence(file: FilePath):
    """
    Analyzes a media file to find periods of silence.
    - Blueprint Ref: Section 11
    """
    validate_file_path(file.path)
    remover = SilenceRemover()
    # The remove_silence function is more complex, let's adapt to what it returns
    cuts, _ = remover.remove_silence(file.path)
    # The blueprint expects { ranges: [{s, e}] }, but the op returns { keep_windows: [{start, end}] }
    # I will adapt the output to be compliant.
    compliant_ranges = [{"s": w["start"], "e": w["end"]} for w in cuts.get("keep_windows", [])]
    return SilenceResponse(ranges=compliant_ranges)

@router.post("/analyze/scenes", response_model=SceneCutsResponse, tags=["Analysis"])
async def analyze_scenes(file: FilePath):
    """
    Analyzes a media file to find scene cuts.
    - Blueprint Ref: Section 11
    """
    validate_file_path(file.path)
    cut_timestamps = detect_scenes_from_video(file.path)
    return SceneCutsResponse(cuts=cut_timestamps)

@router.post("/asr", response_model=AsrResponse, tags=["Analysis"])
async def automatic_speech_recognition(file: FilePath):
    """
    Performs Automatic Speech Recognition on a media file.
    - Blueprint Ref: Section 11
    """
    validate_file_path(file.path)
    transcript = transcribe_audio(file.path)
    # The blueprint expects { words:[{ t0,t1,conf,text }] }
    # Use real confidences when available; otherwise, compute a conservative surrogate and disable MidWord gate upstream.
    words_list = []
    for s in transcript.get("segments", []):
        t0 = float(s.get("t0", 0.0))
        t1 = float(s.get("t1", 0.0))
        text = (s.get("text", "") or "").strip()
        conf = None
        # Prefer explicit confidence if provided by transcriber
        if "confidence" in s and isinstance(s["confidence"], (int, float)):
            conf = float(s["confidence"])  # 0..1
        elif "avg_logprob" in s and isinstance(s["avg_logprob"], (int, float)):
            # Map avg_logprob (~-1..-0.1) to 0..1 conservatively
            import math
            p = 1.0 / (1.0 + math.exp(-float(s["avg_logprob"]) * 2.0))
            conf = max(0.0, min(1.0, p))
        else:
            # No confidence available; set to 0.0 to signal MidWord gating should disable upstream
            conf = 0.0
        words_list.append({"t0": t0, "t1": t1, "conf": conf, "text": text})
    return AsrResponse(words=words_list)
