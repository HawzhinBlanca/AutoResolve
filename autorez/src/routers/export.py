from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import tempfile
from typing import List, Dict, Any

# Import the real EDL export logic
from ops.edl import generate_edl, generate_fcxml

router = APIRouter()

# --- Data Models ---

class TimelineClip(BaseModel):
    t0: float  # Start time in seconds
    t1: float  # End time in seconds
    name: str = "clip"

class Timeline(BaseModel):
    clips: List[TimelineClip] = []
    video_path: str = ""
    fps: int = 30
    format: str = "edl"  # "edl" or "fcxml"

class ExportResponse(BaseModel):
    edl_path: str

# --- API Endpoints ---

@router.post("/export/edl", response_model=ExportResponse, tags=["Export"])
async def export_edl(timeline: Timeline):
    """
    Exports a timeline to an EDL file.
    - Blueprint Ref: Section 11
    """
    try:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.edl', delete=False) as f:
            output_path = f.name
        
        # Convert timeline data to cuts format expected by EDL generator
        cuts_data = {
            "keep": [
                {"t0": clip.t0, "t1": clip.t1}
                for clip in timeline.clips
            ]
        }
        
        # Use real EDL export logic
        if timeline.format.lower() == "fcxml":
            result = generate_fcxml(
                video_path=timeline.video_path or "source.mp4",
                cuts=cuts_data,
                output_path=output_path.replace('.edl', '.fcpxml'),
                fps=timeline.fps
            )
            return ExportResponse(edl_path=result["path"])
        else:
            # Generate EDL format
            result = generate_edl(
                cuts=[cuts_data],
                fps=timeline.fps,
                video_path=timeline.video_path or "source.mp4",
                output_path=output_path
            )
            
            if isinstance(result, dict) and result.get("success"):
                return ExportResponse(edl_path=result["path"])
            else:
                return ExportResponse(edl_path=output_path)
                
    except Exception as e:
        # Fallback to basic EDL if real implementation fails
        with tempfile.NamedTemporaryFile(mode='w', suffix='.edl', delete=False) as f:
            f.write("TITLE: AutoResolve Edit\n\n")
            f.write("FCM: NON-DROP FRAME\n\n")
            
            for i, clip in enumerate(timeline.clips):
                f.write(f"{i+1:03d}  SOURCE   V     C        {clip.t0:08.2f} {clip.t1:08.2f} 00:00:00:00 00:00:10:00\n")
            
            return ExportResponse(edl_path=f.name)
