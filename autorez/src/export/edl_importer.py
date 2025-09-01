import re

# CMX3600 EDL event line with timecodes (non-drop or drop)
EDL_EVENT = re.compile(
    r"^\s*(?P<event>\d+)\s+AX\s+V\s+C\s+"
    r"(?P<src_in>\d{2}:\d{2}:\d{2}:\d{2})\s+"
    r"(?P<src_out>\d{2}:\d{2}:\d{2}:\d{2})\s+"
    r"(?P<rec_in>\d{2}:\d{2}:\d{2}:\d{2})\s+"
    r"(?P<rec_out>\d{2}:\d{2}:\d{2}:\d{2})\s*$"
)

def _tc_to_seconds(tc: str, fps: int = 30) -> float:
    hh, mm, ss, ff = [int(x) for x in tc.split(":")]
    return float(hh * 3600 + mm * 60 + ss) + float(ff) / float(max(1, fps))

def import_edl(edl_path: str, fps: int = 30) -> dict:
    """Parse CMX3600 EDL and produce clips with real start/end (record timeline seconds).
    Raises on unreadable files; returns empty clips when no events found.
    """
    clips = []
    with open(edl_path, "r") as f:
        for line in f:
            m = EDL_EVENT.match(line)
            if not m:
                continue
            rec_in = _tc_to_seconds(m.group("rec_in"), fps=fps)
            rec_out = _tc_to_seconds(m.group("rec_out"), fps=fps)
            if rec_out > rec_in:
                clips.append({"start": rec_in, "end": rec_out})
    return {"clips": clips}


