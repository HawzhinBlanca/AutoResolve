"""
JSON output schemas for AutoResolve.
Defines standard formats for all JSON outputs.
"""

# Transcript output schema
TRANSCRIPT_SCHEMA = {
    "type": "object",
    "required": ["language", "segments", "rtf", "model"],
    "properties": {
        "language": {"type": "string", "description": "Detected language code"},
        "segments": {
            "type": "array",
            "description": "Transcribed segments with timing",
            "items": {
                "type": "object",
                "required": ["t0", "t1", "text"],
                "properties": {
                    "t0": {"type": "number", "description": "Start time in seconds"},
                    "t1": {"type": "number", "description": "End time in seconds"},
                    "text": {"type": "string", "description": "Transcribed text"}
                }
            }
        },
        "rtf": {"type": "number", "description": "Real-time factor (processing time / audio time)"},
        "model": {"type": "string", "description": "Model used for transcription"}
    }
}

# Cuts output schema
CUTS_SCHEMA = {
    "type": "object",
    "required": ["keep_windows", "params", "stats"],
    "properties": {
        "keep_windows": {
            "type": "array",
            "description": "Time windows to keep after silence removal",
            "items": {
                "type": "object",
                "required": ["t0", "t1"],
                "properties": {
                    "t0": {"type": "number", "description": "Start time in seconds"},
                    "t1": {"type": "number", "description": "End time in seconds"}
                }
            }
        },
        "params": {
            "type": "object",
            "description": "Parameters used for silence detection",
            "properties": {
                "rms_thresh_db": {"type": "number"},
                "min_silence_s": {"type": "number"},
                "min_keep_s": {"type": "number"},
                "pad_s": {"type": "number"}
            }
        },
        "stats": {
            "type": "object",
            "description": "Statistics about the cuts",
            "properties": {
                "original_duration": {"type": "number"},
                "kept_duration": {"type": "number"},
                "removed_duration": {"type": "number"},
                "compression_ratio": {"type": "number"}
            }
        }
    }
}

# Shorts index schema
SHORTS_INDEX_SCHEMA = {
    "type": "object",
    "required": ["shorts", "params", "source"],
    "properties": {
        "shorts": {
            "type": "array",
            "description": "Generated short clips",
            "items": {
                "type": "object",
                "required": ["id", "t0", "t1", "score", "output_file"],
                "properties": {
                    "id": {"type": "string", "description": "Short clip ID"},
                    "t0": {"type": "number", "description": "Start time in source"},
                    "t1": {"type": "number", "description": "End time in source"},
                    "score": {"type": "number", "description": "Quality score"},
                    "output_file": {"type": "string", "description": "Output filename"},
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata",
                        "properties": {
                            "content_score": {"type": "number"},
                            "narrative_score": {"type": "number"},
                            "tension_score": {"type": "number"}
                        }
                    }
                }
            }
        },
        "params": {
            "type": "object",
            "description": "Parameters used for generation",
            "properties": {
                "target": {"type": "integer"},
                "min_seg": {"type": "number"},
                "max_seg": {"type": "number"},
                "topk": {"type": "integer"}
            }
        },
        "source": {
            "type": "object",
            "description": "Source video information",
            "properties": {
                "path": {"type": "string"},
                "duration": {"type": "number"}
            }
        }
    }
}

# B-roll selection schema
BROLL_SELECT_SCHEMA = {
    "type": "object",
    "required": ["candidates", "query", "params"],
    "properties": {
        "candidates": {
            "type": "array",
            "description": "Ranked B-roll candidates",
            "items": {
                "type": "object",
                "required": ["id", "score", "path"],
                "properties": {
                    "id": {"type": "string", "description": "B-roll clip ID"},
                    "score": {"type": "number", "description": "Relevance score"},
                    "path": {"type": "string", "description": "Path to B-roll file"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Associated tags"
                    }
                }
            }
        },
        "query": {
            "type": "object",
            "description": "Query information",
            "properties": {
                "text": {"type": "string"},
                "source_segment": {
                    "type": "object",
                    "properties": {
                        "t0": {"type": "number"},
                        "t1": {"type": "number"}
                    }
                }
            }
        },
        "params": {
            "type": "object",
            "description": "Selection parameters",
            "properties": {
                "model": {"type": "string"},
                "strategy": {"type": "string"}
            }
        }
    }
}

# B-roll overlay schema
BROLL_OVERLAY_SCHEMA = {
    "type": "object",
    "required": ["overlays", "timeline", "params"],
    "properties": {
        "overlays": {
            "type": "array",
            "description": "B-roll overlay placements",
            "items": {
                "type": "object",
                "required": ["broll_id", "timeline_t0", "timeline_t1", "broll_t0", "broll_t1"],
                "properties": {
                    "broll_id": {"type": "string", "description": "B-roll clip ID"},
                    "timeline_t0": {"type": "number", "description": "Start time in timeline"},
                    "timeline_t1": {"type": "number", "description": "End time in timeline"},
                    "broll_t0": {"type": "number", "description": "Start time in B-roll clip"},
                    "broll_t1": {"type": "number", "description": "End time in B-roll clip"},
                    "transition_in": {"type": "string", "description": "Transition type"},
                    "transition_out": {"type": "string", "description": "Transition type"}
                }
            }
        },
        "timeline": {
            "type": "object",
            "description": "Timeline information",
            "properties": {
                "duration": {"type": "number"},
                "fps": {"type": "number"}
            }
        },
        "params": {
            "type": "object",
            "description": "Overlay parameters",
            "properties": {
                "max_overlay_s": {"type": "number"},
                "min_gap_s": {"type": "number"},
                "dissolve_s": {"type": "number"}
            }
        }
    }
}

# Creative director analysis schema
CREATIVE_DIRECTOR_SCHEMA = {
    "type": "object",
    "required": ["narrative", "emotion", "rhythm", "continuity", "emphasis"],
    "properties": {
        "narrative": {
            "type": "object",
            "description": "Narrative analysis",
            "properties": {
                "incidents": {"type": "array", "items": {"type": "object"}},
                "climax": {"type": "array", "items": {"type": "object"}},
                "resolution": {"type": "array", "items": {"type": "object"}},
                "energy_curve": {"type": "array", "items": {"type": "number"}}
            }
        },
        "emotion": {
            "type": "object",
            "description": "Emotion/tension analysis",
            "properties": {
                "tension_curve": {"type": "array", "items": {"type": "number"}},
                "peaks": {"type": "array", "items": {"type": "object"}}
            }
        },
        "rhythm": {
            "type": "object",
            "description": "Rhythm and pacing",
            "properties": {
                "cut_points": {"type": "array", "items": {"type": "number"}},
                "pace": {"type": "string", "enum": ["slow", "medium", "fast"]}
            }
        },
        "continuity": {
            "type": "object",
            "description": "Shot continuity analysis",
            "properties": {
                "scores": {"type": "array", "items": {"type": "number"}}
            }
        },
        "emphasis": {
            "type": "object",
            "description": "Emphasis points",
            "properties": {
                "beats": {"type": "array", "items": {"type": "object"}}
            }
        }
    }
}