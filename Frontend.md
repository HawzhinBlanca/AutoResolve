Checking for changes in `/Users/hawzhin/AutoResolve`:

```bash
find /Users/hawzhin/AutoResolve -type f \( -name "*.swift" -o -name "*.py" \) -mmin -10
ls -lat /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/*.swift | head -5
```

# Complete DaVinci Resolve-Style UI Specification for AutoResolve

## Full Edit Page Layout with ALL AutoResolve Features

### Menu Bar (macOS Native)
- File: New Project, Open, Import (with V-JEPA/CLIP options), Recent Projects
- Edit: Undo/Redo, Cut Detection Settings, Silence Threshold Configuration
- Timeline: Neural Analysis, Auto-Cut Silence, Generate Shorts, B-roll Suggestions
- AI Director: Analyze Story, Detect Emphasis, Find Tension Peaks, Continuity Check
- Embedders: V-JEPA Settings, CLIP Settings, A/B Test Results, Performance Gates
- Export: FCPXML, EDL, Resolve Native, Premiere XML

### Main Window Layout

**Left Panel (380px fixed)**
- Media Pool with tabs: Master | V-JEPA Embeddings | CLIP Results | B-roll Library
- Neural Insights section showing:
  - 469 silence regions detected
  - 119 scene changes with confidence
  - Director's story beat markers
- Effects Library (Resolve standard + AutoResolve AI effects)
- Edit Index with AutoResolve's smart cut suggestions

**Center Area (flexible)**
- Dual Viewer Configuration:
  - Source: Shows V-JEPA/CLIP embedding visualization
  - Timeline: Standard preview with neural overlay option
- Timeline (Resolve exact):
  - V3, V2, V1 tracks
  - Director Track (AutoResolve): Energy curves, tension visualization
  - Transcription Track: Word-level timing from Whisper
  - A1-A8 audio with waveforms
  - Silence regions as semi-transparent overlays
  - Neural cut points as colored markers

**Right Panel (380px fixed)**
Inspector Tabs:
- Video (Resolve standard)
- Audio (Resolve standard)
- **Neural Analysis** (AutoResolve):
  - V-JEPA confidence: 0.89
  - CLIP similarity: 0.76
  - Processing: 51x realtime
  - Memory: 892MB/4GB
- **Director** (AutoResolve):
  - Energy graph realtime
  - Momentum indicators
  - Novelty detection status
  - Emphasis points list
- **Cuts** (AutoResolve):
  - 469 silence cuts available
  - Confidence threshold slider
  - Apply/Preview buttons
- **Shorts** (AutoResolve):
  - Viral moment detection
  - Platform presets (TikTok/YouTube/Instagram)

### Timeline Toolbar
Standard Resolve tools plus:
- Neural Timeline toggle (shows/hides AI overlays)
- Auto-Cut button (applies silence removal)
- Director Analysis button (runs full pipeline)
- Embedder selector (V-JEPA/CLIP/Auto)

### Status Bar
- Current embedder: V-JEPA (local model loaded)
- Backend status: Connected to :8000
- Processing queue: 0 tasks
- Performance: 51x realtime

### Color Specifications
- Base: #282828 (Resolve standard)
- Neural overlays: Blue-purple gradient with 40% opacity
- Confidence indicators: Green (>80%), Yellow (60-80%), Red (<60%)
- V-JEPA markers: Purple
- CLIP markers: Blue

This specification includes every AutoResolve feature integrated into Resolve's professional interface design.