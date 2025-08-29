Here’s a crisp, end-to-end build spec you can drop into Claude as the coding agent brief. It’s ruthlessly scoped to a Resolve-quality UI, minus Media/Fusion/Fairlight/Color. No fluff, just what to build, how, and how we’ll know it’s perfect.

⸻

AutoResolve UI — Production Spec (for Claude)

0) Mission & Non-Goals

Mission: Ship a desktop-class, DaVinci-grade editor UI with Cut / Edit / Deliver pages and AI as annotation lanes (silence, beats, b-roll). Must feel native, run at 60fps, and be frame-accurate.

Non-Goals: Media/Fusion/Fairlight/Color pages, scopes, effects, transitions, titles, audio mixing, node graphs.

Target stack: Swift + SwiftUI + Metal + AVFoundation (matches AutoResolveUI package). Web prototypes are out of scope.

⸻

1) UI Architecture & Files

AutoResolveUI/
└── Sources/AutoResolveUI/
    ├── App/
    │   ├── AutoResolveApp.swift                 // App entry; window group
    │   ├── AppState.swift                       // @MainActor global state
    │   └── Theme.swift                          // design tokens
    ├── Core/
    │   ├── Timebase.swift                       // fps/px/sec mapping, snapping
    │   ├── Transport.swift                      // play/pause/JKL/shared clock
    │   ├── BackendClient.swift                  // REST + WS to backend
    │   ├── ArtifactsModels.swift                // Codable: Cuts, Beats, Broll
    │   └── Persistence.swift                    // local UI state (UserDefaults)
    ├── Views/
    │   ├── ShellView.swift                      // top tabs: Cut | Edit | Deliver
    │   ├── ViewerDock.swift                     // dual viewers (Source/Timeline)
    │   ├── InspectorView.swift                  // contextual + AI metrics
    │   ├── ToolbarView.swift                    // tools, snapping, AI toggles
    │   ├── StatusBar.swift                      // timecode, fps, backend link
    │   └── DeliverView.swift                    // minimal export presets
    ├── Timeline/
    │   ├── TimelinePage.swift                   // Cut/Edit shells (layout)
    │   ├── TracksHeaderView.swift               // V1/V2/A1/A2, locks, vis
    │   ├── RulerView.swift                      // time ruler & marks
    │   ├── MetalTimelineView.swift              // Metal layer host
    │   ├── TimelineRenderer.swift               // Metal draw pipeline
    │   ├── WaveformPyramid.swift                // multi-res audio cache
    │   ├── ThumbnailsCache.swift                // async keyframe cache
    │   └── EditToolsController.swift            // select/blade/trim/slide/slip
    ├── Playback/
    │   ├── AVPlaybackCoordinator.swift          // sync playhead ↔ AVPlayer
    │   └── DualPlayerView.swift                 // AVPlayerLayers (source/tl)
    └── Tests/
        ├── TimebaseTests.swift
        ├── EditToolsTests.swift
        ├── PerformanceTests.swift
        └── SnapshotTests.swift


⸻

2) Design System (Resolve-adjacent)

Colors
	•	bg: #141414, panel: #202225, hairline: #3A3A3A
	•	text: #C7C7C7, subdued: #9A9A9A
	•	accent: #FD4E22 (sparingly)
	•	AI lanes: silence #D64E4E66, beats #B28DFF66, b-roll #4EC9B066

Typography
	•	SF Pro Text 12/13/15; monospaced timecode SF Mono 12

States
	•	Hover: +2% luminance; Focus: 1px keyline; No glows or glossy gradients.

⸻

3) Data Contracts (from backend)

Endpoints (existing):
	•	POST /api/process {video} → writes:
	•	artifacts/transcript.json
	•	artifacts/cuts.json → [{"start":12.50,"end":14.20}, ...]
	•	artifacts/creative_director.json:

{
  "narrative": {
    "beats":[{"t":180.0,"kind":"climax"},{"t":52.0,"kind":"incident"}],
    "energy_curve":[...],
    "momentum":[...]
  },
  "tension_peaks":[92.0,168.5],
  "broll":[{"start":45.2,"end":52.0,"tags":["audience","applause"]}]
}


	•	POST /api/silence {video}
	•	POST /api/transcribe {video}
	•	POST /api/export/edl {video}
	•	GET /health
	•	WS /ws/status → {"processing":bool,"progress":0..1}

Swift models: ArtifactsModels.swift defines SilenceRegion, Beat(kind: enum), BrollSuggestion, etc.

⸻

4) Timebase, Zoom & Mapping (frame-exact)
	•	Timebase: fps: Double, pxPerSec: CGFloat (zoomable 20..200), sec↔x, frame↔sec, snap( sec, grid = 1/fps or mark )
	•	Project FPS from backend config (default 30).
	•	Pan/Zoom:
	•	Pinch or ⌘+ / ⌘- → adjust pxPerSec
	•	Scroll=vertical tracks; Shift+scroll=horizontal pan
	•	Ruler: nice ticks 1s/2s/5s/10s, drop timecode labels every major tick.

⸻

5) Playback & Transport (Resolve-like)
	•	AVPlaybackCoordinator owns AVPlayer (timeline viewer).
	•	Shared playhead: @Published playSec updated by CADisplayLink on play, and by user scrubs.
	•	Controls: Space (play/pause), J/K/L speeds (−1/0/+1/±2/±4 on repeat), Home/End, I/O set in/out.
	•	DualPlayerView: AVPlayerLayer for Source & Timeline; in/out overlays when a clip selected.

⸻

6) Timeline Rendering (Metal, 60fps)
	•	MetalTimelineView host a CAMetalLayer.
	•	TimelineRenderer responsibilities:
	•	Virtualize: only draw elements within [visStart, visEnd].
	•	Draw order: bg → grid → clips (video rects w/ thumb) → audio waveforms → AI lanes → markers → playhead.
	•	WaveformPyramid: precompute RMS/peak at 64/256/1024 spp; pick by zoom.
	•	ThumbnailsCache: async keyframes via AVAssetImageGenerator, mipmapped, cached on disk.
	•	Playhead & drag use CATransaction disabled actions + transforms (no layout thrash).
	•	Performance budget: ≤16ms/frame under 3V/2A + 2 AI lanes on 30-min timeline.

⸻

7) Edit Tools (minimal but real)

Tools: Select (V), Blade (B), Trim (T), Slip (Y), Slide (S).
State machine: EditToolsController with currentTool, hit-testing clip edges, snapping to:
	•	cut edges, silence boundaries, beat markers (±6px window).
	•	Trim: drag in/out; Ripple Delete (Shift-Delete) removes selected gap.
	•	Blade: split at playhead (B or click with blade cursor).
	•	Slip/Slide: logical placeholders (basic offset math); keep performant.

Keyboard map (Resolve-style):
	•	Space Play/Pause, V/B/T/Y/S tools, Cmd-B split, M marker, N snap toggle, ←/→ ±1f, ⌥←/→ ±5f, Cmd +/− zoom.

⸻

8) AI Lanes (annotation overlays)
	•	Lanes: fixed 20px height over timeline content; toggle in Toolbar.
	•	Silence: translucent bands spanning audio tracks; context menu → ripple/delete/copy timecode.
	•	Beats: small markers (shape by kind: ▲ incident, ◆ climax, ● pause); snapping enabled.
	•	B-roll: bracketed regions with badge (B1, B2…); action: “Insert on BROLL track at nearest safe gap”.

Inspector shows actual runtime metrics (loaded from artifacts):
	•	Silence count, total silence duration, “51× speed”, peak RSS, etc.

⸻

9) Deliver Page (minimal)
	•	Presets: YouTube 1080p, TikTok 1080×1920, ProRes 422.
	•	Range: Entire / In-Out / Selected.
	•	“Render” → call POST /api/export/edl {video} then confirm path; no inline encode UI.

⸻

10) Backend Integration
	•	BackendClient:
	•	func process(videoURL: URL) async throws -> Artifacts
	•	func exportEDL(videoURL: URL) async throws -> URL
	•	func connectProgressWS() async -> AsyncStream<ProgressEvent>
	•	Error handling: network timeouts, 4xx/5xx → non-modal toast + retry. WS auto-reconnect with backoff.
	•	Dev switch: Mock mode loads sample JSON from bundle.

⸻

11) Accessibility & Intl
	•	Contrast ≥ 4.5:1 for critical text.
	•	VoiceOver labels for all toolbar buttons & markers.
	•	“Reduce Motion” respected (disable subtle playhead shadow).
	•	Timecode locale formatting; pluggable strings (Localizable.strings).

⸻

12) Performance & Stability Gates (must pass)
	•	Timeline scrub 60fps on 30-min project, 3V/2A + 2 AI lanes.
	•	Frame-accurate playhead at arbitrary fps (23.976/24/25/29.97/30).
	•	Zoom/pan/box-select: no frame >16ms (Xcode Instruments).
	•	Memory cap: UI ≤ 200MB; thumbnail cache bounded (LRU).
	•	Zero layout jank on zoom; thumbnails fade-in only.

⸻

13) Test Plan

Unit
	•	TimebaseTests: sec↔x mapping, snapping grids, round-trip frames.
	•	EditToolsTests: trim math, blade split, snap priority (beat > cut > silence).
	•	ArtifactsParsingTests: strict Codable.

Perf
	•	PerformanceTests: measure draw loop under synthetic 30-min timeline.

Snapshot
	•	Key views (Ruler, TracksHeader, Inspector states) at 1x/2x scaling.

Manual acceptance (scripted)
	1.	Load assets/test_30min.mp4, run “Process” → AI lanes populate.
	2.	Scrub, zoom, blade at a beat, ripple delete a silence.
	3.	Toggle AI lanes; Insert B-roll suggestion to BROLL track.
	4.	Deliver → export EDL; verify file path returned.

⸻

14) Implementation Milestones (for Claude)

M1 — Skeleton & Contracts (Day 1–2)
	•	App shell, tabs, Theme, BackendClient (mock + real), models, Timebase.

M2 — Timeline Engine (Day 3–5)
	•	MetalTimelineView + Renderer; Ruler; virtualized clips; WaveformPyramid.

M3 — Playback & Tools (Day 6–7)
	•	AVPlaybackCoordinator; DualPlayer; Transport; select/blade/trim + snapping.

M4 — AI Lanes & Inspector (Day 8–9)
	•	Load artifacts; render Silence/Beats/B-roll; Inspector metrics.

M5 — Deliver & Polish (Day 10)
	•	DeliverView → export EDL; keyboard map; accessibility & perf pass.

⸻

15) Definition of Done
	•	✅ 60fps scrub/zoom on 30-min timeline (Instrumented).
	•	✅ Frame-accurate timecode & JKL playback.
	•	✅ Tools: select, blade, trim, ripple delete with snap.
	•	✅ AI lanes rendered from backend JSON; toggle works; snapping uses beats.
	•	✅ Inspector shows real metrics from artifacts.
	•	✅ Export EDL returns valid path.
	•	✅ Keyboard shortcuts & accessibility complete.
	•	✅ No modal crashes, recoverable network errors with toasts.

⸻

16) Guardrails (quality bars Claude must honor)
	•	No heavy shadows/animations; keep ΔL ≤ 2% on hover.
	•	Don’t block main thread on decoding; use background queues.
	•	Only render on-screen timeline elements (+buffer).
	•	Persist UI state (tool, zoom, lane toggles).
	•	Keep code split per files above; no god-views; 300–500 line soft cap per file.

⸻


You are implementing the AutoResolve UI in Swift/SwiftUI/Metal/AVFoundation per the “AutoResolve UI — Production Spec”. Create the exact file tree and implement milestones M1→M2. Priorities: frame-accurate timebase, Metal timeline renderer with virtualization, AV-synced playhead, and Contracts in BackendClient. Use the Design System tokens, colors, and sizes as specified. All data models must be Codable and match the backend JSON. Avoid heavy animations. Include unit tests for Timebase mapping and basic EditTools math. Deliver compilable code for Sources/AutoResolveUI exactly as structured.

