## PHASE 1 - COMPLETE FILESYSTEM SCAN
- **files_total**: 4821
- **deep_directories(>5)**: 1138
- **empty_directories**: 16
- **wrong_loc_media**: 3
- **nonstandard_names**: 72
- **orphaned_python_files**: 47

### File classification
SOURCE_CODE: {
  ".py": {
    "count": 62,
    "total_lines": 15273,
    "total_size": "528.0KB"
  },
  ".sh": {
    "count": 10,
    "total_lines": 823,
    "total_size": "25.8KB"
  },
  ".swift": {
    "count": 40,
    "total_lines": 25232,
    "total_size": "835.3KB"
  }
}
CONFIGS: {
  "types": [
    ".json",
    ".yaml",
    ".yml",
    ".ini"
  ],
  "count": 116,
  "secrets_exposed": false
}
ASSETS: {
  "types": [
    ".mp4",
    ".npz",
    ".wav"
  ],
  "count": 65,
  "optimized": false
}
DOCUMENTATION: {
  "count": 15,
  "last_updated": [
    {
      "path": "/Users/hawzhin/AutoResolve/Frontend.md",
      "last_updated": "2025-08-21"
    },
    {
      "path": "/Users/hawzhin/AutoResolve/Blueprint.md",
      "last_updated": "2025-08-21"
    },
    {
      "path": "/Users/hawzhin/AutoResolve/autorez/DAY10_BRUTAL_AUDIT.md",
      "last_updated": "2025-08-22"
    },
    {
      "path": "/Users/hawzhin/AutoResolve/autorez/DAY6_COMPLETION.md",
      "last_updated": "2025-08-22"
    },
    {
      "path": "/Users/hawzhin/AutoResolve/autorez/DAY9_PERFORMANCE_BENCHMARK.md",
      "last_updated": "2025-08-22"
    }
  ]
}
JUNK: {
  "count": 561
}

## PHASE 2 - CODE INTELLIGENCE (TOP 30 HEAVIEST)
FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/NeuralTimelineView.swift
PURPOSE: // AUTORESOLVE V3.0 - MACOS SEQUOIA NATIVE
IMPORTS: 8 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/MediaBrowser.swift
PURPOSE: // MARK: - Professional Media Browser
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/CompleteProfessionalTimeline.swift
PURPOSE: // AUTORESOLVE V3.2 - COMPLETE PROFESSIONAL EDITION
IMPORTS: 9 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: Medium ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Telemetry/BackendTelemetryDashboard.swift
PURPOSE: // MARK: - Backend Telemetry Dashboard
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/autorez/src/embedders/vjepa_embedder.py
PURPOSE: Enterprise-grade V-JEPA-2 Video Embedder
IMPORTS: 31 total (3 unused - ['dataclass', 'Any', 'load_file'])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: Yes (orphan/dup)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/ExportPanel.swift
PURPOSE: // MARK: - Professional Export Panel
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/ColorGradingPanel.swift
PURPOSE: // MARK: - Professional Color Grading Panel
IMPORTS: 2 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/EffectsInspector.swift
PURPOSE: // MARK: - Effects Inspector
IMPORTS: 2 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Core/MenuBarCommands.swift
PURPOSE: // AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
IMPORTS: 2 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: Medium ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/KeyframeEditor.swift
PURPOSE: // MARK: - Professional Keyframe Editor
IMPORTS: 2 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/ProjectInspector.swift
PURPOSE: // MARK: - Project Settings Inspector
IMPORTS: 2 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/ClipInspector.swift
PURPOSE: // MARK: - Clip Properties Inspector
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: Medium ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/MinimalWorkingApp.swift
PURPOSE: // AUTORESOLVE NEURAL TIMELINE - MINIMAL WORKING VERSION
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Backend/AutoResolveService.swift
PURPOSE: // MARK: - AutoResolve Backend Service
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/BRoll/BRollSelectionView.swift
PURPOSE: // MARK: - B-Roll Selection Main View
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: Medium ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Export/ResolveExportBridge.swift
PURPOSE: // MARK: - DaVinci Resolve Export Bridge
IMPORTS: 4 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/UnifiedStoreWithBackend.swift
PURPOSE: // AUTORESOLVE V3.0 - UNIFIED STORE
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/BRoll/BRollSelectionViewModel.swift
PURPOSE: // MARK: - B-Roll Selection View Model
IMPORTS: 5 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Silence/SilenceDetectionView.swift
PURPOSE: // MARK: - Silence Detection Main View
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: Medium ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Backend/PipelineIntegration.swift
PURPOSE: // MARK: - Pipeline Integration Manager
IMPORTS: 4 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/artifacts/autopsy_scan.py
PURPOSE: n/a
IMPORTS: 11 total (3 unused - ['sys', 'Counter', 'timedelta'])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: Yes (orphan/dup)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Backend/TimelineBackendBridge.swift
PURPOSE: // MARK: - Timeline Backend Bridge
IMPORTS: 4 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/autorez/DAY9_E2E_TEST.swift
PURPOSE: // DAY 9: END-TO-END TESTING WITH 43-MINUTE VIDEO
IMPORTS: 2 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: Medium ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Components/StatusViews/PipelineStatusView.swift
PURPOSE: // Status Header Bar
IMPORTS: 1 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: Medium ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Core/KeyboardShortcuts.swift
PURPOSE: // AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/main.swift
PURPOSE: // AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: Low ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Core/ProjectPersistence.swift
PURPOSE: // AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
IMPORTS: 4 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Core/UndoRedoSystem.swift
PURPOSE: // AUTORESOLVE NEURAL TIMELINE - APPLE DESIGN AWARD ARCHITECTURE
IMPORTS: 3 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/VideoPlayer/VideoPlayerView.swift
PURPOSE: // MARK: - Professional Video Player View
IMPORTS: 4 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: Medium ()
DELETE_CANDIDATE: No (kept)

FILE: /Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Silence/SilenceDetectionViewModel.swift
PURPOSE: // MARK: - Silence Detection View Model
IMPORTS: 5 total (0 unused - [])
CRITICAL_FUNCTIONS: []
DUPLICATED_IN: {"exact": [], "near": []}
TECH_DEBT_SCORE: High ()
DELETE_CANDIDATE: No (kept)

## PHASE 3 - DUPLICATE & REDUNDANCY
DUPLICATE_SET_1:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/workspace-state.json", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/SourcePackages/workspace-state.json"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/workspace-state.json
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/SourcePackages/workspace-state.json"]
REFACTOR_TO: shared utils

DUPLICATE_SET_2:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/debug.yaml", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/plugin-tools.yaml"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/debug.yaml
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/plugin-tools.yaml"]
REFACTOR_TO: shared utils

DUPLICATE_SET_3:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/swift-version--58304C5D6DBC2206.txt", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/swift-version--58304C5D6DBC2206.txt"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/swift-version--58304C5D6DBC2206.txt
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/swift-version--58304C5D6DBC2206.txt"]
REFACTOR_TO: shared utils

DUPLICATE_SET_4:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/plugin-tools-description.json", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/description.json"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/plugin-tools-description.json
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/description.json"]
REFACTOR_TO: shared utils

DUPLICATE_SET_5:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/AutoResolveUI.dSYM/Contents/Info.plist", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/AutoResolveUI.dSYM/Contents/Info.plist"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/AutoResolveUI.dSYM/Contents/Info.plist
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/AutoResolveUI.dSYM/Contents/Info.plist"]
REFACTOR_TO: shared utils

DUPLICATE_SET_6:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/Symbols-3F8WHRLXCHS2Z.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/Symbols-3F8WHRLXCHS2Z.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/Symbols-3F8WHRLXCHS2Z.swiftmodule"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/Symbols-3F8WHRLXCHS2Z.swiftmodule
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/Symbols-3F8WHRLXCHS2Z.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/Symbols-3F8WHRLXCHS2Z.swiftmodule"]
REFACTOR_TO: shared utils

DUPLICATE_SET_7:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/System-OA9BJKH5PQG.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/System-OA9BJKH5PQG.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/System-OA9BJKH5PQG.swiftmodule"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/System-OA9BJKH5PQG.swiftmodule
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/System-OA9BJKH5PQG.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/System-OA9BJKH5PQG.swiftmodule"]
REFACTOR_TO: shared utils

DUPLICATE_SET_8:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/CoreFoundation-3PPT6YJD3TNVM.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/CoreFoundation-3PPT6YJD3TNVM.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/CoreFoundation-3PPT6YJD3TNVM.swiftmodule"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/CoreFoundation-3PPT6YJD3TNVM.swiftmodule
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/CoreFoundation-3PPT6YJD3TNVM.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/CoreFoundation-3PPT6YJD3TNVM.swiftmodule"]
REFACTOR_TO: shared utils

DUPLICATE_SET_9:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/_Builtin_float-74R7B61QVIAW.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/_Builtin_float-74R7B61QVIAW.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/_Builtin_float-74R7B61QVIAW.swiftmodule"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/_Builtin_float-74R7B61QVIAW.swiftmodule
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/_Builtin_float-74R7B61QVIAW.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/_Builtin_float-74R7B61QVIAW.swiftmodule"]
REFACTOR_TO: shared utils

DUPLICATE_SET_10:
FILES: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/CoreData-C47JKA3DJOHL.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/CoreData-C47JKA3DJOHL.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/CoreData-C47JKA3DJOHL.swiftmodule"]
SIMILARITY: 100%
KEEP: /Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/CoreData-C47JKA3DJOHL.swiftmodule
DELETE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/CoreData-C47JKA3DJOHL.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/CoreData-C47JKA3DJOHL.swiftmodule"]
REFACTOR_TO: shared utils

NEAR_DUPLICATE_1:
FILES: ["/Users/hawzhin/AutoResolve/autorez/scripts/benchmark_clip.py", "/Users/hawzhin/AutoResolve/autorez/scripts/benchmark_vjepa.py"]
SIMILARITY: 95.15%

## PHASE 4 - ARCHITECTURE REALITY CHECK
CLAIMED: [{"doc": "Blueprint.md", "size": 15668}, {"doc": "Frontend.md", "size": 34871}]
REALITY: Python backend under autoresz/src with modules; Swift UI in AutoResolveUI.
LIES: None auto-detected; manual review required.
CIRCULAR_DEPENDENCIES: []
LAYERING_VIOLATIONS: []
FRAMEWORK_MISUSE: Heuristic-only; no blatant reimplementation detected in scan.
## PHASE 5 - BRUTAL HONESTY
🔴 CRITICAL ISSUES: [{"issue": "Print statements in production code", "files": 32, "impact": "noise/perf", "effort": "low"}]
🟡 ARCHITECTURAL DEBT: {"deep_dirs": 1138, "unused_imports": 71, "redundant_wrappers": 10}
⚠️ MAINTENANCE NIGHTMARES: {"dead_symbols": 48, "nonstandard_names": 72}
💀 DEAD WEIGHT: {"dead_code_symbols": 48, "redundant_files": 392, "wasted_bytes": 58238650}
## PHASE 6 - HIDDEN PROBLEMS
SECRETS: []
OLD_TODOs: []
PROD_PRINTS: [{"path": "/Users/hawzhin/AutoResolve/launch_app.py", "count": 23}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/backend_service.py", "count": 4}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/CompleteProfessionalTimeline.swift", "count": 2}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/UnifiedStoreWithBackend.swift", "count": 1}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/MinimalWorkingApp.swift", "count": 3}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Core/VideoProjectStore.swift", "count": 1}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Silence/SilenceDetectionViewModel.swift", "count": 6}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/MediaBrowser.swift", "count": 1}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/VideoPlayer/VideoPlayerView.swift", "count": 1}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/VideoPlayer/AudioWaveformGenerator.swift", "count": 1}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/BRoll/BRollSelectionViewModel.swift", "count": 2}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/BRoll/BRollSelectionView.swift", "count": 1}, {"path": "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Export/ResolveExportBridge.swift", "count": 1}, {"path": "/Users/hawzhin/AutoResolve/artifacts/autopsy_scan.py", "count": 1}, {"path": "/Users/hawzhin/AutoResolve/autorez/run_app.py", "count": 61}, {"path": "/Users/hawzhin/AutoResolve/autorez/DAY9_E2E_TEST.swift", "count": 48}, {"path": "/Users/hawzhin/AutoResolve/autorez/app.py", "count": 92}, {"path": "/Users/hawzhin/AutoResolve/autorez/scripts/benchmark_clip.py", "count": 16}, {"path": "/Users/hawzhin/AutoResolve/autorez/scripts/benchmark_vjepa.py", "count": 16}, {"path": "/Users/hawzhin/AutoResolve/autorez/src/director/creative_director.py", "count": 2}]
ERROR_SWALLOWS: []
GOD_FILES: heuristic-only via smells; see PHASE 2 list
## PHASE 7 - INTELLIGENT RECOMMENDATIONS
1. DELETE_NOW: ["/Users/hawzhin/AutoResolve/launch_app.py", "/Users/hawzhin/AutoResolve/AutoResolveUI/backend_service.py", "/Users/hawzhin/AutoResolve/artifacts/autopsy_scan.py", "/Users/hawzhin/AutoResolve/autorez/run_app.py", "/Users/hawzhin/AutoResolve/autorez/app.py", "/Users/hawzhin/AutoResolve/autorez/scripts/benchmark_clip.py", "/Users/hawzhin/AutoResolve/autorez/scripts/benchmark_vjepa.py", "/Users/hawzhin/AutoResolve/autorez/src/director/creative_director.py", "/Users/hawzhin/AutoResolve/autorez/src/director/narrative.py", "/Users/hawzhin/AutoResolve/autorez/src/director/emotion.py", "/Users/hawzhin/AutoResolve/autorez/src/director/emphasis.py", "/Users/hawzhin/AutoResolve/autorez/src/director/continuity.py", "/Users/hawzhin/AutoResolve/autorez/src/director/rhythm.py", "/Users/hawzhin/AutoResolve/autorez/src/config/schema_validator.py", "/Users/hawzhin/AutoResolve/autorez/src/utils/memory_manager.py", "/Users/hawzhin/AutoResolve/autorez/src/utils/memory.py", "/Users/hawzhin/AutoResolve/autorez/src/utils/promotion.py", "/Users/hawzhin/AutoResolve/autorez/src/utils/cache.py", "/Users/hawzhin/AutoResolve/autorez/src/utils/telemetry.py", "/Users/hawzhin/AutoResolve/autorez/src/utils/common.py"]
2. MERGE: [{"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/workspace-state.json", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/SourcePackages/workspace-state.json"]}, {"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/debug.yaml", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/plugin-tools.yaml"]}, {"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/swift-version--58304C5D6DBC2206.txt", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/swift-version--58304C5D6DBC2206.txt"]}, {"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/plugin-tools-description.json", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/description.json"]}, {"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/AutoResolveUI.dSYM/Contents/Info.plist", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/AutoResolveUI.dSYM/Contents/Info.plist"]}, {"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/Symbols-3F8WHRLXCHS2Z.swiftmodule", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/Symbols-3F8WHRLXCHS2Z.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/Symbols-3F8WHRLXCHS2Z.swiftmodule"]}, {"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/System-OA9BJKH5PQG.swiftmodule", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/System-OA9BJKH5PQG.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/System-OA9BJKH5PQG.swiftmodule"]}, {"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/CoreFoundation-3PPT6YJD3TNVM.swiftmodule", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/CoreFoundation-3PPT6YJD3TNVM.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/CoreFoundation-3PPT6YJD3TNVM.swiftmodule"]}, {"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/_Builtin_float-74R7B61QVIAW.swiftmodule", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/_Builtin_float-74R7B61QVIAW.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/_Builtin_float-74R7B61QVIAW.swiftmodule"]}, {"keep": "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/CoreData-C47JKA3DJOHL.swiftmodule", "delete": ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/ModuleCache/CoreData-C47JKA3DJOHL.swiftmodule", "/Users/hawzhin/AutoResolve/AutoResolveUI/DerivedData/ModuleCache.noindex/CoreData-C47JKA3DJOHL.swiftmodule"]}]
3. REFACTOR_CRITICAL: ["/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/NeuralTimelineView.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/UnifiedStoreWithBackend.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/MinimalWorkingApp.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Core/KeyboardShortcuts.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Core/UndoRedoSystem.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Core/ProjectPersistence.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Core/VideoProject.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Silence/SilenceDetectionViewModel.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/KeyframeEditor.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/EffectsInspector.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/ProjectInspector.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/MediaBrowser.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/ColorGradingPanel.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/Inspector/ExportPanel.swift", "/Users/hawzhin/AutoResolve/AutoResolveUI/Sources/AutoResolveUI/VideoPlayer/AudioWaveformGenerator.swift"]
4. RESTRUCTURE: ["/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/AutoResolveUI.dSYM/Contents", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/AutoResolveUI.dSYM/Contents/Resources", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/AutoResolveUI.dSYM/Contents/Resources/Relocations", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/AutoResolveUI.dSYM/Contents/Resources/Relocations/aarch64", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/AutoResolveUI.dSYM/Contents/Resources/DWARF", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/release/ModuleCache/36ALDCEWRTHZG", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/AutoResolveUI.dSYM/Contents", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/AutoResolveUI.dSYM/Contents/Resources", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/AutoResolveUI.dSYM/Contents/Resources/Relocations", "/Users/hawzhin/AutoResolve/AutoResolveUI/.build/arm64-apple-macosx/debug/AutoResolveUI.dSYM/Contents/Resources/Relocations/aarch64"]
5. EXTRACT: [{"files": ["/Users/hawzhin/AutoResolve/autorez/scripts/benchmark_clip.py", "/Users/hawzhin/AutoResolve/autorez/scripts/benchmark_vjepa.py"], "similarity": 95.15}]
6. MODERNIZE: ["Replace print with structured logging; enforce pre-commit"]
## PHASE 8 - TRUE PROJECT STATE
ACTUAL_PURPOSE: Automated video analysis and resolve pipeline with Swift UI front-end.
CLAIMED_PURPOSE: See Blueprint.md and RESOLVE_INTEGRATION.md; appears aligned.
PRODUCTION_READY: No (blockers: Remove print statements; add logging policy)
TECHNICAL_DEBT_HOURS: ~1024
BUS_FACTOR: 2-3 (inferred by module coupling and docs)
MAINTAINABILITY_SCORE: 76/100
TIME_TO_ONBOARD_DEV: 3-5 days
TRUE_CODEBASE_HEALTH: B-