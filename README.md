# AutoResolve - Minimal Video Editor

## Quick Start

### 1. Start Backend
```bash
cd backend_minimal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 2. Build & Run UI
```bash
cd AutoResolveUI
swift build
.build/debug/MainApp
```

## Project Structure
```
AutoResolve/
├── AIDirector/        # AI-powered timeline director (Swift)
├── AutoResolveUI/     # macOS UI (SwiftUI)
├── backend_minimal/   # Minimal FastAPI backend
└── README.md
```

## Features
- Video timeline editing
- Silence detection
- Basic transcription
- EDL export

## Requirements
- macOS 14+
- Xcode 15+
- Python 3.10+
- Swift 5.9+

## Endpoints
- `GET /health` - Health check
- `POST /api/process` - Process video
- `POST /api/silence` - Detect silence
- `POST /api/transcribe` - Transcribe audio
- `POST /api/export/edl` - Export EDL
- `GET /api/export/edl/{task_id}` - Download EDL
- `GET /api/export/fcpxml/{task_id}` - Download FCPXML
- `WS /ws/status` - Progress updates

## Development

- WebSocket dev mode
  - To skip auth/origin checks locally for rapid UI iteration, set `DEV_WS_NO_AUTH=true` in your environment before starting the backend. This is ignored in production.
  - Example:
    ```bash
    DEV_WS_NO_AUTH=true uvicorn autorez/backend_service_final:app --port 8000
    ```

## Cleanup Summary
Reduced from 8.1GB/54k files to 2.7GB/248 files by removing:
- Overengineered backend (autorez/)
- Python environments (venv/)
- Test artifacts
- Documentation bloat
- Build caches

Now contains only shippable code.