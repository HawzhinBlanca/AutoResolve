# AutoResolve v3.2 - API Documentation

## Base URL
```
Production: https://api.autoresolve.com
Development: http://localhost:8000
```

## Authentication

### API Key (Recommended)
Add header: `x-api-key: YOUR_API_KEY`

### JWT Bearer Token
1. Login to get tokens: `POST /auth/login`
2. Use access token: `Authorization: Bearer ACCESS_TOKEN`
3. Refresh when expired: `POST /auth/refresh`

---

## Core Endpoints

### System Health
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "pipeline": "ready", 
  "memory_mb": 2048,
  "memory_usage_gb": 2.0,
  "active_tasks": 0
}
```

### Start Video Processing
```http
POST /api/pipeline/start
Authorization: x-api-key or Bearer token
Content-Type: application/json

{
  "video_path": "/path/to/video.mp4",
  "output_path": "/path/to/output/", // optional
  "settings": {                       // optional
    "silence_threshold": -34,
    "min_keep_duration": 0.4
  }
}
```

**Response:**
```json
{
  "task_id": "task_1693234567890",
  "status": "started"
}
```

### Get Processing Status
```http
GET /api/pipeline/status/{task_id}
Authorization: x-api-key or Bearer token
```

**Response:**
```json
{
  "task_id": "task_1693234567890",
  "status": "processing",     // pending, processing, completed, failed, cancelled
  "progress": 0.65,          // 0.0 - 1.0
  "result": null,            // populated when completed
  "error": null,             // populated if failed
  "started_at": "2023-08-28T15:22:47.890Z"
}
```

### Cancel Processing
```http
POST /api/pipeline/cancel/{task_id}
Authorization: x-api-key or Bearer token
```

---

## Export Endpoints

### Export as FCPXML
```http
POST /api/export/fcpxml?task_id={task_id}
Authorization: x-api-key or Bearer token
```

**Response:**
```json
{
  "status": "exported",
  "format": "fcpxml",
  "path": "/exports/task_123.fcpxml"
}
```

### Export as EDL
```http
POST /api/export/edl?task_id={task_id}
Authorization: x-api-key or Bearer token
```

### Export as MP4
```http
POST /api/export/mp4
Authorization: x-api-key or Bearer token
Content-Type: application/json

{
  "task_id": "task_123",
  "resolution": "1920x1080",  // optional, default: 1920x1080
  "fps": 30,                  // optional, default: 30
  "preset": "medium",         // optional, default: medium
  "crf": 23                   // optional, default: 23
}
```

---

## Timeline Management

### Create Project
```http
POST /api/timeline/project
Authorization: x-api-key
Content-Type: application/json

{
  "name": "My Project"
}
```

### Add Clip to Timeline
```http
POST /api/timeline/clips?project_id={project_id}
Authorization: x-api-key
Content-Type: application/json

{
  "id": "clip_001",
  "name": "Main Video Segment",
  "track_index": 0,
  "start_time": 10.5,
  "duration": 30.0,
  "source_url": "/path/to/video.mp4",
  "in_point": 0.0,
  "out_point": 30.0,
  "effects": [],
  "transitions": {}
}
```

### Move Clip
```http
PUT /api/timeline/clips/{clip_id}/move?project_id={project_id}
Authorization: x-api-key
Content-Type: application/json

{
  "track_index": 1,
  "start_time": 45.0
}
```

### Get Timeline
```http
GET /api/timeline/{project_id}
Authorization: x-api-key
```

---

## Specialized Operations

### Silence Detection Only
```http
POST /api/silence/detect
Authorization: x-api-key
Content-Type: application/json

{
  "video_path": "/path/to/video.mp4"
}
```

**Response:**
```json
{
  "status": "success",
  "keep_windows": [
    {"start": 0.0, "end": 10.5},
    {"start": 15.2, "end": 45.8}
  ],
  "silence_regions": [
    {"start": 10.5, "end": 15.2, "duration": 4.7}
  ],
  "total_silence": 4.7
}
```

### Validate Input File
```http
POST /api/validate
Authorization: x-api-key
Content-Type: application/json

{
  "input_file": "/path/to/video.mp4"
}
```

---

## Real-time Updates

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/status');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'connected':
      console.log('WebSocket connected');
      break;
      
    case 'progress':
      console.log(`Task ${data.task_id}: ${(data.progress * 100).toFixed(1)}%`);
      break;
      
    case 'completed':
      console.log(`Task ${data.task_id} completed:`, data.result);
      break;
      
    case 'error':
      console.log(`Task ${data.task_id} failed:`, data.error);
      break;
      
    case 'heartbeat':
      // Server alive signal every 1.5s
      break;
  }
};
```

---

## Error Handling

### Standard Error Response
```json
{
  "detail": "Error description",
  "status_code": 400
}
```

### Common Error Codes
- `400` - Invalid request (bad video path, invalid parameters)
- `401` - Unauthorized (missing/invalid API key or token)
- `404` - Resource not found (task ID, project ID)
- `422` - Security scan rejected (malicious content detected)
- `429` - Rate limit exceeded (10 requests/minute)
- `500` - Internal server error

---

## Rate Limits

- **Pipeline Start**: 10 requests per minute per IP
- **API Endpoints**: No specific limit (protected by auth)
- **WebSocket**: No limit (connection-based)

---

## Performance Guarantees

### Processing Speed
- **Minimum**: 30x realtime processing
- **Typical**: 51x realtime processing

### API Response Times
- **Health check**: < 50ms
- **Start pipeline**: < 100ms (excluding processing)
- **Export operations**: < 2000ms
- **Status queries**: < 100ms

### Memory Limits
- **Peak RAM**: < 16GB during V-JEPA processing
- **UI Memory**: < 200MB for frontend
- **Steady state**: < 1GB

---

## Example Workflows

### Complete Video Processing
```bash
# 1. Start processing
curl -X POST http://localhost:8000/api/pipeline/start \
  -H "x-api-key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'

# Returns: {"task_id": "task_123", "status": "started"}

# 2. Monitor progress via WebSocket or polling
curl http://localhost:8000/api/pipeline/status/task_123 \
  -H "x-api-key: your-key"

# 3. Export when complete
curl -X POST http://localhost:8000/api/export/fcpxml?task_id=task_123 \
  -H "x-api-key: your-key"
```

### Timeline Editing
```bash
# 1. Create project
curl -X POST http://localhost:8000/api/timeline/project \
  -H "x-api-key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Project"}'

# 2. Add clips
curl -X POST http://localhost:8000/api/timeline/clips?project_id=project_123 \
  -H "x-api-key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "clip_001",
    "name": "Main Video", 
    "track_index": 0,
    "start_time": 0.0,
    "duration": 60.0,
    "source_url": "/path/to/video.mp4"
  }'

# 3. Export timeline
curl -X POST http://localhost:8000/api/timeline/project_123/export \
  -H "x-api-key: your-key" \
  -d '{"format": "arz"}'
```

---

## Configuration

### Environment Variables
```bash
# Required
API_KEY=your-secret-api-key

# Optional 
BACKEND_PORT=8000
JWT_SECRET=your-jwt-secret
EXPORT_DIR=/path/to/exports
TEMP_DIR=/path/to/temp

# Performance tuning
VJEPA_MEMORY_SAFE=true
CLIP_BATCH_SIZE=32
MAX_CONCURRENT_TASKS=4
```

### Configuration Files
- `conf/embeddings.ini` - Embedder settings
- `conf/director.ini` - Creative director parameters  
- `conf/ops.ini` - Operation-specific settings
- `conf/logging.ini` - Logging configuration

---

## Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
export PYTHONPATH=/path/to/autorez/src:$PYTHONPATH
```

#### Memory issues with V-JEPA
```bash
# Enable memory-safe mode
export VJEPA_MEMORY_SAFE=true
```

#### WebSocket connection fails
- Verify WebSocket URL: `ws://localhost:8000/ws/status` (not `/ws/progress`)
- Check CORS settings allow WebSocket connections

#### Rate limiting
- Wait 60 seconds for rate limit reset
- Use different IP or API key for testing

### Support
For issues and support:
- Check logs: `tail -f logs/autoresolve.log`
- Run health checks: `python scripts/startup_checks.py`
- Performance check: `make verify-gates`