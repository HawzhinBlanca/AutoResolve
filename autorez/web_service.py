#!/usr/bin/env python3
"""
AutoResolve Web Service
REST API backend for the AutoResolve Neural Timeline frontend
"""

import sys
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

# Web framework
from aiohttp import web, WSMsgType
import aiohttp_cors

# AutoResolve modules - simplified imports to avoid errors
try:
    from src.utils.memory import rss_gb
except ImportError:
    def rss_gb():
        import psutil
        return psutil.Process().memory_info().rss / (1024**3)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TaskStatus:
    task_id: str
    status: str  # queued, running, completed, failed
    progress: float = 0.0
    stage: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None
        }

class AutoResolveWebService:
    def __init__(self):
        self.app = web.Application()
        self.tasks: Dict[str, TaskStatus] = {}
        self.setup_routes()
        self.setup_cors()
        
        # Initialize components (simplified)
        self.components_initialized = False
        
    def setup_routes(self):
        """Setup REST API routes"""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/api/projects', self.get_projects)
        self.app.router.add_post('/api/pipeline/start', self.start_pipeline)
        self.app.router.add_get('/api/pipeline/status/{task_id}', self.get_pipeline_status)
        self.app.router.add_post('/api/pipeline/cancel/{task_id}', self.cancel_pipeline)
        self.app.router.add_post('/api/validate', self.validate_config)
        self.app.router.add_get('/api/presets', self.get_presets)
        self.app.router.add_post('/api/presets', self.save_preset)
        
    def setup_cors(self):
        """Setup CORS for development"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "memory_usage_gb": rss_gb(),
            "active_tasks": len([t for t in self.tasks.values() if t.status == "running"])
        })
    
    async def get_projects(self, request):
        """Get available projects"""
        try:
            # Check for DaVinci Resolve projects
            projects = ["Sample Project 1", "Sample Project 2", "Sample Project 3"]
            return web.json_response({"projects": projects})
        except Exception as e:
            logger.error(f"Error getting projects: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def validate_config(self, request):
        """Validate configuration"""
        try:
            data = await request.json()
            input_file = data.get('input_file', '')
            
            errors = []
            info = {}
            
            # Check input file
            if not input_file or not Path(input_file).exists():
                errors.append(f"Input file not found: {input_file}")
            else:
                info['file_size'] = Path(input_file).stat().st_size
                info['file_exists'] = True
            
            # Check config files
            conf_errors = []
            if not Path("conf/director.ini").exists():
                conf_errors.append("Config file not found: conf/director.ini")
            if not Path("conf/embeddings.ini").exists():
                conf_errors.append("Config file not found: conf/embeddings.ini")
            if not Path("conf/ops.ini").exists():
                conf_errors.append("Config file not found: conf/ops.ini")
            
            # Check DaVinci Resolve
            resolve_errors = []
            resolve_info = {"resolve_running": False}
            try:
                # This would attempt to connect to Resolve
                resolve_info["resolve_running"] = False
                resolve_errors.append("DaVinci Resolve not connected")
            except Exception:
                resolve_errors.append("Cannot connect to Resolve API")
            
            return web.json_response({
                "input": {"errors": errors, "info": info},
                "config": {"errors": conf_errors},
                "resolve": {"errors": resolve_errors, "info": resolve_info}
            })
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def start_pipeline(self, request):
        """Start processing pipeline"""
        try:
            data = await request.json()
            input_file = data.get('input_file', '')
            options = data.get('options', {})
            
            # Generate task ID
            import hashlib
            import time
            task_id = hashlib.md5(f"{input_file}_{time.time()}".encode()).hexdigest()[:8]
            
            # Create task
            task = TaskStatus(
                task_id=task_id,
                status="queued",
                message="Task queued for processing",
                started_at=datetime.now()
            )
            self.tasks[task_id] = task
            
            # Start processing in background
            asyncio.create_task(self.process_pipeline(task_id, input_file, options))
            
            return web.json_response({"task_id": task_id})
            
        except Exception as e:
            logger.error(f"Pipeline start error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def process_pipeline(self, task_id: str, input_file: str, options: Dict[str, Any]):
        """Process the pipeline in background"""
        try:
            task = self.tasks[task_id]
            task.status = "running"
            task.stage = "validation"
            task.message = "Validating input..."
            
            # Validate input
            if not Path(input_file).exists():
                task.status = "failed"
                task.error = f"Input file not found: {input_file}"
                return
            
            # Simulate processing stages
            stages = [
                ("initialization", "Initializing components..."),
                ("analysis", "Analyzing video content..."),
                ("processing", "Processing with AI Director..."),
                ("finalization", "Finalizing results...")
            ]
            
            for i, (stage, message) in enumerate(stages):
                task.stage = stage
                task.message = message
                task.progress = (i + 1) / len(stages)
                
                # Simulate work
                await asyncio.sleep(2)
                
                if task.status == "cancelled":
                    return
            
            # Complete task
            task.status = "completed"
            task.progress = 1.0
            task.message = "Pipeline completed successfully"
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"Pipeline processing error: {e}")
            logger.error(traceback.format_exc())
    
    async def get_pipeline_status(self, request):
        """Get pipeline status"""
        task_id = request.match_info['task_id']
        
        if task_id not in self.tasks:
            return web.json_response({"error": "Task not found"}, status=404)
        
        task = self.tasks[task_id]
        return web.json_response(task.to_dict())
    
    async def cancel_pipeline(self, request):
        """Cancel pipeline"""
        task_id = request.match_info['task_id']
        
        if task_id not in self.tasks:
            return web.json_response({"error": "Task not found"}, status=404)
        
        task = self.tasks[task_id]
        task.status = "cancelled"
        task.message = "Task cancelled by user"
        
        return web.json_response({"status": "cancelled"})
    
    async def get_presets(self, request):
        """Get available presets"""
        try:
            presets = [
                {
                    "id": "youtube_1080p",
                    "name": "YouTube 1080p",
                    "settings": {
                        "resolution": "1920x1080",
                        "format": "mp4",
                        "quality": "high"
                    }
                },
                {
                    "id": "shorts_vertical",
                    "name": "Shorts (Vertical)",
                    "settings": {
                        "resolution": "1080x1920",
                        "format": "mp4",
                        "quality": "high"
                    }
                }
            ]
            return web.json_response({"presets": presets})
        except Exception as e:
            logger.error(f"Error getting presets: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def save_preset(self, request):
        """Save new preset"""
        try:
            data = await request.json()
            name = data.get('name', 'custom')
            settings = data.get('settings', {})
            
            # In a real implementation, save to file
            logger.info(f"Saving preset: {name} with settings: {settings}")
            
            return web.json_response({
                "status": "saved",
                "name": name,
                "path": f"presets/{name}.json"
            })
        except Exception as e:
            logger.error(f"Error saving preset: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    def run(self, host='localhost', port=8081):
        """Start the web service"""
        logger.info(f"Starting AutoResolve Web Service on http://{host}:{port}")
        web.run_app(self.app, host=host, port=port)

if __name__ == '__main__':
    service = AutoResolveWebService()
    service.run()