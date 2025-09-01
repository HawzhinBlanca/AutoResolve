#!/usr/bin/env python3
"""
Render Farm Integration and Management
Distributed rendering across multiple machines with job queue management
"""

import os
import json
import uuid
import time
import socket
import asyncio
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
import threading
from queue import PriorityQueue
import psutil
import platform
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import redis
import pickle

logger = logging.getLogger(__name__)

class RenderJobStatus(Enum):
    """Render job status"""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class RenderPriority(Enum):
    """Render job priority levels"""
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class NodeStatus(Enum):
    """Render node status"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class RenderEngine(Enum):
    """Supported render engines"""
    FFMPEG = "ffmpeg"
    DAVINCI = "davinci"
    BLENDER = "blender"
    AFTER_EFFECTS = "after_effects"
    NUKE = "nuke"
    CUSTOM = "custom"

@dataclass
class RenderJob:
    """Render job definition"""
    job_id: str
    project_id: str
    input_file: str
    output_file: str
    settings: Dict[str, Any]
    engine: RenderEngine
    priority: RenderPriority
    status: RenderJobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_node: Optional[str] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    estimated_time: Optional[float] = None
    frame_range: Optional[Tuple[int, int]] = None
    segments: Optional[List[Tuple[int, int]]] = None
    
    def __lt__(self, other):
        """For priority queue sorting"""
        return self.priority.value < other.priority.value

@dataclass
class RenderNode:
    """Render farm node"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus
    cpu_cores: int
    memory_gb: float
    gpu_info: Optional[Dict[str, Any]]
    supported_engines: List[RenderEngine]
    current_job: Optional[str] = None
    performance_score: float = 1.0
    last_heartbeat: Optional[datetime] = None
    total_jobs_completed: int = 0
    total_render_time: float = 0.0

@dataclass
class RenderStatistics:
    """Render farm statistics"""
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_render_time: float
    total_render_time: float
    node_utilization: Dict[str, float]
    queue_length: int
    estimated_queue_time: float

class RenderJobScheduler:
    """Job scheduling and distribution"""
    
    def __init__(self):
        self.job_queue = PriorityQueue()
        self.nodes: Dict[str, RenderNode] = {}
        self.jobs: Dict[str, RenderJob] = {}
        self.lock = threading.Lock()
        
    def add_job(self, job: RenderJob):
        """Add job to queue"""
        with self.lock:
            self.jobs[job.job_id] = job
            self.job_queue.put((job.priority.value, time.time(), job))
            logger.info(f"Job {job.job_id} added to queue with priority {job.priority.name}")
    
    def assign_job_to_node(self) -> Optional[Tuple[RenderJob, RenderNode]]:
        """Assign next job to available node"""
        with self.lock:
            if self.job_queue.empty():
                return None
            
            # Find available node
            available_nodes = [
                node for node in self.nodes.values()
                if node.status == NodeStatus.IDLE
            ]
            
            if not available_nodes:
                return None
            
            # Get highest priority job
            _, _, job = self.job_queue.get()
            
            # Select best node based on performance
            best_node = max(available_nodes, key=lambda n: n.performance_score)
            
            # Assign job
            job.status = RenderJobStatus.ASSIGNED
            job.assigned_node = best_node.node_id
            best_node.status = NodeStatus.BUSY
            best_node.current_job = job.job_id
            
            logger.info(f"Assigned job {job.job_id} to node {best_node.node_id}")
            return job, best_node
    
    def update_node_status(self, node_id: str, status: NodeStatus):
        """Update node status"""
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
                self.nodes[node_id].last_heartbeat = datetime.now()
    
    def complete_job(self, job_id: str, success: bool, render_time: float):
        """Mark job as completed"""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = RenderJobStatus.COMPLETED if success else RenderJobStatus.FAILED
                job.completed_at = datetime.now()
                
                # Update node stats
                if job.assigned_node and job.assigned_node in self.nodes:
                    node = self.nodes[job.assigned_node]
                    node.status = NodeStatus.IDLE
                    node.current_job = None
                    node.total_jobs_completed += 1
                    node.total_render_time += render_time
                    
                    # Update performance score
                    if job.estimated_time and job.estimated_time > 0:
                        accuracy = min(1.0, job.estimated_time / render_time)
                        node.performance_score = 0.9 * node.performance_score + 0.1 * accuracy

class RenderNodeWorker:
    """Worker process for render node"""
    
    def __init__(self, node_id: str, master_url: str):
        self.node_id = node_id
        self.master_url = master_url
        self.current_job: Optional[RenderJob] = None
        self.running = True
        self.render_engines = self._detect_render_engines()
        
    def _detect_render_engines(self) -> List[RenderEngine]:
        """Detect available render engines"""
        engines = []
        
        # Check for FFmpeg
        if shutil.which("ffmpeg"):
            engines.append(RenderEngine.FFMPEG)
        
        # Check for DaVinci Resolve
        if platform.system() == "Darwin":
            if Path("/Applications/DaVinci Resolve.app").exists():
                engines.append(RenderEngine.DAVINCI)
        
        # Check for Blender
        if shutil.which("blender"):
            engines.append(RenderEngine.BLENDER)
        
        return engines
    
    async def start(self):
        """Start worker loop"""
        logger.info(f"Render node {self.node_id} started")
        
        while self.running:
            try:
                # Send heartbeat
                await self.send_heartbeat()
                
                # Request job
                job = await self.request_job()
                
                if job:
                    # Render job
                    success = await self.render_job(job)
                    
                    # Report completion
                    await self.report_completion(job, success)
                else:
                    # No job available, wait
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Node worker error: {e}")
                await asyncio.sleep(10)
    
    async def send_heartbeat(self):
        """Send heartbeat to master"""
        async with aiohttp.ClientSession() as session:
            data = {
                "node_id": self.node_id,
                "status": "idle" if not self.current_job else "busy",
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "timestamp": time.time()
            }
            
            try:
                async with session.post(f"{self.master_url}/heartbeat", json=data) as resp:
                    if resp.status != 200:
                        logger.warning(f"Heartbeat failed: {resp.status}")
            except:
                pass
    
    async def request_job(self) -> Optional[RenderJob]:
        """Request job from master"""
        if self.current_job:
            return None
        
        async with aiohttp.ClientSession() as session:
            data = {
                "node_id": self.node_id,
                "supported_engines": [e.value for e in self.render_engines]
            }
            
            try:
                async with session.post(f"{self.master_url}/request_job", json=data) as resp:
                    if resp.status == 200:
                        job_data = await resp.json()
                        if job_data:
                            # Reconstruct job object
                            job = RenderJob(**job_data)
                            self.current_job = job
                            return job
            except Exception as e:
                logger.error(f"Failed to request job: {e}")
        
        return None
    
    async def render_job(self, job: RenderJob) -> bool:
        """Execute render job"""
        logger.info(f"Starting render job {job.job_id}")
        
        try:
            if job.engine == RenderEngine.FFMPEG:
                return await self._render_ffmpeg(job)
            elif job.engine == RenderEngine.DAVINCI:
                return await self._render_davinci(job)
            elif job.engine == RenderEngine.BLENDER:
                return await self._render_blender(job)
            else:
                logger.error(f"Unsupported render engine: {job.engine}")
                return False
                
        except Exception as e:
            logger.error(f"Render failed: {e}")
            return False
    
    async def _render_ffmpeg(self, job: RenderJob) -> bool:
        """Render using FFmpeg"""
        settings = job.settings
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", job.input_file,
            "-c:v", settings.get("video_codec", "libx264"),
            "-preset", settings.get("preset", "medium"),
            "-crf", str(settings.get("quality", 23)),
            "-c:a", settings.get("audio_codec", "aac"),
            "-b:a", f"{settings.get('audio_bitrate', 192)}k"
        ]
        
        # Add resolution if specified
        if "resolution" in settings:
            cmd.extend(["-s", settings["resolution"]])
        
        # Add frame range if specified
        if job.frame_range:
            start, end = job.frame_range
            cmd.extend(["-ss", str(start), "-to", str(end)])
        
        cmd.extend(["-y", job.output_file])
        
        # Run render
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Monitor progress
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            
            # Parse progress
            line_str = line.decode('utf-8')
            if "time=" in line_str:
                # Extract progress and report
                await self.report_progress(job, self._parse_ffmpeg_progress(line_str))
        
        await process.wait()
        return process.returncode == 0
    
    async def _render_davinci(self, job: RenderJob) -> bool:
        """Render using DaVinci Resolve"""
        # This would use DaVinci Resolve's Python API
        # Placeholder implementation
        logger.info(f"DaVinci render for job {job.job_id}")
        await asyncio.sleep(10)  # Simulate rendering
        return True
    
    async def _render_blender(self, job: RenderJob) -> bool:
        """Render using Blender"""
        settings = job.settings
        
        cmd = [
            "blender",
            "-b", job.input_file,
            "-o", job.output_file,
            "-F", settings.get("format", "PNG"),
            "-x", "1"
        ]
        
        # Add frame range
        if job.frame_range:
            start, end = job.frame_range
            cmd.extend(["-s", str(start), "-e", str(end)])
        
        cmd.append("-a")  # Render animation
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.wait()
        return process.returncode == 0
    
    async def report_progress(self, job: RenderJob, progress: float):
        """Report job progress to master"""
        async with aiohttp.ClientSession() as session:
            data = {
                "job_id": job.job_id,
                "node_id": self.node_id,
                "progress": progress,
                "timestamp": time.time()
            }
            
            try:
                await session.post(f"{self.master_url}/progress", json=data)
            except:
                pass
    
    async def report_completion(self, job: RenderJob, success: bool):
        """Report job completion"""
        async with aiohttp.ClientSession() as session:
            data = {
                "job_id": job.job_id,
                "node_id": self.node_id,
                "success": success,
                "output_file": job.output_file if success else None,
                "timestamp": time.time()
            }
            
            try:
                async with session.post(f"{self.master_url}/complete", json=data) as resp:
                    if resp.status == 200:
                        self.current_job = None
            except Exception as e:
                logger.error(f"Failed to report completion: {e}")
    
    def _parse_ffmpeg_progress(self, line: str) -> float:
        """Parse FFmpeg progress from output"""
        try:
            if "time=" in line:
                time_str = line.split("time=")[1].split()[0]
                # Convert to seconds
                parts = time_str.split(":")
                seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                
                # Calculate progress percentage
                if self.current_job and self.current_job.settings.get("duration"):
                    duration = self.current_job.settings["duration"]
                    return min(100.0, (seconds / duration) * 100)
        except:
            pass
        
        return 0.0

class RenderFarmMaster:
    """Central render farm controller"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.scheduler = RenderJobScheduler()
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.statistics = RenderStatistics(
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            average_render_time=0.0,
            total_render_time=0.0,
            node_utilization={},
            queue_length=0,
            estimated_queue_time=0.0
        )
        
    def submit_job(
        self,
        input_file: str,
        output_file: str,
        settings: Dict[str, Any],
        engine: RenderEngine = RenderEngine.FFMPEG,
        priority: RenderPriority = RenderPriority.NORMAL
    ) -> str:
        """Submit new render job"""
        job = RenderJob(
            job_id=str(uuid.uuid4()),
            project_id=settings.get("project_id", "default"),
            input_file=input_file,
            output_file=output_file,
            settings=settings,
            engine=engine,
            priority=priority,
            status=RenderJobStatus.QUEUED,
            created_at=datetime.now()
        )
        
        # Estimate render time
        job.estimated_time = self._estimate_render_time(job)
        
        # Check if job should be segmented
        if settings.get("enable_segmentation", False):
            job.segments = self._segment_job(job)
        
        # Add to scheduler
        self.scheduler.add_job(job)
        
        # Store in Redis
        self.redis_client.set(f"job:{job.job_id}", pickle.dumps(job))
        
        # Update statistics
        self.statistics.total_jobs += 1
        self.statistics.queue_length = self.scheduler.job_queue.qsize()
        
        logger.info(f"Job {job.job_id} submitted")
        return job.job_id
    
    def register_node(self, node: RenderNode):
        """Register new render node"""
        self.scheduler.nodes[node.node_id] = node
        
        # Store in Redis
        self.redis_client.set(f"node:{node.node_id}", pickle.dumps(node))
        
        logger.info(f"Node {node.node_id} registered")
    
    def get_job_status(self, job_id: str) -> Optional[RenderJob]:
        """Get job status"""
        if job_id in self.scheduler.jobs:
            return self.scheduler.jobs[job_id]
        
        # Check Redis
        job_data = self.redis_client.get(f"job:{job_id}")
        if job_data:
            return pickle.loads(job_data)
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel render job"""
        if job_id in self.scheduler.jobs:
            job = self.scheduler.jobs[job_id]
            
            if job.status in [RenderJobStatus.QUEUED, RenderJobStatus.RENDERING]:
                job.status = RenderJobStatus.CANCELLED
                
                # Notify node if rendering
                if job.assigned_node:
                    # Send cancellation signal to node
                    pass
                
                logger.info(f"Job {job_id} cancelled")
                return True
        
        return False
    
    def get_statistics(self) -> RenderStatistics:
        """Get render farm statistics"""
        # Calculate node utilization
        for node_id, node in self.scheduler.nodes.items():
            if node.status == NodeStatus.BUSY:
                self.statistics.node_utilization[node_id] = 1.0
            elif node.status == NodeStatus.IDLE:
                self.statistics.node_utilization[node_id] = 0.0
            else:
                self.statistics.node_utilization[node_id] = -1.0
        
        # Update queue metrics
        self.statistics.queue_length = self.scheduler.job_queue.qsize()
        
        # Estimate queue time
        if self.statistics.average_render_time > 0:
            active_nodes = sum(1 for n in self.scheduler.nodes.values() 
                             if n.status in [NodeStatus.IDLE, NodeStatus.BUSY])
            if active_nodes > 0:
                self.statistics.estimated_queue_time = (
                    self.statistics.queue_length * self.statistics.average_render_time / active_nodes
                )
        
        return self.statistics
    
    def _estimate_render_time(self, job: RenderJob) -> float:
        """Estimate render time for job"""
        # Simple estimation based on file size and settings
        # In production, use ML model trained on historical data
        
        base_time = 60.0  # Base time in seconds
        
        # Factor in resolution
        if "resolution" in job.settings:
            width, height = map(int, job.settings["resolution"].split("x"))
            pixels = width * height
            base_time *= (pixels / (1920 * 1080))  # Normalize to 1080p
        
        # Factor in quality
        quality = job.settings.get("quality", 23)
        base_time *= (51 - quality) / 28  # Lower CRF = higher quality = more time
        
        # Factor in codec
        codec = job.settings.get("video_codec", "libx264")
        codec_multipliers = {
            "libx264": 1.0,
            "libx265": 2.5,
            "libvpx-vp9": 3.0,
            "prores": 0.5
        }
        base_time *= codec_multipliers.get(codec, 1.0)
        
        return base_time
    
    def _segment_job(self, job: RenderJob) -> List[Tuple[int, int]]:
        """Segment job for distributed rendering"""
        # Get total frames
        total_frames = job.settings.get("total_frames", 1000)
        
        # Calculate segment size
        segment_size = max(100, total_frames // 10)  # At least 100 frames per segment
        
        segments = []
        for start in range(0, total_frames, segment_size):
            end = min(start + segment_size, total_frames)
            segments.append((start, end))
        
        return segments

# Example usage
if __name__ == "__main__":
    # Initialize render farm
    farm = RenderFarmMaster()
    
    # Register nodes
    node = RenderNode(
        node_id="node-001",
        hostname="render-node-1",
        ip_address="192.168.1.100",
        port=8000,
        status=NodeStatus.IDLE,
        cpu_cores=16,
        memory_gb=32.0,
        gpu_info={"model": "RTX 3090", "memory": 24},
        supported_engines=[RenderEngine.FFMPEG, RenderEngine.BLENDER]
    )
    
    farm.register_node(node)
    
    # Submit job
    job_id = farm.submit_job(
        "/path/to/input.mp4",
        "/path/to/output.mp4",
        {
            "video_codec": "libx264",
            "quality": 23,
            "resolution": "1920x1080",
            "audio_codec": "aac",
            "audio_bitrate": 192
        },
        engine=RenderEngine.FFMPEG,
        priority=RenderPriority.HIGH
    )
    
    print(f"✅ Job submitted: {job_id}")
    
    # Get statistics
    stats = farm.get_statistics()
    print(f"📊 Farm statistics: Queue={stats.queue_length}, Nodes={len(farm.scheduler.nodes)}")