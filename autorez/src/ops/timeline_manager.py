"""
Timeline Manager - Persistent timeline state management
Part of AutoResolve V3.0 Production Pipeline
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Pydantic models for timeline operations
class TimelineClip(BaseModel):
    id: str
    name: str
    track_index: int
    start_time: float
    duration: float
    source_url: Optional[str] = None
    in_point: float = 0.0
    out_point: Optional[float] = None
    effects: List[Dict[str, Any]] = []
    transitions: Dict[str, Any] = {}

class TimePosition(BaseModel):
    track_index: int
    start_time: float

class MoveClipRequest(BaseModel):
    clip_id: str
    position: TimePosition

class TimelineProject(BaseModel):
    id: str
    name: str
    created_at: str
    modified_at: str
    clips: List[TimelineClip]
    settings: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class TimelineManager:
    """Manages persistent timeline state with SQLite backend"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_dir = Path(os.getenv("TIMELINE_DB_DIR", "/Users/hawzhin/AutoResolve/timeline_db"))
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "timeline.db")
        
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with timeline schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settings TEXT,
                    metadata TEXT
                )
            """)
            
            # Clips table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clips (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    track_index INTEGER NOT NULL,
                    start_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    source_url TEXT,
                    in_point REAL DEFAULT 0,
                    out_point REAL,
                    effects TEXT,
                    transitions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            
            # Create index for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clips_project ON clips(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clips_time ON clips(start_time)")
            
            conn.commit()
    
    async def create_project(self, name: str) -> str:
        """Create a new timeline project"""
        project_id = f"project_{int(time.time() * 1000)}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO projects (id, name, settings, metadata)
                VALUES (?, ?, ?, ?)
            """, (project_id, name, '{}', '{}'))
            conn.commit()
        
        logger.info(f"Created timeline project: {project_id}")
        return project_id
    
    async def add_clip(self, project_id: str, clip: TimelineClip) -> Dict:
        """Add a clip to the timeline"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check for collisions
            cursor.execute("""
                SELECT id FROM clips 
                WHERE project_id = ? AND track_index = ?
                AND ((start_time <= ? AND start_time + duration > ?)
                  OR (start_time < ? AND start_time + duration >= ?))
            """, (project_id, clip.track_index, 
                  clip.start_time, clip.start_time,
                  clip.start_time + clip.duration, clip.start_time + clip.duration))
            
            collisions = cursor.fetchall()
            if collisions:
                logger.warning(f"Clip collision detected at track {clip.track_index}, time {clip.start_time}")
            
            # Insert clip
            cursor.execute("""
                INSERT INTO clips (id, project_id, name, track_index, start_time, 
                                 duration, source_url, in_point, out_point, effects, transitions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (clip.id, project_id, clip.name, clip.track_index, clip.start_time,
                  clip.duration, clip.source_url, clip.in_point, clip.out_point,
                  json.dumps(clip.effects), json.dumps(clip.transitions)))
            
            conn.commit()
        
        logger.info(f"Added clip {clip.id} to project {project_id}")
        return {"status": "added", "clip_id": clip.id, "collisions": len(collisions) > 0}
    
    async def move_clip(self, project_id: str, clip_id: str, position: TimePosition) -> Dict:
        """Move a clip to a new position"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current clip info
            cursor.execute("""
                SELECT duration FROM clips WHERE id = ? AND project_id = ?
            """, (clip_id, project_id))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Clip {clip_id} not found")
            
            duration = result[0]
            
            # Check for collisions at new position
            cursor.execute("""
                SELECT id FROM clips 
                WHERE project_id = ? AND track_index = ? AND id != ?
                AND ((start_time <= ? AND start_time + duration > ?)
                  OR (start_time < ? AND start_time + duration >= ?))
            """, (project_id, position.track_index, clip_id,
                  position.start_time, position.start_time,
                  position.start_time + duration, position.start_time + duration))
            
            collisions = cursor.fetchall()
            
            # Update clip position
            cursor.execute("""
                UPDATE clips 
                SET track_index = ?, start_time = ?
                WHERE id = ? AND project_id = ?
            """, (position.track_index, position.start_time, clip_id, project_id))
            
            # Update project modified time
            cursor.execute("""
                UPDATE projects SET modified_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (project_id,))
            
            conn.commit()
        
        logger.info(f"Moved clip {clip_id} to track {position.track_index}, time {position.start_time}")
        return {"status": "moved", "collisions": [c[0] for c in collisions]}
    
    async def delete_clip(self, project_id: str, clip_id: str) -> Dict:
        """Delete a clip from the timeline"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM clips WHERE id = ? AND project_id = ?
            """, (clip_id, project_id))
            
            deleted = cursor.rowcount > 0
            
            if deleted:
                cursor.execute("""
                    UPDATE projects SET modified_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (project_id,))
            
            conn.commit()
        
        if deleted:
            logger.info(f"Deleted clip {clip_id} from project {project_id}")
            return {"status": "deleted", "clip_id": clip_id}
        else:
            return {"status": "not_found", "clip_id": clip_id}
    
    async def get_timeline(self, project_id: str) -> Dict:
        """Get complete timeline for a project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get project info
            cursor.execute("""
                SELECT name, created_at, modified_at, settings, metadata
                FROM projects WHERE id = ?
            """, (project_id,))
            
            project = cursor.fetchone()
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            # Get all clips
            cursor.execute("""
                SELECT id, name, track_index, start_time, duration, 
                       source_url, in_point, out_point, effects, transitions
                FROM clips WHERE project_id = ?
                ORDER BY track_index, start_time
            """, (project_id,))
            
            clips = []
            for row in cursor.fetchall():
                clips.append({
                    "id": row[0],
                    "name": row[1],
                    "track_index": row[2],
                    "start_time": row[3],
                    "duration": row[4],
                    "source_url": row[5],
                    "in_point": row[6],
                    "out_point": row[7],
                    "effects": json.loads(row[8] or "[]"),
                    "transitions": json.loads(row[9] or "{}")
                })
        
        return {
            "id": project_id,
            "name": project[0],
            "created_at": project[1],
            "modified_at": project[2],
            "settings": json.loads(project[3] or "{}"),
            "metadata": json.loads(project[4] or "{}"),
            "clips": clips
        }
    
    async def save_as_arz(self, project_id: str, output_path: str) -> str:
        """Save timeline as .arz project file"""
        timeline = await self.get_timeline(project_id)
        
        # Add file format version
        timeline["format_version"] = "1.0"
        timeline["saved_at"] = datetime.now().isoformat()
        
        # Ensure output has .arz extension
        if not output_path.endswith(".arz"):
            output_path += ".arz"
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(timeline, f, indent=2)
        
        logger.info(f"Saved project {project_id} to {output_path}")
        return output_path
    
    async def load_from_arz(self, file_path: str) -> str:
        """Load timeline from .arz project file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Create new project
        project_id = await self.create_project(data.get("name", "Imported Project"))
        
        # Add all clips
        for clip_data in data.get("clips", []):
            clip = TimelineClip(**clip_data)
            await self.add_clip(project_id, clip)
        
        logger.info(f"Loaded project from {file_path} as {project_id}")
        return project_id

# Export singleton instance
timeline_manager = TimelineManager()