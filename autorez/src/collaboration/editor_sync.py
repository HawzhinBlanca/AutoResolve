#!/usr/bin/env python3
"""
Collaborative Editing System
Real-time multi-user editing with conflict resolution and version control
"""

import asyncio
import json
import uuid
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging
import hashlib
from pathlib import Path
import aioredis
import websockets
from websockets.server import WebSocketServerProtocol
import sqlite3
import difflib
import pickle

logger = logging.getLogger(__name__)

class EditOperationType(Enum):
    """Types of edit operations"""
    ADD_CLIP = "add_clip"
    REMOVE_CLIP = "remove_clip"
    MOVE_CLIP = "move_clip"
    TRIM_CLIP = "trim_clip"
    ADD_EFFECT = "add_effect"
    REMOVE_EFFECT = "remove_effect"
    MODIFY_EFFECT = "modify_effect"
    ADD_TRANSITION = "add_transition"
    ADD_MARKER = "add_marker"
    MODIFY_AUDIO = "modify_audio"
    COLOR_GRADE = "color_grade"
    ADD_TEXT = "add_text"

class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE = "merge"
    MANUAL = "manual"
    OPERATIONAL_TRANSFORM = "operational_transform"

class UserRole(Enum):
    """User roles in collaborative session"""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    REVIEWER = "reviewer"

@dataclass
class User:
    """Collaborative user"""
    user_id: str
    username: str
    email: str
    role: UserRole
    color: str  # For UI highlighting
    avatar_url: Optional[str] = None
    active: bool = True
    last_seen: datetime = None

@dataclass
class EditOperation:
    """Single edit operation"""
    operation_id: str
    user_id: str
    operation_type: EditOperationType
    timestamp: float
    data: Dict[str, Any]
    timeline_id: str
    track_id: Optional[int] = None
    clip_id: Optional[str] = None
    position: Optional[float] = None
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps({
            "operation_id": self.operation_id,
            "user_id": self.user_id,
            "operation_type": self.operation_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "timeline_id": self.timeline_id,
            "track_id": self.track_id,
            "clip_id": self.clip_id,
            "position": self.position
        })

@dataclass
class TimelineVersion:
    """Version of timeline state"""
    version_id: str
    timeline_id: str
    version_number: int
    parent_version: Optional[str]
    operations: List[EditOperation]
    state_snapshot: Dict[str, Any]
    created_by: str
    created_at: datetime
    message: Optional[str] = None

@dataclass
class CollaborativeSession:
    """Collaborative editing session"""
    session_id: str
    project_id: str
    timeline_id: str
    owner_id: str
    users: List[User]
    created_at: datetime
    active: bool = True
    conflict_resolution: ConflictResolution = ConflictResolution.OPERATIONAL_TRANSFORM
    max_users: int = 10

class OperationalTransform:
    """Operational Transformation for conflict-free editing"""
    
    @staticmethod
    def transform(op1: EditOperation, op2: EditOperation) -> Tuple[EditOperation, EditOperation]:
        """Transform two concurrent operations"""
        # Both operations on same clip
        if op1.clip_id and op1.clip_id == op2.clip_id:
            return OperationalTransform._transform_same_clip(op1, op2)
        
        # Operations on same track
        if op1.track_id == op2.track_id:
            return OperationalTransform._transform_same_track(op1, op2)
        
        # Independent operations
        return op1, op2
    
    @staticmethod
    def _transform_same_clip(op1: EditOperation, op2: EditOperation) -> Tuple[EditOperation, EditOperation]:
        """Transform operations on same clip"""
        type1, type2 = op1.operation_type, op2.operation_type
        
        # Both moving same clip
        if type1 == type2 == EditOperationType.MOVE_CLIP:
            # Last operation wins for position
            if op1.timestamp > op2.timestamp:
                return op1, EditOperation(
                    operation_id=op2.operation_id,
                    user_id=op2.user_id,
                    operation_type=EditOperationType.MOVE_CLIP,
                    timestamp=op2.timestamp,
                    data={},  # No-op
                    timeline_id=op2.timeline_id,
                    clip_id=op2.clip_id
                )
            else:
                return EditOperation(
                    operation_id=op1.operation_id,
                    user_id=op1.user_id,
                    operation_type=EditOperationType.MOVE_CLIP,
                    timestamp=op1.timestamp,
                    data={},  # No-op
                    timeline_id=op1.timeline_id,
                    clip_id=op1.clip_id
                ), op2
        
        # One removes, other modifies
        if type1 == EditOperationType.REMOVE_CLIP:
            # Remove wins, nullify other operation
            return op1, EditOperation(
                operation_id=op2.operation_id,
                user_id=op2.user_id,
                operation_type=op2.operation_type,
                timestamp=op2.timestamp,
                data={},  # No-op
                timeline_id=op2.timeline_id
            )
        
        if type2 == EditOperationType.REMOVE_CLIP:
            return EditOperation(
                operation_id=op1.operation_id,
                user_id=op1.user_id,
                operation_type=op1.operation_type,
                timestamp=op1.timestamp,
                data={},  # No-op
                timeline_id=op1.timeline_id
            ), op2
        
        # Default: both operations proceed
        return op1, op2
    
    @staticmethod
    def _transform_same_track(op1: EditOperation, op2: EditOperation) -> Tuple[EditOperation, EditOperation]:
        """Transform operations on same track"""
        type1, type2 = op1.operation_type, op2.operation_type
        
        # Both adding clips
        if type1 == type2 == EditOperationType.ADD_CLIP:
            pos1 = op1.position or 0
            pos2 = op2.position or 0
            
            # Adjust positions if overlapping
            if abs(pos1 - pos2) < 0.01:  # Same position
                if op1.timestamp < op2.timestamp:
                    # op2 shifts right
                    op2.position = pos2 + op1.data.get("duration", 1.0)
                else:
                    # op1 shifts right
                    op1.position = pos1 + op2.data.get("duration", 1.0)
        
        return op1, op2

class VersionControl:
    """Version control for collaborative editing"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize version control database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeline_versions (
                    version_id TEXT PRIMARY KEY,
                    timeline_id TEXT,
                    version_number INTEGER,
                    parent_version TEXT,
                    operations BLOB,
                    state_snapshot BLOB,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    message TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeline_versions 
                ON timeline_versions(timeline_id, version_number)
            """)
            
            conn.commit()
    
    def create_version(
        self,
        timeline_id: str,
        operations: List[EditOperation],
        state_snapshot: Dict[str, Any],
        user_id: str,
        message: Optional[str] = None
    ) -> TimelineVersion:
        """Create new timeline version"""
        # Get latest version number
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(version_number) FROM timeline_versions 
                WHERE timeline_id = ?
            """, (timeline_id,))
            
            result = cursor.fetchone()
            version_number = (result[0] + 1) if result[0] else 1
            
            # Get parent version
            cursor.execute("""
                SELECT version_id FROM timeline_versions 
                WHERE timeline_id = ? AND version_number = ?
            """, (timeline_id, version_number - 1))
            
            parent_result = cursor.fetchone()
            parent_version = parent_result[0] if parent_result else None
            
            # Create version
            version = TimelineVersion(
                version_id=str(uuid.uuid4()),
                timeline_id=timeline_id,
                version_number=version_number,
                parent_version=parent_version,
                operations=operations,
                state_snapshot=state_snapshot,
                created_by=user_id,
                created_at=datetime.now(),
                message=message
            )
            
            # Save to database
            cursor.execute("""
                INSERT INTO timeline_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.version_id,
                version.timeline_id,
                version.version_number,
                version.parent_version,
                pickle.dumps(version.operations),
                pickle.dumps(version.state_snapshot),
                version.created_by,
                version.created_at.isoformat(),
                version.message
            ))
            
            conn.commit()
            
        return version
    
    def get_version(self, version_id: str) -> Optional[TimelineVersion]:
        """Get specific version"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM timeline_versions WHERE version_id = ?
            """, (version_id,))
            
            row = cursor.fetchone()
            if row:
                return TimelineVersion(
                    version_id=row[0],
                    timeline_id=row[1],
                    version_number=row[2],
                    parent_version=row[3],
                    operations=pickle.loads(row[4]),
                    state_snapshot=pickle.loads(row[5]),
                    created_by=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                    message=row[8]
                )
        
        return None
    
    def get_timeline_history(self, timeline_id: str) -> List[TimelineVersion]:
        """Get version history for timeline"""
        versions = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM timeline_versions 
                WHERE timeline_id = ?
                ORDER BY version_number DESC
            """, (timeline_id,))
            
            for row in cursor.fetchall():
                versions.append(TimelineVersion(
                    version_id=row[0],
                    timeline_id=row[1],
                    version_number=row[2],
                    parent_version=row[3],
                    operations=pickle.loads(row[4]),
                    state_snapshot=pickle.loads(row[5]),
                    created_by=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                    message=row[8]
                ))
        
        return versions

class CollaborationServer:
    """WebSocket server for real-time collaboration"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.sessions: Dict[str, CollaborativeSession] = {}
        self.connections: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.user_connections: Dict[str, WebSocketServerProtocol] = {}
        self.operation_queue: asyncio.Queue = asyncio.Queue()
        self.version_control = VersionControl("/tmp/collab_versions.db")
        self.redis = None  # For distributed setup
        
    async def start(self):
        """Start collaboration server"""
        # Initialize Redis for distributed setup
        try:
            self.redis = await aioredis.create_redis_pool('redis://localhost')
        except:
            logger.warning("Redis not available, using in-memory state")
        
        # Start WebSocket server
        async with websockets.serve(self.handle_connection, self.host, self.port):
            logger.info(f"Collaboration server started on {self.host}:{self.port}")
            
            # Start operation processor
            asyncio.create_task(self.process_operations())
            
            await asyncio.Future()  # Run forever
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        user_id = None
        session_id = None
        
        try:
            # Authentication
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            user_id = auth_data.get("user_id")
            session_id = auth_data.get("session_id")
            
            if not user_id or not session_id:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Authentication required"
                }))
                return
            
            # Add to session
            if session_id not in self.connections:
                self.connections[session_id] = set()
            
            self.connections[session_id].add(websocket)
            self.user_connections[user_id] = websocket
            
            # Send session info
            if session_id in self.sessions:
                session = self.sessions[session_id]
                await websocket.send(json.dumps({
                    "type": "session_info",
                    "session": asdict(session),
                    "users": [asdict(u) for u in session.users]
                }))
            
            # Notify other users
            await self.broadcast_to_session(session_id, {
                "type": "user_joined",
                "user_id": user_id,
                "timestamp": time.time()
            }, exclude=websocket)
            
            # Handle messages
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(session_id, user_id, data, websocket)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Clean up connection
            if session_id and session_id in self.connections:
                self.connections[session_id].discard(websocket)
                
            if user_id in self.user_connections:
                del self.user_connections[user_id]
            
            # Notify disconnect
            if session_id:
                await self.broadcast_to_session(session_id, {
                    "type": "user_left",
                    "user_id": user_id,
                    "timestamp": time.time()
                })
    
    async def handle_message(
        self,
        session_id: str,
        user_id: str,
        data: Dict[str, Any],
        websocket: WebSocketServerProtocol
    ):
        """Handle incoming message"""
        message_type = data.get("type")
        
        if message_type == "operation":
            # Create edit operation
            operation = EditOperation(
                operation_id=str(uuid.uuid4()),
                user_id=user_id,
                operation_type=EditOperationType(data["operation_type"]),
                timestamp=time.time(),
                data=data.get("data", {}),
                timeline_id=data["timeline_id"],
                track_id=data.get("track_id"),
                clip_id=data.get("clip_id"),
                position=data.get("position")
            )
            
            # Queue for processing
            await self.operation_queue.put((session_id, operation))
            
        elif message_type == "cursor":
            # Broadcast cursor position
            await self.broadcast_to_session(session_id, {
                "type": "cursor_update",
                "user_id": user_id,
                "position": data["position"],
                "timestamp": time.time()
            }, exclude=websocket)
            
        elif message_type == "selection":
            # Broadcast selection change
            await self.broadcast_to_session(session_id, {
                "type": "selection_update",
                "user_id": user_id,
                "selection": data["selection"],
                "timestamp": time.time()
            }, exclude=websocket)
            
        elif message_type == "chat":
            # Broadcast chat message
            await self.broadcast_to_session(session_id, {
                "type": "chat_message",
                "user_id": user_id,
                "message": data["message"],
                "timestamp": time.time()
            })
            
        elif message_type == "save_version":
            # Create version checkpoint
            await self.create_version_checkpoint(
                session_id,
                user_id,
                data.get("message")
            )
    
    async def process_operations(self):
        """Process queued operations with OT"""
        pending_operations: Dict[str, List[EditOperation]] = {}
        
        while True:
            try:
                # Get operation from queue
                session_id, operation = await self.operation_queue.get()
                
                # Initialize pending list for session
                if session_id not in pending_operations:
                    pending_operations[session_id] = []
                
                # Transform against pending operations
                transformed_op = operation
                for pending_op in pending_operations[session_id]:
                    transformed_op, pending_op = OperationalTransform.transform(
                        transformed_op, pending_op
                    )
                
                # Add to pending
                pending_operations[session_id].append(transformed_op)
                
                # Broadcast transformed operation
                await self.broadcast_to_session(session_id, {
                    "type": "operation",
                    "operation": json.loads(transformed_op.to_json()),
                    "timestamp": time.time()
                })
                
                # Periodically clear processed operations
                if len(pending_operations[session_id]) > 100:
                    pending_operations[session_id] = pending_operations[session_id][-50:]
                    
            except Exception as e:
                logger.error(f"Error processing operation: {e}")
    
    async def broadcast_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
        exclude: Optional[WebSocketServerProtocol] = None
    ):
        """Broadcast message to all users in session"""
        if session_id not in self.connections:
            return
        
        message_json = json.dumps(message)
        
        for websocket in self.connections[session_id]:
            if websocket != exclude:
                try:
                    await websocket.send(message_json)
                except:
                    # Connection lost
                    pass
    
    async def create_version_checkpoint(
        self,
        session_id: str,
        user_id: str,
        message: Optional[str] = None
    ):
        """Create version checkpoint"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Collect recent operations
        # In production, this would come from operation history
        operations = []
        
        # Get current state snapshot
        # In production, this would be the actual timeline state
        state_snapshot = {
            "timeline_id": session.timeline_id,
            "tracks": [],
            "clips": [],
            "effects": []
        }
        
        # Create version
        version = self.version_control.create_version(
            session.timeline_id,
            operations,
            state_snapshot,
            user_id,
            message
        )
        
        # Notify users
        await self.broadcast_to_session(session_id, {
            "type": "version_created",
            "version_id": version.version_id,
            "version_number": version.version_number,
            "created_by": user_id,
            "message": message,
            "timestamp": time.time()
        })

class CollaborationClient:
    """Client for collaborative editing"""
    
    def __init__(self, server_url: str, user_id: str, session_id: str):
        self.server_url = server_url
        self.user_id = user_id
        self.session_id = session_id
        self.websocket = None
        self.connected = False
        
    async def connect(self):
        """Connect to collaboration server"""
        self.websocket = await websockets.connect(self.server_url)
        
        # Authenticate
        await self.websocket.send(json.dumps({
            "user_id": self.user_id,
            "session_id": self.session_id
        }))
        
        self.connected = True
        logger.info(f"Connected to collaboration server as {self.user_id}")
        
        # Start message handler
        asyncio.create_task(self.handle_messages())
    
    async def handle_messages(self):
        """Handle incoming messages"""
        async for message in self.websocket:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "operation":
                await self.on_operation(data["operation"])
            elif message_type == "cursor_update":
                await self.on_cursor_update(data)
            elif message_type == "selection_update":
                await self.on_selection_update(data)
            elif message_type == "user_joined":
                await self.on_user_joined(data)
            elif message_type == "user_left":
                await self.on_user_left(data)
            elif message_type == "chat_message":
                await self.on_chat_message(data)
    
    async def send_operation(self, operation: EditOperation):
        """Send edit operation to server"""
        if not self.connected:
            return
        
        await self.websocket.send(json.dumps({
            "type": "operation",
            "operation_type": operation.operation_type.value,
            "data": operation.data,
            "timeline_id": operation.timeline_id,
            "track_id": operation.track_id,
            "clip_id": operation.clip_id,
            "position": operation.position
        }))
    
    async def send_cursor_position(self, position: Dict[str, float]):
        """Send cursor position"""
        if not self.connected:
            return
        
        await self.websocket.send(json.dumps({
            "type": "cursor",
            "position": position
        }))
    
    async def send_selection(self, selection: List[str]):
        """Send selection update"""
        if not self.connected:
            return
        
        await self.websocket.send(json.dumps({
            "type": "selection",
            "selection": selection
        }))
    
    async def send_chat_message(self, message: str):
        """Send chat message"""
        if not self.connected:
            return
        
        await self.websocket.send(json.dumps({
            "type": "chat",
            "message": message
        }))
    
    # Event handlers (to be overridden)
    async def on_operation(self, operation: Dict[str, Any]):
        """Handle incoming operation"""
        pass
    
    async def on_cursor_update(self, data: Dict[str, Any]):
        """Handle cursor update"""
        pass
    
    async def on_selection_update(self, data: Dict[str, Any]):
        """Handle selection update"""
        pass
    
    async def on_user_joined(self, data: Dict[str, Any]):
        """Handle user joined"""
        pass
    
    async def on_user_left(self, data: Dict[str, Any]):
        """Handle user left"""
        pass
    
    async def on_chat_message(self, data: Dict[str, Any]):
        """Handle chat message"""
        pass
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False

# Example usage
if __name__ == "__main__":
    # Start server
    server = CollaborationServer()
    
    async def run_server():
        await server.start()
    
    # Run server
    # asyncio.run(run_server())
    
    print("✅ Collaborative editing system ready")