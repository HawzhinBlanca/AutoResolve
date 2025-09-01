#!/usr/bin/env python3
"""
Proxy Workflow Management System
Handles proxy generation, switching, and management for efficient editing
"""

import os
import json
import hashlib
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sqlite3
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ProxyResolution(Enum):
    """Proxy video resolutions"""
    QUARTER = "quarter"      # 1/4 resolution
    HALF = "half"           # 1/2 resolution  
    HD_720 = "hd720"        # 1280x720
    HD_1080 = "hd1080"      # 1920x1080
    CUSTOM = "custom"       # User-defined

class ProxyCodec(Enum):
    """Proxy video codecs"""
    H264 = "h264"
    H265 = "h265"
    PRORES_PROXY = "prores_proxy"
    PRORES_LT = "prores_lt"
    DNXHD = "dnxhd"
    DNXHR = "dnxhr"
    MJPEG = "mjpeg"

class ProxyStatus(Enum):
    """Proxy generation status"""
    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
    OUTDATED = "outdated"

@dataclass
class ProxySettings:
    """Proxy generation settings"""
    resolution: ProxyResolution = ProxyResolution.HALF
    codec: ProxyCodec = ProxyCodec.H264
    bitrate: int = 5000  # kb/s
    quality: int = 23    # CRF for H264/H265
    audio_codec: str = "aac"
    audio_bitrate: int = 128  # kb/s
    custom_width: Optional[int] = None
    custom_height: Optional[int] = None
    preserve_aspect: bool = True
    hardware_accel: bool = True
    
@dataclass
class MediaFile:
    """Original media file information"""
    file_path: str
    file_hash: str
    file_size: int
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    bitrate: int
    created_at: datetime
    modified_at: datetime
    
@dataclass
class ProxyFile:
    """Proxy file information"""
    proxy_id: str
    original_path: str
    proxy_path: str
    settings: ProxySettings
    status: ProxyStatus
    file_size: int
    generation_time: float
    created_at: datetime
    error_message: Optional[str] = None

class ProxyDatabase:
    """SQLite database for proxy management"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Media files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS media_files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT UNIQUE,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    duration REAL,
                    codec TEXT,
                    bitrate INTEGER,
                    created_at TIMESTAMP,
                    modified_at TIMESTAMP
                )
            """)
            
            # Proxy files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS proxy_files (
                    proxy_id TEXT PRIMARY KEY,
                    original_path TEXT,
                    proxy_path TEXT,
                    settings TEXT,
                    status TEXT,
                    file_size INTEGER,
                    generation_time REAL,
                    created_at TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (original_path) REFERENCES media_files(file_path)
                )
            """)
            
            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_proxy_original 
                ON proxy_files(original_path)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_proxy_status 
                ON proxy_files(status)
            """)
            
            conn.commit()
    
    def add_media_file(self, media: MediaFile) -> bool:
        """Add media file to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO media_files 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    media.file_path,
                    media.file_hash,
                    media.file_size,
                    media.width,
                    media.height,
                    media.fps,
                    media.duration,
                    media.codec,
                    media.bitrate,
                    media.created_at.isoformat(),
                    media.modified_at.isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to add media file: {e}")
            return False
    
    def add_proxy_file(self, proxy: ProxyFile) -> bool:
        """Add proxy file to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO proxy_files 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    proxy.proxy_id,
                    proxy.original_path,
                    proxy.proxy_path,
                    json.dumps(asdict(proxy.settings)),
                    proxy.status.value,
                    proxy.file_size,
                    proxy.generation_time,
                    proxy.created_at.isoformat(),
                    proxy.error_message
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to add proxy file: {e}")
            return False
    
    def get_proxy_for_media(self, media_path: str) -> Optional[ProxyFile]:
        """Get proxy file for media"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM proxy_files 
                    WHERE original_path = ? AND status = ?
                    ORDER BY created_at DESC LIMIT 1
                """, (media_path, ProxyStatus.READY.value))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_proxy(row)
                return None
        except Exception as e:
            logger.error(f"Failed to get proxy: {e}")
            return None
    
    def update_proxy_status(
        self,
        proxy_id: str,
        status: ProxyStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Update proxy status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE proxy_files 
                    SET status = ?, error_message = ?
                    WHERE proxy_id = ?
                """, (status.value, error_message, proxy_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update proxy status: {e}")
            return False
    
    def _row_to_proxy(self, row: tuple) -> ProxyFile:
        """Convert database row to ProxyFile"""
        return ProxyFile(
            proxy_id=row[0],
            original_path=row[1],
            proxy_path=row[2],
            settings=ProxySettings(**json.loads(row[3])),
            status=ProxyStatus(row[4]),
            file_size=row[5],
            generation_time=row[6],
            created_at=datetime.fromisoformat(row[7]),
            error_message=row[8]
        )

class ProxyGenerator:
    """Generate proxy media files"""
    
    def __init__(self, settings: ProxySettings):
        self.settings = settings
        
    def generate(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """Generate proxy file using FFmpeg"""
        try:
            # Get input video info
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json",
                input_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to probe video: {result.stderr}")
                return False
            
            info = json.loads(result.stdout)
            stream = info["streams"][0]
            
            orig_width = int(stream["width"])
            orig_height = int(stream["height"])
            
            # Calculate proxy dimensions
            if self.settings.resolution == ProxyResolution.QUARTER:
                width = orig_width // 4
                height = orig_height // 4
            elif self.settings.resolution == ProxyResolution.HALF:
                width = orig_width // 2
                height = orig_height // 2
            elif self.settings.resolution == ProxyResolution.HD_720:
                width = 1280
                height = 720
            elif self.settings.resolution == ProxyResolution.HD_1080:
                width = 1920
                height = 1080
            else:  # CUSTOM
                width = self.settings.custom_width or orig_width
                height = self.settings.custom_height or orig_height
            
            # Preserve aspect ratio
            if self.settings.preserve_aspect:
                aspect = orig_width / orig_height
                if width / height != aspect:
                    height = int(width / aspect)
            
            # Make dimensions even (required for many codecs)
            width = width if width % 2 == 0 else width - 1
            height = height if height % 2 == 0 else height - 1
            
            # Build FFmpeg command
            cmd = ["ffmpeg", "-i", input_path]
            
            # Hardware acceleration
            if self.settings.hardware_accel:
                # Try VideoToolbox on macOS
                if os.uname().sysname == "Darwin":
                    cmd.extend(["-hwaccel", "videotoolbox"])
            
            # Video settings
            cmd.extend(["-vf", f"scale={width}:{height}"])
            
            if self.settings.codec == ProxyCodec.H264:
                cmd.extend(["-c:v", "libx264", "-crf", str(self.settings.quality)])
                cmd.extend(["-preset", "fast"])
            elif self.settings.codec == ProxyCodec.H265:
                cmd.extend(["-c:v", "libx265", "-crf", str(self.settings.quality)])
                cmd.extend(["-preset", "fast"])
            elif self.settings.codec == ProxyCodec.PRORES_PROXY:
                cmd.extend(["-c:v", "prores_ks", "-profile:v", "0"])
            elif self.settings.codec == ProxyCodec.PRORES_LT:
                cmd.extend(["-c:v", "prores_ks", "-profile:v", "1"])
            elif self.settings.codec == ProxyCodec.DNXHD:
                cmd.extend(["-c:v", "dnxhd", "-b:v", f"{self.settings.bitrate}k"])
            elif self.settings.codec == ProxyCodec.MJPEG:
                cmd.extend(["-c:v", "mjpeg", "-q:v", "5"])
            
            # Audio settings
            cmd.extend([
                "-c:a", self.settings.audio_codec,
                "-b:a", f"{self.settings.audio_bitrate}k"
            ])
            
            # Output file
            cmd.extend(["-y", output_path])
            
            # Run FFmpeg with progress monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                
                # Parse progress
                if "time=" in line and progress_callback:
                    try:
                        time_str = line.split("time=")[1].split()[0]
                        # Convert to seconds
                        parts = time_str.split(":")
                        seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                        progress_callback(seconds)
                    except:
                        pass
            
            process.wait()
            
            if process.returncode == 0:
                logger.info(f"Proxy generated: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg failed with code {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to generate proxy: {e}")
            return False

class ProxyWorkflowManager:
    """Complete proxy workflow management system"""
    
    def __init__(self, proxy_dir: str, db_path: Optional[str] = None):
        self.proxy_dir = Path(proxy_dir)
        self.proxy_dir.mkdir(parents=True, exist_ok=True)
        
        if not db_path:
            db_path = self.proxy_dir / "proxy_database.db"
        
        self.database = ProxyDatabase(str(db_path))
        self.generator_pool = ProcessPoolExecutor(max_workers=2)
        self.active_generations: Dict[str, threading.Event] = {}
        self.proxy_mapping: Dict[str, str] = {}  # original -> proxy path
        
    def analyze_media(self, file_path: str) -> Optional[MediaFile]:
        """Analyze media file properties"""
        try:
            # Get file stats
            stat = os.stat(file_path)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Get video properties
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return None
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Get codec info via ffprobe
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,bit_rate",
                "-of", "json",
                file_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            codec = "unknown"
            bitrate = 0
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                if info.get("streams"):
                    stream = info["streams"][0]
                    codec = stream.get("codec_name", "unknown")
                    bitrate = int(stream.get("bit_rate", 0))
            
            media = MediaFile(
                file_path=file_path,
                file_hash=file_hash,
                file_size=stat.st_size,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                codec=codec,
                bitrate=bitrate,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime)
            )
            
            # Add to database
            self.database.add_media_file(media)
            
            return media
            
        except Exception as e:
            logger.error(f"Failed to analyze media: {e}")
            return None
    
    def generate_proxy(
        self,
        file_path: str,
        settings: Optional[ProxySettings] = None,
        force: bool = False
    ) -> Optional[str]:
        """Generate proxy for media file"""
        # Check if proxy already exists
        if not force:
            existing_proxy = self.database.get_proxy_for_media(file_path)
            if existing_proxy and os.path.exists(existing_proxy.proxy_path):
                logger.info(f"Using existing proxy: {existing_proxy.proxy_path}")
                self.proxy_mapping[file_path] = existing_proxy.proxy_path
                return existing_proxy.proxy_path
        
        # Analyze media if needed
        media = self.analyze_media(file_path)
        if not media:
            return None
        
        # Use default settings if not provided
        if not settings:
            settings = ProxySettings()
        
        # Generate proxy ID and path
        proxy_id = hashlib.md5(
            f"{file_path}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        proxy_filename = f"{proxy_id}_{Path(file_path).stem}_proxy.mp4"
        proxy_path = self.proxy_dir / proxy_filename
        
        # Create proxy record
        proxy = ProxyFile(
            proxy_id=proxy_id,
            original_path=file_path,
            proxy_path=str(proxy_path),
            settings=settings,
            status=ProxyStatus.PENDING,
            file_size=0,
            generation_time=0.0,
            created_at=datetime.now()
        )
        
        self.database.add_proxy_file(proxy)
        
        # Mark as generating
        self.database.update_proxy_status(proxy_id, ProxyStatus.GENERATING)
        
        # Generate proxy
        start_time = datetime.now()
        generator = ProxyGenerator(settings)
        
        success = generator.generate(
            file_path,
            str(proxy_path),
            progress_callback=lambda s: logger.debug(f"Progress: {s:.1f}s")
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        if success:
            # Update proxy record
            proxy.status = ProxyStatus.READY
            proxy.file_size = os.path.getsize(proxy_path)
            proxy.generation_time = generation_time
            
            self.database.add_proxy_file(proxy)
            self.database.update_proxy_status(proxy_id, ProxyStatus.READY)
            
            # Update mapping
            self.proxy_mapping[file_path] = str(proxy_path)
            
            logger.info(f"Proxy generated in {generation_time:.1f}s: {proxy_path}")
            return str(proxy_path)
        else:
            self.database.update_proxy_status(
                proxy_id,
                ProxyStatus.FAILED,
                "Generation failed"
            )
            return None
    
    def batch_generate(
        self,
        file_paths: List[str],
        settings: Optional[ProxySettings] = None,
        max_parallel: int = 2
    ) -> Dict[str, str]:
        """Generate proxies for multiple files"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(self.generate_proxy, path, settings): path
                for path in file_paths
            }
            
            for future in futures:
                path = futures[future]
                try:
                    proxy_path = future.result(timeout=300)  # 5 min timeout
                    if proxy_path:
                        results[path] = proxy_path
                except Exception as e:
                    logger.error(f"Failed to generate proxy for {path}: {e}")
        
        return results
    
    def switch_to_proxy(self, original_path: str) -> Optional[str]:
        """Switch to proxy version of media"""
        # Check mapping cache
        if original_path in self.proxy_mapping:
            return self.proxy_mapping[original_path]
        
        # Check database
        proxy = self.database.get_proxy_for_media(original_path)
        if proxy and os.path.exists(proxy.proxy_path):
            self.proxy_mapping[original_path] = proxy.proxy_path
            return proxy.proxy_path
        
        return None
    
    def switch_to_original(self, proxy_path: str) -> Optional[str]:
        """Switch back to original media"""
        # Reverse lookup in mapping
        for original, proxy in self.proxy_mapping.items():
            if proxy == proxy_path:
                return original
        
        # Check database
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT original_path FROM proxy_files 
                    WHERE proxy_path = ?
                """, (proxy_path,))
                
                row = cursor.fetchone()
                if row:
                    return row[0]
        except Exception as e:
            logger.error(f"Failed to find original: {e}")
        
        return None
    
    def clean_orphaned_proxies(self) -> int:
        """Remove proxy files without original media"""
        cleaned = 0
        
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM proxy_files")
                
                for row in cursor.fetchall():
                    proxy = self.database._row_to_proxy(row)
                    
                    # Check if original exists
                    if not os.path.exists(proxy.original_path):
                        # Remove proxy file
                        if os.path.exists(proxy.proxy_path):
                            os.remove(proxy.proxy_path)
                            cleaned += 1
                        
                        # Remove from database
                        cursor.execute("""
                            DELETE FROM proxy_files WHERE proxy_id = ?
                        """, (proxy.proxy_id,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to clean proxies: {e}")
        
        logger.info(f"Cleaned {cleaned} orphaned proxy files")
        return cleaned
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get proxy storage statistics"""
        stats = {
            "total_proxies": 0,
            "total_size": 0,
            "by_status": {},
            "by_codec": {},
            "average_compression": 0.0
        }
        
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                # Total proxies
                cursor.execute("SELECT COUNT(*) FROM proxy_files")
                stats["total_proxies"] = cursor.fetchone()[0]
                
                # Total size
                cursor.execute("SELECT SUM(file_size) FROM proxy_files WHERE status = ?",
                             (ProxyStatus.READY.value,))
                total_size = cursor.fetchone()[0]
                stats["total_size"] = total_size or 0
                
                # By status
                cursor.execute("""
                    SELECT status, COUNT(*) FROM proxy_files 
                    GROUP BY status
                """)
                for row in cursor.fetchall():
                    stats["by_status"][row[0]] = row[1]
                
                # Average compression ratio
                cursor.execute("""
                    SELECT 
                        AVG(CAST(p.file_size AS REAL) / m.file_size) as compression
                    FROM proxy_files p
                    JOIN media_files m ON p.original_path = m.file_path
                    WHERE p.status = ?
                """, (ProxyStatus.READY.value,))
                
                compression = cursor.fetchone()[0]
                stats["average_compression"] = compression or 0.0
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
        
        return stats
    
    def _calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Read first and last chunks for speed
            f.seek(0)
            sha256.update(f.read(chunk_size))
            
            file_size = os.path.getsize(file_path)
            if file_size > chunk_size * 2:
                f.seek(-chunk_size, 2)
                sha256.update(f.read(chunk_size))
        
        return sha256.hexdigest()
    
    def cleanup(self):
        """Clean up resources"""
        self.generator_pool.shutdown(wait=True)
        logger.info("Proxy workflow manager cleaned up")

# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = ProxyWorkflowManager("/tmp/proxy_cache")
    
    # Generate proxy
    proxy_path = manager.generate_proxy(
        "/Users/hawzhin/Videos/test_30s.mp4",
        ProxySettings(
            resolution=ProxyResolution.HALF,
            codec=ProxyCodec.H264,
            quality=23
        )
    )
    
    if proxy_path:
        print(f"✅ Proxy generated: {proxy_path}")
    
    # Get storage stats
    stats = manager.get_storage_stats()
    print(f"📊 Storage stats: {stats}")
    
    # Clean orphaned proxies
    cleaned = manager.clean_orphaned_proxies()
    print(f"🧹 Cleaned {cleaned} orphaned proxies")
    
    manager.cleanup()