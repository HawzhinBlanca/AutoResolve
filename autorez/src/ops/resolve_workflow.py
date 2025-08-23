#!/usr/bin/env python3
"""
Complete DaVinci Resolve workflow with API and fallback strategies
Production-ready integration for AutoResolve v3.0
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Resolve API
RESOLVE_API_AVAILABLE = False
resolve_instance = None

# Add Resolve Python paths
resolve_paths = [
    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules",
    "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/Modules",
    os.path.expanduser("~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Scripting/Modules")
]

for path in resolve_paths:
    if Path(path).exists():
        sys.path.insert(0, path)
        break

try:
    import DaVinciResolveScript as dvr
    RESOLVE_API_AVAILABLE = True
    logger.info("✅ DaVinci Resolve API available")
except ImportError:
    logger.warning("⚠️ DaVinci Resolve API not available - using fallback")

class ResolveWorkflow:
    """
    Complete Resolve workflow with automatic fallback.
    Tries API first, then FCXML, then EDL.
    """
    
    def __init__(self):
        self.resolve = None
        self.project = None
        self.timeline = None
        self.method = "none"
        
        if RESOLVE_API_AVAILABLE:
            self._init_resolve_api()
    
    def _init_resolve_api(self):
        """Initialize Resolve API connection"""
        try:
            self.resolve = dvr.scriptapp("Resolve")
            if self.resolve:
                self.method = "api"
                logger.info("Connected to Resolve via API")
        except Exception as e:
            logger.warning(f"Could not connect to Resolve: {e}")
    
    def create_project(self, project_name: str) -> bool:
        """Create or open a Resolve project"""
        if not self.resolve:
            logger.warning("Resolve API not available")
            return False
        
        try:
            pm = self.resolve.GetProjectManager()
            
            # Try to open existing project
            self.project = pm.LoadProject(project_name)
            
            if not self.project:
                # Create new project
                self.project = pm.CreateProject(project_name)
            
            if self.project:
                logger.info(f"Project '{project_name}' ready")
                return True
            else:
                logger.error(f"Could not create/open project '{project_name}'")
                return False
                
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            return False
    
    def import_media(self, video_path: str) -> bool:
        """Import media into project"""
        if not self.project:
            logger.warning("No active project")
            return False
        
        try:
            media_pool = self.project.GetMediaPool()
            
            # Import media
            media_items = media_pool.ImportMedia([video_path])
            
            if media_items:
                logger.info(f"Imported media: {video_path}")
                return True
            else:
                logger.error(f"Failed to import media: {video_path}")
                return False
                
        except Exception as e:
            logger.error(f"Media import failed: {e}")
            return False
    
    def create_timeline_from_cuts(self, 
                                 timeline_name: str,
                                 video_path: str,
                                 cuts: Dict) -> bool:
        """Create timeline from cuts data"""
        
        # Try API method first
        if self.resolve and self.project:
            success = self._create_timeline_api(timeline_name, video_path, cuts)
            if success:
                self.method = "api"
                return True
        
        # Fallback to FCXML
        success = self._create_timeline_fcxml(timeline_name, video_path, cuts)
        if success:
            self.method = "fcxml"
            return True
        
        # Fallback to EDL
        success = self._create_timeline_edl(timeline_name, video_path, cuts)
        if success:
            self.method = "edl"
            return True
        
        logger.error("All timeline creation methods failed")
        return False
    
    def _create_timeline_api(self, name: str, video_path: str, cuts: Dict) -> bool:
        """Create timeline using Resolve API"""
        try:
            media_pool = self.project.GetMediaPool()
            
            # Create empty timeline
            self.timeline = media_pool.CreateEmptyTimeline(name)
            
            if not self.timeline:
                logger.error("Failed to create timeline")
                return False
            
            # Import media if not already imported
            media_items = media_pool.GetRootFolder().GetClipList()
            
            if not media_items:
                self.import_media(video_path)
                media_items = media_pool.GetRootFolder().GetClipList()
            
            if not media_items:
                logger.error("No media items available")
                return False
            
            # Add clips to timeline based on cuts
            for region in cuts.get("keep", []):
                # Set in/out points
                media_item = media_items[0]
                media_item.SetClipProperty("Start", str(region["t0"]))
                media_item.SetClipProperty("End", str(region["t1"]))
                
                # Add to timeline
                media_pool.AppendToTimeline([media_item])
            
            logger.info(f"Created timeline '{name}' via API")
            return True
            
        except Exception as e:
            logger.warning(f"API timeline creation failed: {e}")
            return False
    
    def _create_timeline_fcxml(self, name: str, video_path: str, cuts: Dict) -> bool:
        """Create timeline using FCXML export"""
        try:
            from src.ops.edl import generate_fcxml
            
            output_path = f"artifacts/{name}.fcpxml"
            
            result = generate_fcxml(
                video_path=video_path,
                cuts=cuts,
                output_path=output_path,
                fps=30
            )
            
            if result.get("success"):
                logger.info(f"Created FCXML timeline: {output_path}")
                logger.info("Import instructions:")
                logger.info("  1. In Resolve: File → Import → Timeline")
                logger.info(f"  2. Select: {output_path}")
                logger.info("  3. Choose import settings and click Import")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"FCXML generation failed: {e}")
            return False
    
    def _create_timeline_edl(self, name: str, video_path: str, cuts: Dict) -> bool:
        """Create timeline using EDL export"""
        try:
            from src.ops.edl import generate_edl
            
            output_path = f"artifacts/{name}.edl"
            
            result = generate_edl(
                cuts=cuts,
                fps=30,
                video_path=video_path,
                output_path=output_path
            )
            
            if result.get("success"):
                logger.info(f"Created EDL timeline: {output_path}")
                logger.info("Import instructions:")
                logger.info("  1. In Resolve: File → Import → Timeline → Import EDL")
                logger.info(f"  2. Select: {output_path}")
                logger.info("  3. Set frame rate to 30fps")
                logger.info("  4. Click Import")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"EDL generation failed: {e}")
            return False
    
    def export_timeline(self, format: str = "mov", preset: str = "H.264") -> str:
        """Export timeline to video file"""
        if not self.timeline:
            logger.warning("No active timeline")
            return None
        
        try:
            project = self.project
            
            # Set render settings
            project.SetRenderSettings({
                "SelectAllFrames": True,
                "TargetDir": str(Path("artifacts").absolute()),
                "CustomName": f"export_{int(time.time())}",
                "FormatWidth": 1920,
                "FormatHeight": 1080,
                "FrameRate": 30,
                "VideoQuality": 0,  # Automatic
                "AudioCodec": "aac",
                "VideoCodec": "H.264" if preset == "H.264" else "ProRes",
                "FileFormat": format
            })
            
            # Add job to render queue
            project.AddRenderJob()
            
            # Start render
            project.StartRendering()
            
            logger.info(f"Rendering started...")
            
            # Wait for render to complete
            while project.IsRenderingInProgress():
                time.sleep(1)
            
            logger.info("Rendering complete")
            return "artifacts/export_*.mov"
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    def get_status(self) -> Dict:
        """Get current workflow status"""
        return {
            "method": self.method,
            "api_available": RESOLVE_API_AVAILABLE,
            "resolve_connected": self.resolve is not None,
            "project_active": self.project is not None,
            "timeline_active": self.timeline is not None
        }

class ResolveExporter:
    """Simplified exporter for headless operation"""
    
    @staticmethod
    def export_for_resolve(video_path: str, 
                          cuts: Dict,
                          broll: Optional[Dict] = None,
                          format: str = "auto") -> Dict:
        """
        Export timeline for Resolve import.
        Auto-selects best format based on availability.
        """
        
        results = {
            "video": video_path,
            "exports": []
        }
        
        # Determine best format
        if format == "auto":
            if RESOLVE_API_AVAILABLE:
                format = "api"
            else:
                format = "fcxml"  # Prefer FCXML over EDL
        
        # Generate exports
        if format in ["fcxml", "all"]:
            from src.ops.edl import generate_fcxml
            
            fcxml_path = "artifacts/timeline.fcpxml"
            result = generate_fcxml(video_path, cuts, fcxml_path, 30)
            
            if result.get("success"):
                results["exports"].append({
                    "format": "fcxml",
                    "path": fcxml_path,
                    "clips": result.get("clips", 0)
                })
        
        if format in ["edl", "all"]:
            from src.ops.edl import generate_edl
            
            edl_path = "artifacts/timeline.edl"
            result = generate_edl(cuts, 30, video_path, edl_path)
            
            if result.get("success"):
                results["exports"].append({
                    "format": "edl",
                    "path": edl_path,
                    "events": result.get("events", 0)
                })
        
        # Add import instructions
        if results["exports"]:
            results["instructions"] = get_import_instructions(results["exports"][0]["format"])
            results["success"] = True
        else:
            results["success"] = False
            results["error"] = "No successful exports"
        
        return results

def get_import_instructions(format: str) -> str:
    """Get import instructions for format"""
    instructions = {
        "fcxml": """
FCXML Import Instructions:
1. Open DaVinci Resolve
2. Create or open a project
3. File → Import → Timeline
4. Select the .fcpxml file
5. Choose import settings
6. Click Import
""",
        "edl": """
EDL Import Instructions:
1. Open DaVinci Resolve
2. Create or open a project
3. Import source media first
4. File → Import → Timeline → Import EDL
5. Select the .edl file
6. Set frame rate to match (30fps)
7. Link to source media
8. Click Import
""",
        "api": """
API Import:
Timeline created directly in Resolve.
Check the Media Pool for the new timeline.
"""
    }
    
    return instructions.get(format, "Unknown format")

def test_workflow():
    """Test the complete workflow"""
    logger.info("Testing Resolve Workflow...")
    
    workflow = ResolveWorkflow()
    
    # Test data
    video_path = "assets/test_30min.mp4"
    cuts = {
        "keep": [
            {"t0": 0, "t1": 60},
            {"t0": 300, "t1": 360},
            {"t0": 600, "t1": 660}
        ]
    }
    
    # Try to create project
    if workflow.create_project("AutoResolve_Test"):
        logger.info("✅ Project created/opened")
        
        # Try to create timeline
        if workflow.create_timeline_from_cuts("Test_Timeline", video_path, cuts):
            logger.info(f"✅ Timeline created via {workflow.method}")
        else:
            logger.warning("⚠️ Timeline creation failed")
    else:
        # Fallback to export-only mode
        logger.info("Using export-only mode...")
        
        result = ResolveExporter.export_for_resolve(video_path, cuts)
        
        if result["success"]:
            logger.info(f"✅ Exported {len(result['exports'])} format(s)")
            for export in result["exports"]:
                logger.info(f"  - {export['format']}: {export['path']}")
            logger.info(result["instructions"])
        else:
            logger.error("❌ Export failed")
    
    # Get status
    status = workflow.get_status()
    logger.info(f"Status: {json.dumps(status, indent=2)}")
    
    return workflow.method != "none" or result.get("success", False)

if __name__ == "__main__":
    success = test_workflow()
    exit(0 if success else 1)