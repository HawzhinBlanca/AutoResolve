#!/usr/bin/env python3
"""Test DaVinci Resolve integration with fallback strategies"""

import os
import sys
import json
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResolveIntegrationTest:
    """Test DaVinci Resolve integration and fallback strategies"""
    
    def __init__(self):
        self.video_path = "assets/test_30min.mp4"
        self.resolve_available = False
        self.api_available = False
        
    def check_resolve_installation(self) -> Dict:
        """Check if DaVinci Resolve is installed"""
        logger.info("Checking DaVinci Resolve installation...")
        
        resolve_paths = [
            "/Applications/DaVinci Resolve/DaVinci Resolve.app",
            "/Applications/DaVinci Resolve Studio/DaVinci Resolve.app",
            "/Applications/DaVinci Resolve/Resolve.app"
        ]
        
        for path in resolve_paths:
            if Path(path).exists():
                logger.info(f"  ✅ Found: {path}")
                self.resolve_available = True
                return {
                    "installed": True,
                    "path": path
                }
        
        logger.warning("  ❌ DaVinci Resolve not found")
        return {
            "installed": False,
            "path": None
        }

    def launch_resolve(self):
        """Attempt to launch DaVinci Resolve"""
        if self.resolve_available:
            try:
                from src.ops.resolve_api import ResolveAPI
                resolve_api = ResolveAPI()
                if not resolve_api._is_resolve_running():
                    logger.info("Attempting to launch DaVinci Resolve...")
                    resolve_api._start_resolve()
                    time.sleep(15) # Wait for Resolve to start
                    return resolve_api._is_resolve_running()
                return True
            except Exception as e:
                logger.error(f"Failed to launch Resolve: {e}")
                return False
        return False
    
    def check_scripting_api(self) -> Dict:
        """Check if Resolve scripting API is available"""
        logger.info("Checking DaVinci Resolve scripting API...")
        
        # Try different methods to access the API
        methods_tried = []
        
        # Method 1: Direct import
        try:
            import DaVinciResolveScript as dvr
            resolve = dvr.scriptapp("Resolve")
            if resolve:
                logger.info("  ✅ API available via direct import")
                self.api_available = True
                return {
                    "available": True,
                    "method": "direct_import",
                    "resolve_object": str(resolve)
                }
        except ImportError:
            methods_tried.append("direct_import")
        
        # Method 2: Add Resolve Python path
        resolve_py_paths = [
            "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules",
            "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/Modules",
            os.path.expanduser("~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Scripting/Modules")
        ]
        
        for py_path in resolve_py_paths:
            if Path(py_path).exists():
                sys.path.insert(0, py_path)
                try:
                    import DaVinciResolveScript as dvr
                    resolve = dvr.scriptapp("Resolve")
                    if resolve:
                        logger.info(f"  ✅ API available via path: {py_path}")
                        self.api_available = True
                        return {
                            "available": True,
                            "method": "path_import",
                            "path": py_path,
                            "resolve_object": str(resolve)
                        }
                except:
                    pass
        
        methods_tried.append("path_import")
        
        logger.warning(f"  ❌ API not available (tried: {methods_tried})")
        return {
            "available": False,
            "methods_tried": methods_tried
        }
    
    def test_edl_generation(self) -> Dict:
        """Test EDL generation"""
        logger.info("Testing EDL generation...")
        
        from src.ops.edl import generate_edl
        
        # Create test cuts data
        cuts_data = {
            "keep": [
                {"t0": 0, "t1": 60},
                {"t0": 120, "t1": 180},
                {"t0": 300, "t1": 360},
                {"t0": 600, "t1": 660},
                {"t0": 900, "t1": 960}
            ]
        }
        
        output_path = "artifacts/test_timeline.edl"
        
        try:
            # Generate EDL
            result = generate_edl(
                cuts=cuts_data,
                fps=30,
                video_path=self.video_path,
                output_path=output_path
            )
            
            # Validate EDL file
            if Path(output_path).exists():
                with open(output_path) as f:
                    edl_content = f.read()
                
                # Check for valid EDL markers
                has_title = "TITLE:" in edl_content
                has_fcm = "FCM:" in edl_content
                has_events = "001" in edl_content  # At least one event
                
                valid = has_title and has_fcm and has_events
                
                logger.info(f"  ✅ EDL generated: {output_path}")
                logger.info(f"  Events: {result.get('events', 0)}")
                logger.info(f"  Valid format: {valid}")
                
                return {
                    "success": True,
                    "path": output_path,
                    "events": result.get("events", 0),
                    "valid_format": valid
                }
            else:
                logger.error("  ❌ EDL file not created")
                return {"success": False, "error": "File not created"}
                
        except Exception as e:
            logger.error(f"  ❌ EDL generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_fcxml_generation(self) -> Dict:
        """Test FCXML generation"""
        logger.info("Testing FCXML generation...")
        
        from src.ops.edl import generate_fcxml
        
        # Create test cuts data
        cuts_data = {
            "keep": [
                {"t0": 0, "t1": 60},
                {"t0": 120, "t1": 180},
                {"t0": 300, "t1": 360}
            ]
        }
        
        output_path = "artifacts/test_timeline.fcpxml"
        
        try:
            # Generate FCXML
            result = generate_fcxml(
                video_path=self.video_path,
                cuts=cuts_data,
                output_path=output_path,
                fps=30
            )
            
            # Validate FCXML file
            if Path(output_path).exists():
                with open(output_path) as f:
                    fcxml_content = f.read()
                
                # Check for valid FCXML markers
                has_fcpxml = "<fcpxml" in fcxml_content
                has_resources = "<resources>" in fcxml_content
                has_library = "<library>" in fcxml_content
                has_clips = "<clip" in fcxml_content
                
                valid = has_fcpxml and has_resources and has_library and has_clips
                
                logger.info(f"  ✅ FCXML generated: {output_path}")
                logger.info(f"  Clips: {result.get('clips', 0)}")
                logger.info(f"  Valid format: {valid}")
                
                return {
                    "success": True,
                    "path": output_path,
                    "clips": result.get("clips", 0),
                    "valid_format": valid
                }
            else:
                logger.error("  ❌ FCXML file not created")
                return {"success": False, "error": "File not created"}
                
        except Exception as e:
            logger.error(f"  ❌ FCXML generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_resolve_import(self, file_path: str) -> Dict:
        """Test importing EDL/FCXML into Resolve (if available)"""
        logger.info(f"Testing Resolve import of {file_path}...")
        
        if not self.resolve_available:
            logger.warning("  ⚠️ Resolve not available, skipping import test")
            return {"skipped": True, "reason": "Resolve not installed"}
        
        if not self.launch_resolve():
            logger.warning("  ⚠️ Failed to launch Resolve, skipping import test")
            return {"skipped": True, "reason": "Failed to launch Resolve"}

        # If API is available, try to import
        if self.api_available:
            try:
                from src.ops.resolve_api import ResolveAPI
                resolve_api = ResolveAPI()
                script_content = f"""import DaVinciResolveScript as dvr_script
resolve = dvr_script.scriptapp('Resolve')
project = resolve.GetProjectManager().GetCurrentProject()
media_pool = project.GetMediaPool()
media_pool.ImportTimelineFromFile('{file_path}')
"""
                script_path = "/tmp/resolve_import_script.py"
                with open(script_path, 'w') as f:
                    f.write(script_content)

                success = resolve_api._execute_resolve_script(script_path)
                os.remove(script_path)

                if success:
                    logger.info(f"  ✅ Successfully imported {file_path}")
                    return {"success": True, "method": "api"}
                else:
                    logger.warning(f"  ⚠️ Import failed via API")
                    return {"success": False, "method": "api", "error": "Import failed"}
            except Exception as e:
                logger.warning(f"  ⚠️ API import error: {e}")
        
        # Fallback: Check if file can be opened manually
        logger.info("  ℹ️ Manual import required:")
        logger.info(f"    1. Open DaVinci Resolve")
        logger.info(f"    2. File → Import → Timeline")
        logger.info(f"    3. Select: {file_path}")
        
        return {
            "success": False,
            "method": "manual",
            "instructions": "Manual import required"
        }
    
    def run_full_test(self) -> Dict:
        """Run complete Resolve integration test"""
        logger.info("="*50)
        logger.info("DAVINCI RESOLVE INTEGRATION TEST")
        logger.info("="*50)
        
        results = {}
        
        # Test 1: Check installation
        logger.info("\nTest 1: Installation Check")
        results["installation"] = self.check_resolve_installation()
        
        # Test 2: Check API
        logger.info("\nTest 2: Scripting API Check")
        results["api"] = self.check_scripting_api()
        
        # Test 3: EDL generation
        logger.info("\nTest 3: EDL Generation")
        results["edl"] = self.test_edl_generation()
        
        # Test 4: FCXML generation
        logger.info("\nTest 4: FCXML Generation")
        results["fcxml"] = self.test_fcxml_generation()
        
        # Test 5: Import tests (if possible)
        if results["edl"].get("success"):
            logger.info("\nTest 5: EDL Import")
            results["edl_import"] = self.test_resolve_import(results["edl"]["path"])
        
        if results["fcxml"].get("success"):
            logger.info("\nTest 6: FCXML Import")
            results["fcxml_import"] = self.test_resolve_import(results["fcxml"]["path"])
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("="*50)
        
        # Determine strategy
        if self.api_available:
            strategy = "scripting"
            logger.info("✅ PRIMARY: Resolve Scripting API available")
        elif results["fcxml"].get("success"):
            strategy = "fcxml"
            logger.info("✅ FALLBACK: FCXML export working")
        elif results["edl"].get("success"):
            strategy = "edl"
            logger.info("✅ FALLBACK: EDL export working")
        else:
            strategy = "none"
            logger.error("❌ No working integration method")
        
        results["strategy"] = strategy
        results["compliance"] = strategy != "none"
        
        # Save results
        Path("artifacts").mkdir(exist_ok=True)
        with open("artifacts/resolve_integration_test.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to artifacts/resolve_integration_test.json")
        logger.info(f"Recommended strategy: {strategy.upper()}")
        
        return results

def main():
    tester = ResolveIntegrationTest()
    results = tester.run_full_test()
    return 0 if results.get("compliance", False) else 1

if __name__ == "__main__":
    exit(main())