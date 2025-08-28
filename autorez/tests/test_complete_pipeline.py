#!/usr/bin/env python3
"""
AutoResolve V3.0 - Comprehensive Test Suite
Tests all major components end-to-end
"""

import pytest
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
import requests
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"
TEST_VIDEO = "/Users/hawzhin/Videos/test_video.mp4"
EXPORT_DIR = "/Users/hawzhin/AutoResolve/exports"

class TestPipeline:
    """Test complete video processing pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.base_url = BASE_URL
        self.test_video = TEST_VIDEO
        self.session = requests.Session()
        yield
        self.session.close()
    
    def test_health_check(self):
        """Test backend health"""
        response = self.session.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["pipeline"] == "ready"
    
    def test_complete_pipeline(self):
        """Test full pipeline processing"""
        # Start pipeline
        response = self.session.post(
            f"{self.base_url}/api/pipeline/start",
            json={"video_path": self.test_video}
        )
        assert response.status_code == 200
        task_id = response.json()["task_id"]
        assert task_id is not None
        
        # Wait for completion
        max_wait = 60  # seconds
        start = time.time()
        while time.time() - start < max_wait:
            status_response = self.session.get(
                f"{self.base_url}/api/pipeline/status/{task_id}"
            )
            status = status_response.json()
            if status["status"] == "completed":
                break
            elif status["status"] == "failed":
                pytest.fail(f"Pipeline failed: {status.get('error')}")
            time.sleep(1)
        
        # Verify results
        assert status["status"] == "completed"
        assert status["progress"] == 1.0
        assert "result" in status
        assert status["result"]["timeline_clips"] > 0
        assert status["result"]["performance"]["realtime_factor"] > 1
    
    def test_silence_detection(self):
        """Test silence detection module"""
        response = self.session.post(
            f"{self.base_url}/api/silence/detect",
            json={"video_path": self.test_video}
        )
        assert response.status_code == 200
        data = response.json()
        assert "keep_windows" in data
        assert "silence_regions" in data
        assert len(data["keep_windows"]) > 0
        assert data["total_silence"] >= 0
    
    def test_mp4_export(self):
        """Test MP4 export functionality"""
        # First run pipeline to get task_id
        pipeline_response = self.session.post(
            f"{self.base_url}/api/pipeline/start",
            json={"video_path": self.test_video}
        )
        task_id = pipeline_response.json()["task_id"]
        
        # Wait for completion
        time.sleep(10)
        
        # Export MP4
        export_response = self.session.post(
            f"{self.base_url}/api/export/mp4",
            json={
                "task_id": task_id,
                "resolution": "1280x720",
                "fps": 30,
                "preset": "fast"
            }
        )
        assert export_response.status_code == 200
        export_data = export_response.json()
        assert export_data["status"] == "success"
        assert "output_path" in export_data
        assert os.path.exists(export_data["output_path"])
        assert export_data["output_size"] > 0
    
    def test_timeline_persistence(self):
        """Test timeline save/load"""
        project_name = f"test_{int(time.time())}"
        clips = [
            {"id": "1", "name": "Test Clip 1", "start": 0, "duration": 5},
            {"id": "2", "name": "Test Clip 2", "start": 5, "duration": 3}
        ]
        
        # Save timeline
        save_response = self.session.post(
            f"{self.base_url}/api/timeline/save",
            json={"project_name": project_name, "clips": clips}
        )
        assert save_response.status_code == 200
        assert save_response.json()["status"] == "success"
        
        # Load timeline
        load_response = self.session.post(
            f"{self.base_url}/api/timeline/load",
            json={"project_name": project_name}
        )
        assert load_response.status_code == 200
        loaded_data = load_response.json()
        assert len(loaded_data["timeline"]["clips"]) == 2
        assert loaded_data["timeline"]["clips"][0]["name"] == "Test Clip 1"


class TestTimelineTools:
    """Test timeline editing tools"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.project_id = "test_project"
        yield
        self.session.close()
    
    def test_trim_tool(self):
        """Test trim functionality"""
        response = self.session.post(
            f"{self.base_url}/api/timeline/trim?project_id={self.project_id}",
            json={
                "clip_id": "test_clip",
                "trim_type": "start",
                "new_time": 2.5
            }
        )
        assert response.status_code == 200
        assert response.json()["status"] == "trimmed"
    
    def test_slip_tool(self):
        """Test slip edit"""
        response = self.session.post(
            f"{self.base_url}/api/timeline/slip?project_id={self.project_id}",
            json={
                "clip_id": "test_clip",
                "offset": 1.5
            }
        )
        assert response.status_code == 200
        assert response.json()["status"] == "slipped"
    
    def test_blade_tool(self):
        """Test blade/cut tool"""
        response = self.session.post(
            f"{self.base_url}/api/timeline/blade?project_id={self.project_id}",
            json={
                "clip_id": "test_clip",
                "cut_time": 5.0
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cut"
        assert "new_clip_id" in data


class TestPerformance:
    """Performance benchmarks"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.base_url = BASE_URL
        self.test_video = TEST_VIDEO
        self.session = requests.Session()
        yield
        self.session.close()
    
    def test_processing_speed(self):
        """Test pipeline processing speed"""
        start_time = time.time()
        
        # Start pipeline
        response = self.session.post(
            f"{self.base_url}/api/pipeline/start",
            json={"video_path": self.test_video}
        )
        task_id = response.json()["task_id"]
        
        # Wait for completion
        while True:
            status = self.session.get(
                f"{self.base_url}/api/pipeline/status/{task_id}"
            ).json()
            if status["status"] in ["completed", "failed"]:
                break
            time.sleep(0.5)
        
        processing_time = time.time() - start_time
        
        # Check performance metrics
        assert status["status"] == "completed"
        rtf = status["result"]["performance"]["realtime_factor"]
        assert rtf > 30  # Should be at least 30x realtime
        assert processing_time < 30  # Should complete within 30 seconds
        
        print(f"Processing speed: {rtf:.1f}x realtime")
        print(f"Total time: {processing_time:.2f}s")
    
    def test_memory_usage(self):
        """Test memory consumption"""
        # Get initial memory
        health = self.session.get(f"{self.base_url}/health").json()
        initial_memory = health["memory_mb"]
        
        # Process video
        response = self.session.post(
            f"{self.base_url}/api/pipeline/start",
            json={"video_path": self.test_video}
        )
        task_id = response.json()["task_id"]
        
        # Monitor memory during processing
        peak_memory = initial_memory
        while True:
            status = self.session.get(
                f"{self.base_url}/api/pipeline/status/{task_id}"
            ).json()
            health = self.session.get(f"{self.base_url}/health").json()
            peak_memory = max(peak_memory, health["memory_mb"])
            
            if status["status"] in ["completed", "failed"]:
                break
            time.sleep(0.5)
        
        # Check memory bounds
        memory_increase = peak_memory - initial_memory
        assert memory_increase < 2000  # Should not exceed 2GB increase
        assert peak_memory < 4000  # Total should stay under 4GB
        
        print(f"Peak memory: {peak_memory}MB")
        print(f"Memory increase: {memory_increase}MB")


class TestExports:
    """Test export formats"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.base_url = BASE_URL
        self.session = requests.Session()
        yield
        self.session.close()
    
    def test_fcpxml_export(self):
        """Test FCPXML export"""
        # Need a completed task first
        pipeline = self.session.post(
            f"{self.base_url}/api/pipeline/start",
            json={"video_path": TEST_VIDEO}
        )
        task_id = pipeline.json()["task_id"]
        time.sleep(10)  # Wait for completion
        
        response = self.session.post(
            f"{self.base_url}/api/export/fcpxml",
            json={"task_id": task_id}
        )
        assert response.status_code == 200
        assert response.json()["format"] == "fcpxml"
    
    def test_edl_export(self):
        """Test EDL export"""
        # Need a completed task first
        pipeline = self.session.post(
            f"{self.base_url}/api/pipeline/start",
            json={"video_path": TEST_VIDEO}
        )
        task_id = pipeline.json()["task_id"]
        time.sleep(10)  # Wait for completion
        
        response = self.session.post(
            f"{self.base_url}/api/export/edl",
            json={"task_id": task_id}
        )
        assert response.status_code == 200
        assert response.json()["format"] == "edl"


def run_all_tests():
    """Run all tests and generate report"""
    import subprocess
    
    # Run pytest with coverage
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Simple test runner for manual execution
    print("Running AutoResolve Test Suite...")
    print("=" * 60)
    
    # Basic connectivity test
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("❌ Backend returned non-200 status")
    except:
        print("❌ Backend is not running. Start it first!")
        exit(1)
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")