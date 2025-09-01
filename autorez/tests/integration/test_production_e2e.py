#!/usr/bin/env python3
"""
AutoResolve v3.2 - Production E2E Integration Test Suite
Comprehensive end-to-end testing for production readiness
"""

import os
import sys
import json
import time
import requests
import asyncio
import websockets
from pathlib import Path
import logging

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

class ProductionE2ETest:
    """Production-grade end-to-end test suite"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.api_key = "test"
        self.test_video = "assets/test_30min.mp4"
        
    def test_system_startup(self):
        """Test system startup and health"""
        print("🚀 Testing System Startup...")
        
        # Check health endpoint
        response = requests.get(f"{self.base_url}/health", timeout=10)
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "memory_mb" in health_data
        
        print("  ✅ Health check passed")
        
    def test_authentication_flow(self):
        """Test complete authentication workflow"""
        print("🔐 Testing Authentication...")
        
        # Test API key auth
        response = requests.get(
            f"{self.base_url}/api/projects",
            headers={"x-api-key": self.api_key},
            timeout=5
        )
        assert response.status_code == 200
        
        print("  ✅ Authentication working")
    
    def test_websocket_connection(self):
        """Test WebSocket real-time communication"""
        print("🔌 Testing WebSocket Connection...")
        
        async def websocket_test():
            ws_url = self.base_url.replace('http', 'ws') + '/ws/status'
            
            async with websockets.connect(ws_url) as websocket:
                # Should receive connection confirmation
                message = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(message)
                assert data["type"] == "connected"
                
                return True
        
        result = asyncio.run(websocket_test())
        assert result
        print("  ✅ WebSocket communication working")
    
    def test_rate_limiting(self):
        """Test API rate limiting"""
        print("🚦 Testing Rate Limiting...")
        
        rate_limited_count = 0
        
        for i in range(15):
            response = requests.post(
                f"{self.base_url}/api/pipeline/start",
                headers={"x-api-key": self.api_key},
                json={"video_path": "test.mp4"},
                timeout=2
            )
            
            if response.status_code == 429:
                rate_limited_count += 1
            
            
        
        assert rate_limited_count >= 3, f"Expected rate limiting, got {rate_limited_count} blocks"
        print(f"  ✅ Rate limiting active ({rate_limited_count} blocks)")
    
    def test_silence_detection_module(self):
        """Test silence detection in isolation"""
        print("🔇 Testing Silence Detection...")
        
        if not os.path.exists(self.test_video):
            print("  ⚠️  Skipping - test video not found")
            return
        
        from src.ops.silence import SilenceRemover
        
        start_time = time.time()
        silence_remover = SilenceRemover()
        cuts_data, metrics = silence_remover.remove_silence(self.test_video)
        duration = time.time() - start_time
        
        # Verify structure
        assert isinstance(cuts_data, dict)
        assert "keep_windows" in cuts_data
        
        print(f"  ✅ Silence detection: {len(cuts_data['keep_windows'])} segments in {duration:.2f}s")
    
    def test_performance_gates(self):
        """Test performance gates compliance"""
        print("📊 Testing Performance Gates...")
        
        from src.eval.gates import verify_gates
        
        # Load current metrics
        with open('artifacts/metrics.json') as f:
            metrics = json.load(f)
        
        # Should pass without exceptions
        result = verify_gates(metrics)
        assert result == True
        
        print("  ✅ All performance gates pass")
    
    def run_all_tests(self):
        """Run complete production test suite"""
        print("🧪 AutoResolve v3.2 - Production E2E Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_system_startup,
            self.test_authentication_flow,
            self.test_websocket_connection,
            self.test_rate_limiting,
            self.test_silence_detection_module,
            self.test_performance_gates
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                test_method()
                passed += 1
            except Exception as e:
                print(f"  ❌ {test_method.__name__} FAILED: {e}")
                failed += 1
        
        print("=" * 60)
        print(f"📋 Production Test Results:")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Success Rate: {(passed/(passed+failed)*100):.1f}%")
        
        if failed == 0:
            print("🎉 ALL PRODUCTION TESTS PASSED")
            return True
        else:
            print("❌ PRODUCTION TESTS FAILED")
            return False

def main():
    """Run production tests from command line"""
    test_suite = ProductionE2ETest()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()