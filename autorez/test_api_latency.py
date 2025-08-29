#!/usr/bin/env python3
"""
Enterprise API Latency Testing for AutoResolve v3.2
Blueprint Compliance: REQ-010 - Backend API latency under 100ms
100% Compliance Protocol - Zero Tolerance for Deviation
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Tuple
import aiohttp
import numpy as np
from pathlib import Path

class APILatencyTester:
    """Production-grade API latency testing with statistical analysis"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        self.test_video = "/Users/hawzhin/AutoResolve/autorez/assets/test_media/test_video_5min.mp4"
        
    async def measure_endpoint(self, session: aiohttp.ClientSession, 
                              method: str, endpoint: str, 
                              payload: dict = None,
                              iterations: int = 100) -> Dict:
        """Measure latency for a single endpoint"""
        latencies = []
        errors = 0
        
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                if method == "GET":
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        await response.text()
                        latency = (time.perf_counter() - start) * 1000  # Convert to ms
                        if response.status == 200:
                            latencies.append(latency)
                        else:
                            errors += 1
                elif method == "POST":
                    async with session.post(f"{self.base_url}{endpoint}", json=payload) as response:
                        await response.text()
                        latency = (time.perf_counter() - start) * 1000  # Convert to ms
                        if response.status in [200, 202]:
                            latencies.append(latency)
                        else:
                            errors += 1
            except Exception as e:
                errors += 1
                print(f"Error testing {endpoint}: {e}")
            
            await asyncio.sleep(0.01)  # Small delay between requests
        
        if latencies:
            return {
                "endpoint": endpoint,
                "method": method,
                "samples": len(latencies),
                "errors": errors,
                "min": min(latencies),
                "max": max(latencies),
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            }
        else:
            return {
                "endpoint": endpoint,
                "method": method,
                "samples": 0,
                "errors": errors,
                "status": "FAILED"
            }
    
    async def run_tests(self) -> Dict:
        """Run all latency tests"""
        async with aiohttp.ClientSession() as session:
            # Test endpoints per Blueprint requirements
            tests = [
                ("GET", "/health", None, 10),  # REQ-010a: Health < 10ms
                ("GET", "/api/timeline/list", None, 50),  # REQ-010b: Timeline list < 50ms
                ("POST", "/api/pipeline/start", {
                    "video_path": self.test_video,
                    "options": {"silence_detection": True}
                }, 100),  # REQ-010c: Start < 100ms
                ("GET", "/api/telemetry/metrics", None, 50),  # REQ-010d: Metrics < 50ms
                ("POST", "/api/export/fcpxml", {
                    "timeline_id": "test"
                }, 100),  # REQ-010e: Export < 100ms
            ]
            
            results = []
            for method, endpoint, payload, max_latency in tests:
                print(f"Testing {method} {endpoint}...")
                result = await self.measure_endpoint(session, method, endpoint, payload)
                result["max_allowed"] = max_latency
                result["passed"] = result.get("p95", float('inf')) <= max_latency
                results.append(result)
            
            return results
    
    def validate_compliance(self, results: List[Dict]) -> Tuple[bool, Dict]:
        """Validate Blueprint compliance for REQ-010"""
        all_passed = True
        violations = []
        summary = {
            "total_endpoints": len(results),
            "passed": 0,
            "failed": 0,
            "violations": []
        }
        
        for result in results:
            if result.get("status") == "FAILED":
                all_passed = False
                summary["failed"] += 1
                violations.append({
                    "endpoint": result["endpoint"],
                    "reason": "Endpoint unreachable",
                    "errors": result["errors"]
                })
            elif not result["passed"]:
                all_passed = False
                summary["failed"] += 1
                violations.append({
                    "endpoint": result["endpoint"],
                    "requirement": f"<{result['max_allowed']}ms",
                    "actual_p95": f"{result['p95']:.2f}ms",
                    "violation_ratio": result['p95'] / result['max_allowed']
                })
            else:
                summary["passed"] += 1
        
        summary["violations"] = violations
        summary["compliance"] = all_passed
        summary["compliance_rate"] = (summary["passed"] / summary["total_endpoints"]) * 100
        
        return all_passed, summary
    
    def generate_report(self, results: List[Dict], summary: Dict) -> str:
        """Generate compliance report"""
        report = []
        report.append("=" * 80)
        report.append("ENTERPRISE API LATENCY COMPLIANCE REPORT")
        report.append("Blueprint: REQ-010 - API Latency Requirements")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("COMPLIANCE SUMMARY:")
        report.append(f"  Status: {'✅ PASSED' if summary['compliance'] else '❌ FAILED'}")
        report.append(f"  Compliance Rate: {summary['compliance_rate']:.1f}%")
        report.append(f"  Endpoints Tested: {summary['total_endpoints']}")
        report.append(f"  Passed: {summary['passed']}")
        report.append(f"  Failed: {summary['failed']}")
        report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS:")
        report.append("-" * 80)
        report.append(f"{'Endpoint':<30} {'P95 Latency':<15} {'Requirement':<15} {'Status':<10}")
        report.append("-" * 80)
        
        for result in results:
            if result.get("status") == "FAILED":
                report.append(f"{result['endpoint']:<30} {'N/A':<15} {'N/A':<15} {'❌ FAILED':<10}")
            else:
                p95_str = f"{result['p95']:.2f}ms"
                req_str = f"<{result['max_allowed']}ms"
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                report.append(f"{result['endpoint']:<30} {p95_str:<15} {req_str:<15} {status:<10}")
        
        report.append("-" * 80)
        report.append("")
        
        # Violations
        if summary['violations']:
            report.append("VIOLATIONS DETECTED:")
            for v in summary['violations']:
                report.append(f"  • {v['endpoint']}: {v.get('actual_p95', 'N/A')} "
                            f"(requirement: {v.get('requirement', 'N/A')})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Run API latency tests and generate compliance report"""
    print("Starting AutoResolve API Latency Compliance Testing...")
    print("Blueprint: REQ-010 - Zero Tolerance Protocol")
    print()
    
    tester = APILatencyTester()
    
    # Check if backend is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                if response.status != 200:
                    print("❌ Backend not healthy. Start with: cd autorez && uvicorn backend_service_final:app --port 8000")
                    return
    except:
        print("❌ Backend not running. Start with: cd autorez && uvicorn backend_service_final:app --port 8000")
        return
    
    # Run tests
    results = await tester.run_tests()
    
    # Validate compliance
    passed, summary = tester.validate_compliance(results)
    
    # Generate report
    report = tester.generate_report(results, summary)
    print(report)
    
    # Save report
    report_path = Path("/Users/hawzhin/AutoResolve/autorez/artifacts/api_latency_report.json")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "results": results,
            "summary": summary,
            "compliance": passed
        }, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    # Exit with proper code
    exit(0 if passed else 1)

if __name__ == "__main__":
    asyncio.run(main())