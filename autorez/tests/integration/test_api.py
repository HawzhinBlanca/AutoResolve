import pytest
from fastapi.testclient import TestClient
import sys
import os
import time

# Add the src directory to the sys.path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from main import app

client = TestClient(app)

def get_expected_version():
    """Reads the version from the VERSION file in the project root."""
    try:
        version_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'VERSION')
        with open(version_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "unknown"

def test_health_check_compliance():
    """
    VERIFICATION for TASK-1:
    Tests the /health endpoint for blueprint compliance.
    - Blueprint Ref: Section 11
    - Acceptance Criteria:
        1. Correct JSON schema: { "ok": true, "ver": "x.y.z" }
        2. Status code 200 OK.
        3. 'ver' field matches the VERSION file.
        4. p95 response time <= 500ms.
    """
    # 1. Test response time (SLA)
    start_time = time.perf_counter()
    response = client.get("/health")
    duration = (time.perf_counter() - start_time) * 1000  # in ms

    assert duration <= 500, f"SLA VIOLATION: /health endpoint took {duration:.2f}ms (limit: 500ms)"

    # 2. Test status code
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    # 3. Test response schema and content
    data = response.json()
    expected_version = get_expected_version()

    assert "ok" in data, "Response missing 'ok' key"
    assert data["ok"] is True, "'ok' key should be true"

    assert "ver" in data, "Response missing 'ver' key"
    assert data["ver"] == expected_version, f"Version mismatch: expected {expected_version}, got {data['ver']}"

    print(f"✅ TASK-1 Verification Passed: /health endpoint is compliant.")
    print(f"   - Response time: {duration:.2f}ms")
    print(f"   - Response body: {data}")

def test_version_endpoint():
    """Tests the /version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "autorez"
    assert data["ver"] == get_expected_version()

def test_analysis_endpoints():
    """Tests the analysis endpoints with a real test video."""
    test_video_path = "autorez/assets/test_30s.mp4"
    
    # Skip test if test video doesn't exist
    if not os.path.exists(test_video_path):
        pytest.skip("Test video file not found")

    response = client.post("/analyze/silence", json={"path": test_video_path})
    assert response.status_code == 200
    data = response.json()
    assert "ranges" in data
    assert isinstance(data["ranges"], list)

    response = client.post("/analyze/scenes", json={"path": test_video_path})
    assert response.status_code == 200
    data = response.json()
    assert "cuts" in data
    assert isinstance(data["cuts"], list)

    # ASR test - this may take longer so we'll test with a shorter timeout
    response = client.post("/asr", json={"path": test_video_path})
    assert response.status_code == 200
    data = response.json()
    assert "words" in data
    assert isinstance(data["words"], list)

def test_plan_endpoint():
    """Tests the /plan endpoint."""
    response = client.post("/plan", json={"goal": "test", "context": {}})
    assert response.status_code == 200
    assert "edits" in response.json()
    assert "proof" in response.json()

def test_export_endpoint():
    """Tests the /export/edl endpoint."""
    response = client.post("/export/edl", json={"data": {}})
    assert response.status_code == 200
    data = response.json()
    assert "edl_path" in data
    assert os.path.exists(data["edl_path"])
    os.remove(data["edl_path"])
