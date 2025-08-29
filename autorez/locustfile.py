from locust import HttpUser, task, between
import os

API_BASE = os.getenv("LOCUST_API_BASE", "http://localhost:8000/api")
API_KEY = os.getenv("API_KEY")

class BackendUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.headers = {"Content-Type": "application/json"}
        if API_KEY:
            self.headers["x-api-key"] = API_KEY

    @task(5)
    def get_projects(self):
        self.client.get(f"{API_BASE}/projects", headers=self.headers, name="GET /projects")

    @task(3)
    def telemetry(self):
        self.client.get(f"{API_BASE}/telemetry/metrics", headers=self.headers, name="GET /telemetry/metrics")

    @task(1)
    def validate(self):
        self.client.post(f"{API_BASE}/validate", json={"input_file": "README.md"}, headers=self.headers, name="POST /validate")


