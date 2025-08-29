
# PRODUCTION-GRADE IMPLEMENTATION
# Blueprint Compliance: REQ-001
# Security: SEC-002, SEC-003 (to be implemented)
# Performance: PERF-001 (to be implemented)

import sys
import fastapi

# Blueprint Requirement: REQ-001
# Verify Python version
MIN_PYTHON_VERSION = (3, 12)
if sys.version_info < MIN_PYTHON_VERSION:
    sys.exit(f"FATAL: Python {'.'.join(map(str, MIN_PYTHON_VERSION))} or later is required. Current version: {sys.version}")

# Verify FastAPI version
from fastapi import FastAPI
if fastapi.__version__ != "0.111.0": # Note: 0.111 becomes 0.111.0
    sys.exit(f"FATAL: FastAPI version 0.111.0 is required. Current version: {fastapi.__version__}")

app = FastAPI(
    title="AutoResolve Enterprise API",
    version="1.0.0",
    description="Enterprise-grade backend for the AutoResolve suite, compliant with Blueprint v1."
)

@app.get("/", tags=["Health Check"])
async def root():
    """
    Provides a basic health check and welcome message.
    Confirms the API is running and reachable.
    """
    return {"status": "ok", "message": "AutoResolve Enterprise API is running."}

# Future routers will be included here
# from .routers import timeline, pipeline, export, telemetry
# app.include_router(timeline.router)
# app.include_router(pipeline.router)
# app.include_router(export.router)
# app.include_router(telemetry.router)

# The auth router will be added in Milestone 1, Task 1.2
# from .security import auth
# app.include_router(auth.router)

