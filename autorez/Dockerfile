# AutoResolve v3.2 - Production Dockerfile
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development target
FROM base as development
COPY . .
CMD ["uvicorn", "backend_service_final:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production target
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p logs exports temp artifacts/cache

# Set Python path
ENV PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server with multiple workers
CMD ["uvicorn", "backend_service_final:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--access-log"]