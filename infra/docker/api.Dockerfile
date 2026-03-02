# API Dockerfile - FastAPI prediction server + web frontend
#
# Build:  docker build -f infra/docker/api.Dockerfile -t epl-api .
# Run:    docker run -p 8000:8000 -v ./models:/app/models:ro epl-api

FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (only what the API needs)
COPY requirements-simple.txt .
RUN pip install --no-cache-dir -r requirements-simple.txt && \
    pip install --no-cache-dir fastapi uvicorn pydantic mlflow boto3

# Copy source code
COPY src/ src/
COPY config.yaml .
COPY scripts/run_api.py scripts/

# Copy web frontend
COPY web/ web/

# Copy trained model artifacts (needed by ModelLoader)
COPY models/ models/

# Copy historical match data (needed by FeatureAssembler at inference time)
COPY data/raw/matches.csv data/raw/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the API server
CMD ["python", "scripts/run_api.py"]
