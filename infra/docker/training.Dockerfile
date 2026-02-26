# Training Dockerfile - ML model training environment
#
# Build:  docker build -f infra/docker/training.Dockerfile -t epl-train .
# Run:    docker run -v ./models:/app/models -v ./data:/app/data epl-train

FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (full ML stack)
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt && \
    pip install --no-cache-dir mlflow

# Copy source code
COPY src/ src/
COPY config.yaml .
COPY scripts/train.py scripts/

# Default command: run training with MLflow
CMD ["python", "scripts/train.py"]
