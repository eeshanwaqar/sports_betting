# Training Dockerfile - ML model training environment
#
# Build:  docker build -f infra/docker/training.Dockerfile -t epl-train .
# Run:    docker run -v ./models:/app/models -v ./data:/app/data epl-train

FROM python:3.11-slim

WORKDIR /app

# Git commit SHA for MLflow lineage tagging
ARG GIT_COMMIT_SHA=unknown
ENV GIT_COMMIT_SHA=${GIT_COMMIT_SHA}

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (full ML stack)
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt && \
    pip install --no-cache-dir mlflow==2.20.0

# Copy source code
COPY src/ src/
COPY config.yaml .
COPY scripts/train.py scripts/

# Copy feature data for self-contained training
COPY data/features/model_ready.csv data/features/

# Default command: run training with MLflow
CMD ["python", "scripts/train.py"]
