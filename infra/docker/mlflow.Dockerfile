# MLflow Dockerfile - Official image + PostgreSQL driver
#
# The official ghcr.io/mlflow/mlflow image does not ship with psycopg2,
# which is required when using a PostgreSQL backend store.
# This image adds psycopg2-binary on top.
#
# Build:  docker build -f infra/docker/mlflow.Dockerfile -t epl-mlflow .
# Run:    docker run -p 5000:5000 epl-mlflow mlflow server --host 0.0.0.0

FROM ghcr.io/mlflow/mlflow:v2.20.0

RUN pip install --no-cache-dir psycopg2-binary
