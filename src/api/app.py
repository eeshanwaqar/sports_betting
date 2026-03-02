"""
FastAPI Application - EPL Betting Odds Predictor API.

Creates the FastAPI app, registers routes, and configures middleware.
The frontend is served separately on port 3000.

Usage:
    uvicorn src.api.app:app --reload
    # or
    python scripts/run_api.py
"""

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import get_config
from src.api.routes import (
    health_router,
    matches_router,
    predictions_router,
    teams_router,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _download_model_from_mlflow(config) -> bool:
    """
    Download model artifacts from MLflow if local files are missing.

    Tries the 'champion' alias first, then falls back to the latest registered version.
    Returns True if download succeeded, False otherwise.
    """
    models_dir = Path(config.model.models_dir)
    sentinel = models_dir / "best_model.joblib"

    if sentinel.exists():
        logger.info(f"Model artifacts already present at {models_dir}")
        return True

    if not config.mlflow.enabled or config.mlflow.tracking_uri == "mlruns":
        logger.warning("MLflow not configured for remote — skipping model download")
        return False

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        client = MlflowClient()
        registry_name = config.mlflow.registry_name

        # Try champion alias first, fall back to latest version
        run_id = None
        try:
            version = client.get_model_version_by_alias(registry_name, "champion")
            run_id = version.run_id
            logger.info(f"Downloading champion model (version {version.version})")
        except Exception:
            versions = client.search_model_versions(
                f"name='{registry_name}'",
                order_by=["version_number DESC"],
                max_results=1,
            )
            if versions:
                run_id = versions[0].run_id
                logger.info(f"Downloading latest model (version {versions[0].version})")

        if run_id is None:
            logger.warning(f"No registered model found for '{registry_name}'")
            return False

        models_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="model_artifacts",
                dst_path=tmp,
            )
            # local_path = tmp/model_artifacts/ — copy contents into models_dir
            src = Path(local_path)
            for artifact in src.iterdir():
                shutil.copy2(artifact, models_dir / artifact.name)

        logger.info(f"Model artifacts downloaded to {models_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to download model from MLflow: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup: download model artifacts from MLflow if not present locally.
    Shutdown: clean up resources.
    """
    config = get_config()
    logger.info(
        f"Starting EPL Predictor API | "
        f"model_dir={config.model.models_dir} | "
        f"mlflow={config.mlflow.enabled}"
    )
    _download_model_from_mlflow(config)
    yield
    logger.info("Shutting down EPL Predictor API")


app = FastAPI(
    title="EPL Betting Odds Predictor",
    description=(
        "Predict English Premier League match outcomes with probabilities and "
        "decimal betting odds. Powered by ML models trained on historical data "
        "with MLflow experiment tracking."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS Middleware (allow frontend on port 3000) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins="*"
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# --- Register API Routers ---
app.include_router(health_router)
app.include_router(predictions_router)
app.include_router(teams_router)
app.include_router(matches_router)


@app.get("/", tags=["Root"])
async def root():
    """API root -- basic info and links."""
    return {
        "name": "EPL Betting Odds Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "frontend": "http://localhost:3000",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "teams": "GET /teams",
            "matches": "GET /matches/recent",
            "head_to_head": "GET /matches/head-to-head?team_a=...&team_b=...",
        },
    }
