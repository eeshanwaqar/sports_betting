"""
FastAPI Application - EPL Betting Odds Predictor API.

Creates the FastAPI app, registers routes, and configures middleware.
The frontend is served separately on port 3000.

Usage:
    uvicorn src.api.app:app --reload
    # or
    python scripts/run_api.py
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import (
    health_router,
    predictions_router,
    teams_router,
    matches_router,
)
from src.api.dependencies import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup: log configuration.
    Shutdown: clean up resources.
    """
    config = get_config()
    logger.info(
        f"Starting EPL Predictor API | "
        f"model_dir={config.model.models_dir} | "
        f"mlflow={config.mlflow.enabled}"
    )
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
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
