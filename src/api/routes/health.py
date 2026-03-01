"""
Health Routes - Health check and readiness endpoints.

Endpoints:
    GET /health       - Liveness check (always 200 if the app is running)
    GET /health/ready - Readiness check (verifies model is loadable)
"""

from fastapi import APIRouter

from src.api.dependencies import get_predictor
from src.api.schemas import HealthResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Liveness probe.

    Returns 200 if the application is running.
    Does not check external dependencies.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=False,
        version="1.0.0",
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """
    Readiness probe.

    Verifies that the model is loaded and ready to serve predictions.
    Returns 503 if the model cannot be loaded.
    """
    try:
        predictor = get_predictor()
        model_info = predictor.loader.model_info
        return HealthResponse(
            status="ready",
            model_loaded=True,
            model_type=model_info.get("model_type", "unknown"),
            version="1.0.0",
        )
    except Exception as exc:
        logger.warning(f"Readiness check failed: {exc}")
        return HealthResponse(
            status="not_ready",
            model_loaded=False,
            version="1.0.0",
        )
