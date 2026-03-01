"""
Dependencies - FastAPI dependency injection.

Provides singleton instances for the prediction engine and configuration.
These are injected into route handlers via FastAPI's Depends() system.
"""

from functools import lru_cache

from src.utils.config import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton for the predictor (lazy initialized)
_predictor_instance = None


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache application config (singleton)."""
    return load_config()


def get_predictor():
    """
    Get or create the MatchPredictor singleton.

    Lazy-loads the predictor on first call. This avoids loading
    the model and historical data at import time.
    """
    global _predictor_instance

    if _predictor_instance is None:
        from src.inference.predictor import MatchPredictor

        config = get_config()
        _predictor_instance = MatchPredictor(config)
        logger.info("MatchPredictor initialized (lazy)")

    return _predictor_instance


def reset_predictor() -> None:
    """
    Force re-initialization of the predictor.

    Call this after retraining to pick up new model artifacts.
    """
    global _predictor_instance
    _predictor_instance = None
    logger.info("Predictor reset — will reload on next request")
