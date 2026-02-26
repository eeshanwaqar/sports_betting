"""
Model Loader - Load trained model artifacts from disk.

Loads: best_model.joblib, scaler.joblib, label_encoder.joblib,
       features.txt, model_info.json
"""

import json
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Loads and caches trained model artifacts.

    Usage:
        loader = ModelLoader("models/")
        model = loader.model
        scaler = loader.scaler
        features = loader.feature_names
    """

    def __init__(self, models_dir: str = "models"):
        self._dir = Path(models_dir)
        self._model: Optional[Any] = None
        self._calibrator: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._label_encoder: Optional[Any] = None
        self._feature_names: Optional[List[str]] = None
        self._model_info: Optional[Dict] = None
        self._train_medians: Optional[Dict[str, float]] = None

        if not self._dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self._dir}")

    @property
    def model(self) -> Any:
        """Trained model (lazy loaded, cached)."""
        if self._model is None:
            path = self._dir / "best_model.joblib"
            self._model = joblib.load(path)
            logger.info(f"Loaded model from {path}")
        return self._model

    @property
    def calibrator(self) -> Optional[Any]:
        """Probability calibrator (lazy loaded, cached). Returns None if not available."""
        if self._calibrator is None:
            path = self._dir / "best_calibrator.joblib"
            if path.exists():
                self._calibrator = joblib.load(path)
                logger.info(f"Loaded calibrator from {path}")
            else:
                logger.info("No calibrator found — using raw probabilities")
        return self._calibrator

    @property
    def scaler(self) -> Any:
        """Feature scaler (lazy loaded, cached)."""
        if self._scaler is None:
            path = self._dir / "scaler.joblib"
            self._scaler = joblib.load(path)
            logger.info(f"Loaded scaler from {path}")
        return self._scaler

    @property
    def label_encoder(self) -> Any:
        """Target label encoder (lazy loaded, cached)."""
        if self._label_encoder is None:
            path = self._dir / "label_encoder.joblib"
            self._label_encoder = joblib.load(path)
            logger.info(f"Loaded label encoder from {path}")
        return self._label_encoder

    @property
    def feature_names(self) -> List[str]:
        """List of feature column names."""
        if self._feature_names is None:
            path = self._dir / "features.txt"
            with open(path) as f:
                self._feature_names = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self._feature_names)} feature names")
        return self._feature_names

    @property
    def model_info(self) -> Dict:
        """Model metadata and metrics."""
        if self._model_info is None:
            path = self._dir / "model_info.json"
            with open(path) as f:
                self._model_info = json.load(f)
        return self._model_info

    @property
    def train_medians(self) -> Dict[str, float]:
        """
        Training set feature medians for NaN imputation at inference time.

        Features unavailable at prediction time (e.g. betting odds for future
        matches) are filled with training medians instead of 0.0 to avoid
        sending extreme out-of-distribution values to the model.
        """
        if self._train_medians is None:
            path = self._dir / "train_medians.joblib"
            if path.exists():
                self._train_medians = joblib.load(path)
                logger.info(f"Loaded training medians from {path}")
            else:
                logger.warning(
                    f"Training medians not found at {path}. "
                    "NaN features will fall back to 0.0. "
                    "Retrain the model to generate train_medians.joblib."
                )
                self._train_medians = {}
        return self._train_medians

    def reload(self) -> None:
        """Force reload all artifacts (e.g. after retraining)."""
        self._model = None
        self._calibrator = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = None
        self._model_info = None
        self._train_medians = None
        logger.info("Model cache cleared — will reload on next access")
