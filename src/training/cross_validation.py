"""
Cross Validation - Time-series aware cross-validation.

Maps to: notebooks/04_modeling.ipynb (CV section)

Uses TimeSeriesSplit to respect temporal ordering.
"""

import numpy as np
from typing import Dict, List, Any

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def time_series_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scoring: str = "accuracy",
) -> np.ndarray:
    """
    Run cross-validation with TimeSeriesSplit.

    Args:
        model: Sklearn-compatible model (will be cloned).
        X: Feature matrix.
        y: Target vector.
        n_splits: Number of CV folds.
        scoring: Scoring metric.

    Returns:
        Array of scores for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Clone model to avoid mutating the original
    model_clone = model.__class__(**model.get_params())

    scores = cross_val_score(model_clone, X, y, cv=tscv, scoring=scoring)

    logger.info(
        f"CV ({n_splits} folds): mean={scores.mean():.4f} (+/- {scores.std():.4f}), "
        f"folds={[f'{s:.4f}' for s in scores]}"
    )
    return scores


def cv_all_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Run CV for all models. Uses scaled data for linear models, raw for trees.

    Args:
        models: Dict of model name → model instance.
        X_train: Raw feature matrix.
        X_train_scaled: Scaled feature matrix.
        y_train: Target vector.
        n_splits: Number of folds.

    Returns:
        Dict of model name → CV scores array.
    """
    cv_results: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        X_cv = X_train_scaled if "Logistic" in name else X_train
        logger.info(f"Running CV for {name}...")
        cv_results[name] = time_series_cv(model, X_cv, y_train, n_splits)

    return cv_results
