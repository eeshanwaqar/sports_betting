"""
Evaluator - Model evaluation metrics and analysis.

Maps to: notebooks/04_modeling.ipynb (analysis cells) + 05_evaluation.ipynb

Provides classification metrics, feature importance extraction,
and model comparison utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Returns:
        Dict with accuracy, f1_macro, per-class metrics, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def get_feature_importance(model: Any, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Extract feature importance from any model type.

    Works for:
    - Tree models (feature_importances_)
    - Linear models (coef_ → mean absolute coefficient)

    Returns:
        Sorted DataFrame with Feature and Importance columns, or None.
    """
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    if hasattr(model, "coef_"):
        mean_abs_coef = np.abs(model.coef_).mean(axis=0)
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": mean_abs_coef,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return None


def bookmaker_accuracy(df: pd.DataFrame) -> Optional[float]:
    """
    Calculate bookmaker prediction accuracy from odds features.

    Uses odds_prob_home/draw/away to determine bookmaker's predicted outcome,
    then compares to actual target.

    Returns:
        Accuracy float, or None if odds columns not available.
    """
    required = ["odds_prob_home", "odds_prob_draw", "odds_prob_away", "target"]
    if not all(c in df.columns for c in required):
        return None

    bookie_pred = df[["odds_prob_home", "odds_prob_draw", "odds_prob_away"]].idxmax(axis=1)
    bookie_pred = bookie_pred.map({
        "odds_prob_home": "H",
        "odds_prob_draw": "D",
        "odds_prob_away": "A",
    })

    return (bookie_pred == df["target"].values).mean()


def compute_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Compute log loss (measures calibration + discrimination).

    Lower is better. Random baseline = -log(1/3) ≈ 1.099.
    """
    prob_normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    return log_loss(y_true, prob_normalized)
