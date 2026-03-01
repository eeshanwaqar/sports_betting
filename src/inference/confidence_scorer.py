"""
Confidence Scorer - Rate prediction confidence.

Combines model probability confidence with data freshness indicators.
"""


import numpy as np


def score_confidence(
    max_probability: float,
    data_recency_days: int = 0,
    h2h_matches: int = 0,
) -> float:
    """
    Compute a composite confidence score.

    Factors:
    - Model probability (higher max prob = more confident)
    - Data recency (more recent data = more reliable)
    - H2H sample size (more meetings = better H2H estimate)

    Args:
        max_probability: Maximum predicted probability (0-1).
        data_recency_days: Days since most recent match in features.
        h2h_matches: Number of H2H meetings used.

    Returns:
        Confidence score between 0 and 1.
    """
    # Base confidence from probability
    prob_score = max_probability

    # Recency penalty (features based on old data are less reliable)
    recency_penalty = min(data_recency_days / 365, 0.2)  # Max 20% penalty

    # H2H bonus (more data = slightly more confident)
    h2h_bonus = min(h2h_matches * 0.01, 0.05)  # Max 5% bonus

    confidence = prob_score - recency_penalty + h2h_bonus
    return float(np.clip(confidence, 0.0, 1.0))
