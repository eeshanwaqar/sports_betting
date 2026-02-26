"""
Odds Calculator - Convert model probabilities to betting odds.

Maps to: notebooks/05_evaluation.ipynb (prob_to_odds, value bet analysis)
"""

import numpy as np
from typing import Dict

from src.utils.constants import DEFAULT_ODDS_MARGIN


def prob_to_odds(prob: float, margin: float = DEFAULT_ODDS_MARGIN) -> float:
    """
    Convert probability to decimal betting odds.

    Args:
        prob: Probability (0 to 1).
        margin: Bookmaker margin (e.g. 0.05 = 5%).

    Returns:
        Decimal odds (e.g. 2.50).
    """
    if prob <= 0:
        return float("inf")
    return round(1 / (prob * (1 + margin)), 2)


def probs_to_odds(
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    margin: float = DEFAULT_ODDS_MARGIN,
) -> Dict[str, float]:
    """
    Convert all three probabilities to decimal odds.

    Returns:
        Dict with home, draw, away odds.
    """
    return {
        "home_win": prob_to_odds(prob_home, margin),
        "draw": prob_to_odds(prob_draw, margin),
        "away_win": prob_to_odds(prob_away, margin),
    }


def calc_value(
    model_prob: float,
    bookmaker_prob: float,
) -> float:
    """
    Calculate value: model_prob - bookmaker_prob.

    Positive value = model sees more value than bookmaker.
    """
    if np.isnan(model_prob) or np.isnan(bookmaker_prob):
        return 0.0
    return model_prob - bookmaker_prob


def is_value_bet(
    model_prob: float,
    bookmaker_prob: float,
    threshold: float = 0.0,
) -> bool:
    """Check if this is a value bet (model prob exceeds bookmaker by threshold)."""
    return calc_value(model_prob, bookmaker_prob) > threshold
