"""
Odds Features - Betting odds feature extraction.

Maps to: notebooks/03_feature_engineering.ipynb (calc_odds_features)

EDA Insight: Bookmaker ~52% accuracy — strong benchmark and feature source.
"""

import pandas as pd
import numpy as np


def calc_odds_features(row: pd.Series) -> dict:
    """
    Extract normalized implied probabilities from bookmaker odds.

    Converts B365 decimal odds to probabilities, removes the overround,
    and computes derived features.

    Args:
        row: A single match row with B365H, B365D, B365A columns.

    Returns:
        Dict with odds_prob_home, odds_prob_draw, odds_prob_away,
        odds_home_away_diff, odds_fav_is_home.
    """
    if pd.isna(row.get("B365H")) or pd.isna(row.get("B365D")) or pd.isna(row.get("B365A")):
        return {
            "odds_prob_home": np.nan,
            "odds_prob_draw": np.nan,
            "odds_prob_away": np.nan,
            "odds_home_away_diff": np.nan,
            "odds_fav_is_home": np.nan,
        }

    # Convert to implied probabilities
    p_h = 1 / row["B365H"]
    p_d = 1 / row["B365D"]
    p_a = 1 / row["B365A"]

    # Normalize (remove overround)
    total = p_h + p_d + p_a
    p_h /= total
    p_d /= total
    p_a /= total

    return {
        "odds_prob_home": p_h,
        "odds_prob_draw": p_d,
        "odds_prob_away": p_a,
        "odds_home_away_diff": p_h - p_a,
        "odds_fav_is_home": int(p_h > p_a),
    }
