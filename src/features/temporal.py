"""
Temporal Features - Time-based features.

Maps to: notebooks/03_feature_engineering.ipynb (temporal section)
"""

import pandas as pd

from src.utils.helpers import get_season_start


def calc_temporal_features(row: pd.Series) -> dict:
    """
    Extract temporal features from match date.

    Args:
        row: Match row with a Date column.

    Returns:
        Dict with day_of_week, month, is_weekend, matchweek.
    """
    date = row["Date"]

    season_start = get_season_start(date)
    matchweek = max(1, (date - season_start).days // 7)

    return {
        "day_of_week": date.dayofweek,
        "month": date.month,
        "is_weekend": int(date.dayofweek >= 5),
        "matchweek": matchweek,
    }
