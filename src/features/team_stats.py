"""
Team Statistics Features - Season stats and shooting metrics.

Maps to: notebooks/03_feature_engineering.ipynb
  - calc_season_stats()
  - calc_shooting_stats()

EDA Insight: Season win rate has highest correlation (0.316) with outcome.
Shot conversion rates vary significantly between teams.
"""

import numpy as np
import pandas as pd

from src.features.base import get_prior_matches, get_team_matches
from src.utils.helpers import get_season_start


def calc_season_stats(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
) -> dict:
    """
    Calculate season-to-date statistics.

    Falls back to last 10 matches if fewer than 3 season matches
    (start of season).

    Returns:
        Dict with season_played, season_win_rate, season_ppg,
        season_avg_goals_for, season_avg_goals_against, season_clean_sheets.
    """
    matches = get_team_matches(df, team)
    prior = matches[matches["Date"] < before_date]

    season_start = get_season_start(before_date)
    season_matches = prior[prior["Date"] >= season_start]

    # Fallback if too few season matches
    if len(season_matches) < 3:
        season_matches = prior.tail(10)

    if len(season_matches) == 0:
        return {
            "season_played": 0,
            "season_win_rate": np.nan,
            "season_ppg": np.nan,
            "season_avg_goals_for": np.nan,
            "season_avg_goals_against": np.nan,
            "season_clean_sheets": np.nan,
        }

    return {
        "season_played": len(season_matches),
        "season_win_rate": (season_matches["Result"] == "W").mean(),
        "season_ppg": season_matches["Points"].mean(),
        "season_avg_goals_for": season_matches["GoalsFor"].mean(),
        "season_avg_goals_against": season_matches["GoalsAgainst"].mean(),
        "season_clean_sheets": (season_matches["GoalsAgainst"] == 0).mean(),
    }


def calc_shooting_stats(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n: int = 10,
) -> dict:
    """
    Shot-based features from last N matches.

    EDA: Shot conversion rates vary significantly between teams.

    Returns:
        Dict with avg_shots, avg_sot, shot_accuracy, shot_conversion.
    """
    prior = get_prior_matches(df, team, before_date, n=n)

    if len(prior) == 0 or "ShotsFor" not in prior.columns:
        return {
            "avg_shots": np.nan,
            "avg_sot": np.nan,
            "shot_accuracy": np.nan,
            "shot_conversion": np.nan,
        }

    valid = prior[prior["ShotsFor"] > 0]
    if len(valid) == 0:
        return {
            "avg_shots": np.nan,
            "avg_sot": np.nan,
            "shot_accuracy": np.nan,
            "shot_conversion": np.nan,
        }

    total_shots = valid["ShotsFor"].sum()
    total_sot = valid["SOTFor"].sum() if "SOTFor" in valid.columns else 0
    total_goals = valid["GoalsFor"].sum()

    return {
        "avg_shots": valid["ShotsFor"].mean(),
        "avg_sot": valid["SOTFor"].mean() if "SOTFor" in valid.columns else np.nan,
        "shot_accuracy": total_sot / total_shots if total_shots > 0 else np.nan,
        "shot_conversion": total_goals / total_shots if total_shots > 0 else np.nan,
    }
