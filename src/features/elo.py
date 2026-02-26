"""
Elo Rating System - Dynamic team strength encoding.

Maps to: notebooks/03_feature_engineering.ipynb (build_elo_ratings)

EDA Insight: Team quality varies from 20% to 65% win rate.
Elo captures this dynamically, updating after each match.
The difference HomeElo - AwayElo is one of the strongest predictors.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.constants import DEFAULT_ELO_K, DEFAULT_ELO_HOME_ADV, DEFAULT_INITIAL_ELO

logger = get_logger(__name__)


def build_elo_ratings(
    df: pd.DataFrame,
    k: int = DEFAULT_ELO_K,
    home_advantage: int = DEFAULT_ELO_HOME_ADV,
    initial_elo: int = DEFAULT_INITIAL_ELO,
) -> Tuple[Dict[Tuple[str, pd.Timestamp], float], Dict[str, float]]:
    """
    Build Elo ratings for all teams across the dataset.

    Processes matches chronologically, updating ratings after each match.
    Uses a goal-difference multiplier to reward bigger wins.

    Args:
        df: Match DataFrame sorted by date.
        k: K-factor controlling sensitivity to recent results.
        home_advantage: Elo bonus for the home team in expected score.
        initial_elo: Starting Elo for new teams.

    Returns:
        Tuple of:
            - elo_history: Dict mapping (team, date) → Elo rating BEFORE that match.
            - current_ratings: Dict mapping team → latest Elo rating AFTER all matches.
    """
    elo_ratings: Dict[str, float] = {}
    elo_history: Dict[Tuple[str, pd.Timestamp], float] = {}

    df_sorted = df.sort_values("Date").reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]

        # Initialize new teams
        if home not in elo_ratings:
            elo_ratings[home] = initial_elo
        if away not in elo_ratings:
            elo_ratings[away] = initial_elo

        home_elo = elo_ratings[home]
        away_elo = elo_ratings[away]

        # Store pre-match Elo
        elo_history[(home, date)] = home_elo
        elo_history[(away, date)] = away_elo

        # Expected scores (with home advantage)
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))
        exp_away = 1 - exp_home

        # Actual scores
        if row["FTR"] == "H":
            actual_home, actual_away = 1.0, 0.0
        elif row["FTR"] == "A":
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        # Goal difference multiplier (rewards bigger wins)
        goal_diff = abs(row["FTHG"] - row["FTAG"])
        gd_mult = max(1.0, np.log(goal_diff + 1) + 1) if goal_diff > 0 else 1.0

        # Update ratings
        elo_ratings[home] = home_elo + k * gd_mult * (actual_home - exp_home)
        elo_ratings[away] = away_elo + k * gd_mult * (actual_away - exp_away)

    logger.info(f"Built Elo ratings: {len(elo_history)} entries for {len(elo_ratings)} teams")
    return elo_history, dict(elo_ratings)


def calc_elo_features(
    row: pd.Series,
    elo_history: Dict[Tuple[str, pd.Timestamp], float],
    initial_elo: int = DEFAULT_INITIAL_ELO,
    current_ratings: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Extract Elo features for a single match.

    Falls back to current (latest) Elo ratings when the exact (team, date)
    key is not found in elo_history (e.g. during live inference for a future match).
    If current_ratings is also unavailable, falls back to initial_elo.

    Returns:
        Dict with home_elo, away_elo, elo_diff.
    """
    home_team = row["HomeTeam"]
    away_team = row["AwayTeam"]
    date = row["Date"]

    # Try exact history lookup first, then fall back to latest known rating
    home_elo = elo_history.get((home_team, date))
    if home_elo is None:
        fallback = current_ratings or {}
        home_elo = fallback.get(home_team, initial_elo)

    away_elo = elo_history.get((away_team, date))
    if away_elo is None:
        fallback = current_ratings or {}
        away_elo = fallback.get(away_team, initial_elo)

    return {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
    }
