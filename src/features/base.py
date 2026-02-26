"""
Base Feature Utilities - Shared helpers for all feature modules.

Maps to: notebooks/03_feature_engineering.ipynb (helper functions)

Provides get_team_matches() and get_prior_matches() used by all calculators.
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.utils.constants import HOME_RESULT_MAP, AWAY_RESULT_MAP, POINTS_MAP


def get_team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Get all matches for a team with standardized columns.

    Combines home and away matches into a unified view where:
    - GoalsFor/Against are from the team's perspective
    - Result is W/D/L from the team's perspective
    - IsHome indicates venue

    Args:
        df: Full match DataFrame with HomeTeam, AwayTeam, FTHG, FTAG, FTR columns.
        team: Team name.

    Returns:
        Chronologically sorted DataFrame of the team's matches.
    """
    # Home matches
    home = df[df["HomeTeam"] == team].copy()
    home["IsHome"] = True
    home["GoalsFor"] = home["FTHG"]
    home["GoalsAgainst"] = home["FTAG"]
    home["Result"] = home["FTR"].map(HOME_RESULT_MAP)
    home["Points"] = home["Result"].map(POINTS_MAP)

    if "HS" in df.columns:
        home["ShotsFor"] = home["HS"]
        home["ShotsAgainst"] = home["AS"]
    if "HST" in df.columns:
        home["SOTFor"] = home["HST"]
        home["SOTAgainst"] = home["AST"]

    # Away matches
    away = df[df["AwayTeam"] == team].copy()
    away["IsHome"] = False
    away["GoalsFor"] = away["FTAG"]
    away["GoalsAgainst"] = away["FTHG"]
    away["Result"] = away["FTR"].map(AWAY_RESULT_MAP)
    away["Points"] = away["Result"].map(POINTS_MAP)

    if "HS" in df.columns:
        away["ShotsFor"] = away["AS"]
        away["ShotsAgainst"] = away["HS"]
    if "HST" in df.columns:
        away["SOTFor"] = away["AST"]
        away["SOTAgainst"] = away["HST"]

    combined = pd.concat([home, away]).sort_values("Date").reset_index(drop=True)
    return combined


def get_prior_matches(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n: Optional[int] = None,
    home_only: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Get prior matches for a team before a given date.

    Args:
        df: Full match DataFrame.
        team: Team name.
        before_date: Only include matches strictly before this date.
        n: If provided, limit to last N matches.
        home_only: If True, only home matches. If False, only away. None for both.

    Returns:
        DataFrame of prior matches (may be empty).
    """
    matches = get_team_matches(df, team)
    prior = matches[matches["Date"] < before_date]

    if home_only is not None:
        prior = prior[prior["IsHome"] == home_only]

    if n is not None:
        prior = prior.tail(n)

    return prior
