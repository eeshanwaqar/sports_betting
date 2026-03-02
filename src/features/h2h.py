"""
Head-to-Head Features - Historical matchup statistics.

Maps to: notebooks/03_feature_engineering.ipynb (calc_h2h)
"""

import numpy as np
import pandas as pd

from src.utils.constants import DEFAULT_H2H_WINDOW


def calc_h2h(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_date: pd.Timestamp,
    n: int = DEFAULT_H2H_WINDOW,
) -> dict:
    """
    Head-to-head stats from the home team's perspective.

    Args:
        df: Full match DataFrame.
        home_team: Home team name.
        away_team: Away team name.
        before_date: Only matches before this date.
        n: Max number of recent meetings to consider.

    Returns:
        Dict with h2h_played, h2h_win_rate, h2h_avg_gf.
    """
    h2h = df[
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team))
        | ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    ]
    h2h = h2h[h2h["Date"] < before_date].tail(n)

    if len(h2h) == 0:
        return {"h2h_played": 0, "h2h_win_rate": np.nan, "h2h_avg_gf": np.nan}

    wins = 0
    goals_for = 0
    for _, m in h2h.iterrows():
        if m["HomeTeam"] == home_team:
            goals_for += m["FTHG"]
            if m["FTR"] == "H":
                wins += 1
        else:
            goals_for += m["FTAG"]
            if m["FTR"] == "A":
                wins += 1

    return {
        "h2h_played": len(h2h),
        "h2h_win_rate": wins / len(h2h),
        "h2h_avg_gf": goals_for / len(h2h),
    }
