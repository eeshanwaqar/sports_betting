"""
Form Features - Rolling form, exponential form, venue form, and streaks.

Maps to: notebooks/03_feature_engineering.ipynb
  - calc_form()
  - calc_exp_form()
  - calc_venue_form()
  - calc_streaks()

EDA Insight: Form diff is a strong predictor (corr ~0.25-0.28).
All teams perform better at home. Momentum effect confirmed.
"""


import numpy as np
import pandas as pd

from src.features.base import get_prior_matches
from src.utils.constants import (
    DEFAULT_EXP_DECAY,
    DEFAULT_VENUE_WINDOW,
)


def calc_form(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n: int = 5,
) -> dict:
    """
    Calculate rolling form from last N matches before a date.

    Args:
        df: Full match DataFrame.
        team: Team name.
        before_date: Temporal boundary.
        n: Number of recent matches.

    Returns:
        Dict with form{n}_points, form{n}_win_rate, form{n}_goal_diff,
        form{n}_avg_goals_for, form{n}_avg_goals_against.
    """
    prior = get_prior_matches(df, team, before_date, n=n)

    if len(prior) == 0:
        return {
            f"form{n}_points": np.nan,
            f"form{n}_win_rate": np.nan,
            f"form{n}_goal_diff": np.nan,
            f"form{n}_avg_goals_for": np.nan,
            f"form{n}_avg_goals_against": np.nan,
        }

    return {
        f"form{n}_points": prior["Points"].sum(),
        f"form{n}_win_rate": (prior["Result"] == "W").mean(),
        f"form{n}_goal_diff": prior["GoalsFor"].sum() - prior["GoalsAgainst"].sum(),
        f"form{n}_avg_goals_for": prior["GoalsFor"].mean(),
        f"form{n}_avg_goals_against": prior["GoalsAgainst"].mean(),
    }


def calc_exp_form(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n: int = 5,
    decay: float = DEFAULT_EXP_DECAY,
) -> dict:
    """
    Exponentially weighted form. More recent matches count more.

    Decay=0.7 means last match has weight 1.0, previous 0.7, then 0.49, etc.

    Returns:
        Dict with exp_form value.
    """
    prior = get_prior_matches(df, team, before_date, n=n)

    if len(prior) == 0:
        return {"exp_form": np.nan}

    weights = np.array([decay ** i for i in range(len(prior) - 1, -1, -1)])
    weighted_points = np.average(prior["Points"].values, weights=weights)

    return {"exp_form": weighted_points}


def calc_venue_form(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    is_home: bool = True,
    n: int = DEFAULT_VENUE_WINDOW,
) -> dict:
    """
    Venue-specific form (last N home or away matches).

    EDA: All teams perform better at home. Captures venue-specific momentum.

    Returns:
        Dict with venue_win_rate, venue_avg_goals_for,
        venue_avg_goals_against, venue_clean_sheets.
    """
    prior = get_prior_matches(df, team, before_date, n=n, home_only=is_home)

    if len(prior) == 0:
        return {
            "venue_win_rate": np.nan,
            "venue_avg_goals_for": np.nan,
            "venue_avg_goals_against": np.nan,
            "venue_clean_sheets": np.nan,
        }

    return {
        "venue_win_rate": (prior["Result"] == "W").mean(),
        "venue_avg_goals_for": prior["GoalsFor"].mean(),
        "venue_avg_goals_against": prior["GoalsAgainst"].mean(),
        "venue_clean_sheets": (prior["GoalsAgainst"] == 0).mean(),
    }


def calc_streaks(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
) -> dict:
    """
    Win streak, unbeaten streak, and loss streak entering the match.

    EDA: Momentum effect confirmed — teams on winning runs tend to continue.

    Returns:
        Dict with win_streak, unbeaten_streak, loss_streak.
    """
    prior = get_prior_matches(df, team, before_date)

    if len(prior) == 0:
        return {"win_streak": 0, "unbeaten_streak": 0, "loss_streak": 0}

    results = prior["Result"].values[::-1]  # Most recent first

    win_streak = 0
    for r in results:
        if r == "W":
            win_streak += 1
        else:
            break

    unbeaten_streak = 0
    for r in results:
        if r in ("W", "D"):
            unbeaten_streak += 1
        else:
            break

    loss_streak = 0
    for r in results:
        if r == "L":
            loss_streak += 1
        else:
            break

    return {
        "win_streak": win_streak,
        "unbeaten_streak": unbeaten_streak,
        "loss_streak": loss_streak,
    }
