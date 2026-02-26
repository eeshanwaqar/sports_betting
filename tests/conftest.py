"""
Pytest Configuration - Shared fixtures for all tests.

Provides:
- sample_matches_df: A minimal but realistic EPL match DataFrame
- sample_config: A test-safe AppConfig instance
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_matches_df() -> pd.DataFrame:
    """
    Build a small, realistic EPL match DataFrame with 12 matches.

    Covers two teams (Arsenal, Chelsea) plus a third (Liverpool) across
    multiple dates, with Home/Draw/Away outcomes and shot stats.
    This is enough to exercise form, h2h, elo, team_stats, and streaks.
    """
    records = [
        # Season opener: Arsenal home win
        {"Date": "2023-08-12", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
         "FTHG": 2, "FTAG": 1, "FTR": "H", "HS": 14, "AS": 8, "HST": 6, "AST": 3,
         "B365H": 1.80, "B365D": 3.50, "B365A": 4.50},
        # Chelsea home win
        {"Date": "2023-08-19", "HomeTeam": "Chelsea", "AwayTeam": "Liverpool",
         "FTHG": 3, "FTAG": 0, "FTR": "H", "HS": 16, "AS": 5, "HST": 8, "AST": 2,
         "B365H": 2.10, "B365D": 3.40, "B365A": 3.50},
        # Draw
        {"Date": "2023-08-26", "HomeTeam": "Liverpool", "AwayTeam": "Arsenal",
         "FTHG": 1, "FTAG": 1, "FTR": "D", "HS": 10, "AS": 12, "HST": 4, "AST": 5,
         "B365H": 2.20, "B365D": 3.30, "B365A": 3.30},
        # Arsenal home win
        {"Date": "2023-09-02", "HomeTeam": "Arsenal", "AwayTeam": "Liverpool",
         "FTHG": 3, "FTAG": 1, "FTR": "H", "HS": 18, "AS": 7, "HST": 9, "AST": 3,
         "B365H": 1.70, "B365D": 3.60, "B365A": 5.00},
        # Chelsea away win
        {"Date": "2023-09-16", "HomeTeam": "Liverpool", "AwayTeam": "Chelsea",
         "FTHG": 0, "FTAG": 2, "FTR": "A", "HS": 9, "AS": 13, "HST": 3, "AST": 6,
         "B365H": 2.00, "B365D": 3.50, "B365A": 3.80},
        # Draw
        {"Date": "2023-09-23", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
         "FTHG": 1, "FTAG": 1, "FTR": "D", "HS": 11, "AS": 10, "HST": 4, "AST": 4,
         "B365H": 1.90, "B365D": 3.40, "B365A": 4.20},
        # Liverpool home win
        {"Date": "2023-10-07", "HomeTeam": "Liverpool", "AwayTeam": "Arsenal",
         "FTHG": 2, "FTAG": 0, "FTR": "H", "HS": 15, "AS": 6, "HST": 7, "AST": 2,
         "B365H": 2.30, "B365D": 3.30, "B365A": 3.10},
        # Chelsea home draw
        {"Date": "2023-10-21", "HomeTeam": "Chelsea", "AwayTeam": "Arsenal",
         "FTHG": 2, "FTAG": 2, "FTR": "D", "HS": 12, "AS": 14, "HST": 5, "AST": 6,
         "B365H": 2.40, "B365D": 3.30, "B365A": 3.00},
        # Arsenal home win
        {"Date": "2023-11-04", "HomeTeam": "Arsenal", "AwayTeam": "Liverpool",
         "FTHG": 1, "FTAG": 0, "FTR": "H", "HS": 13, "AS": 9, "HST": 5, "AST": 4,
         "B365H": 1.85, "B365D": 3.50, "B365A": 4.30},
        # Chelsea away loss
        {"Date": "2023-11-11", "HomeTeam": "Liverpool", "AwayTeam": "Chelsea",
         "FTHG": 4, "FTAG": 1, "FTR": "H", "HS": 20, "AS": 7, "HST": 10, "AST": 3,
         "B365H": 1.90, "B365D": 3.50, "B365A": 4.10},
        # Arsenal away win
        {"Date": "2023-11-25", "HomeTeam": "Chelsea", "AwayTeam": "Arsenal",
         "FTHG": 0, "FTAG": 3, "FTR": "A", "HS": 8, "AS": 17, "HST": 3, "AST": 8,
         "B365H": 2.50, "B365D": 3.30, "B365A": 2.90},
        # Final: Liverpool home win
        {"Date": "2023-12-02", "HomeTeam": "Liverpool", "AwayTeam": "Arsenal",
         "FTHG": 2, "FTAG": 1, "FTR": "H", "HS": 14, "AS": 11, "HST": 6, "AST": 5,
         "B365H": 2.20, "B365D": 3.40, "B365A": 3.20},
    ]

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df


@pytest.fixture
def cutoff_date() -> pd.Timestamp:
    """A mid-season cutoff date used to separate 'known' vs 'future' matches."""
    return pd.Timestamp("2023-11-01")


@pytest.fixture
def late_cutoff_date() -> pd.Timestamp:
    """A late cutoff ensuring all 12 sample matches are in the past."""
    return pd.Timestamp("2024-01-01")
