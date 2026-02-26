"""
League Position Features - Standing-derived features.

Placeholder for future Feast/standings integration.
Currently league position is approximated via season PPG in team_stats.
"""

import pandas as pd
import numpy as np
from typing import Optional


def calc_league_position(
    standings: Optional[pd.DataFrame],
    team: str,
    season: str,
) -> dict:
    """
    Get league position for a team in a season from standings data.

    Args:
        standings: DataFrame with Season, Team, Position columns (or None).
        team: Team name.
        season: Season label (e.g. '2017-18').

    Returns:
        Dict with league_position.
    """
    if standings is None or len(standings) == 0:
        return {"league_position": np.nan}

    match = standings[(standings["Team"] == team) & (standings["Season"] == season)]
    if len(match) == 0:
        return {"league_position": np.nan}

    return {"league_position": match.iloc[0]["Position"]}
