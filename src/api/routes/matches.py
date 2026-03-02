"""
Matches Routes - Historical match data endpoints.

Endpoints:
    GET /matches/recent          - Recent match results
    GET /matches/head-to-head    - Head-to-head record between two teams
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.dependencies import get_predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/matches", tags=["Matches"])


@router.get("/recent")
async def recent_matches(
    team: Optional[str] = Query(None, description="Filter by team name"),
    limit: int = Query(10, ge=1, le=50, description="Number of matches"),
):
    """
    Get recent historical match results.

    Optionally filter by team name.
    """
    predictor = get_predictor()
    df = predictor.assembler.historical_df.copy()

    if team:
        available = predictor.get_available_teams()
        if team not in available:
            raise HTTPException(status_code=404, detail=f"Team '{team}' not found")
        df = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]

    recent = df.sort_values("Date", ascending=False).head(limit)

    matches = []
    for _, row in recent.iterrows():
        matches.append({
            "date": str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"]),
            "home_team": row["HomeTeam"],
            "away_team": row["AwayTeam"],
            "home_goals": int(row["FTHG"]),
            "away_goals": int(row["FTAG"]),
            "result": row["FTR"],
        })

    return {"matches": matches, "count": len(matches)}


@router.get("/head-to-head")
async def head_to_head(
    team_a: str = Query(..., description="First team"),
    team_b: str = Query(..., description="Second team"),
    limit: int = Query(10, ge=1, le=50, description="Max matches to return"),
):
    """
    Get head-to-head record between two teams.

    Returns recent meetings and overall record summary.
    """
    predictor = get_predictor()
    available = predictor.get_available_teams()

    for team in [team_a, team_b]:
        if team not in available:
            raise HTTPException(status_code=404, detail=f"Team '{team}' not found")

    df = predictor.assembler.historical_df

    # Both directions
    h2h = df[
        ((df["HomeTeam"] == team_a) & (df["AwayTeam"] == team_b))
        | ((df["HomeTeam"] == team_b) & (df["AwayTeam"] == team_a))
    ].sort_values("Date", ascending=False)

    total = len(h2h)

    # Calculate wins for each team
    team_a_wins = 0
    team_b_wins = 0
    draws = 0

    for _, row in h2h.iterrows():
        if row["FTR"] == "D":
            draws += 1
        elif row["HomeTeam"] == team_a and row["FTR"] == "H":
            team_a_wins += 1
        elif row["AwayTeam"] == team_a and row["FTR"] == "A":
            team_a_wins += 1
        else:
            team_b_wins += 1

    recent = h2h.head(limit)
    matches = []
    for _, row in recent.iterrows():
        matches.append({
            "date": str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"]),
            "home_team": row["HomeTeam"],
            "away_team": row["AwayTeam"],
            "home_goals": int(row["FTHG"]),
            "away_goals": int(row["FTAG"]),
            "result": row["FTR"],
        })

    return {
        "team_a": team_a,
        "team_b": team_b,
        "total_meetings": total,
        "summary": {
            f"{team_a}_wins": team_a_wins,
            f"{team_b}_wins": team_b_wins,
            "draws": draws,
        },
        "recent_matches": matches,
    }
