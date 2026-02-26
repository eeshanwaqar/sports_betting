"""
Teams Routes - Team information endpoints.

Endpoints:
    GET /teams           - List all available teams
    GET /teams/{name}    - Team details and record
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import TeamResponse, TeamListResponse
from src.api.dependencies import get_predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/teams", tags=["Teams"])


@router.get("", response_model=TeamListResponse)
async def list_teams():
    """List all teams available for prediction."""
    predictor = get_predictor()
    teams = predictor.get_available_teams()
    return TeamListResponse(teams=teams, count=len(teams))


@router.get("/{team_name}", response_model=TeamResponse)
async def get_team(team_name: str):
    """
    Get a team's overall record and recent form.

    The team name must match exactly (case-sensitive).
    """
    predictor = get_predictor()
    available = predictor.get_available_teams()

    if team_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Team '{team_name}' not found. Check /teams for available names.",
        )

    df = predictor.assembler.historical_df

    # Home record
    home_matches = df[df["HomeTeam"] == team_name]
    home_record = {
        "wins": int((home_matches["FTR"] == "H").sum()),
        "draws": int((home_matches["FTR"] == "D").sum()),
        "losses": int((home_matches["FTR"] == "A").sum()),
    }

    # Away record
    away_matches = df[df["AwayTeam"] == team_name]
    away_record = {
        "wins": int((away_matches["FTR"] == "A").sum()),
        "draws": int((away_matches["FTR"] == "D").sum()),
        "losses": int((away_matches["FTR"] == "H").sum()),
    }

    total_matches = len(home_matches) + len(away_matches)

    # Recent form (last 5 matches overall, as W/D/L string)
    all_matches = df[
        (df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)
    ].sort_values("Date").tail(5)

    form_chars = []
    for _, row in all_matches.iterrows():
        if row["HomeTeam"] == team_name:
            form_chars.append(
                "W" if row["FTR"] == "H" else ("D" if row["FTR"] == "D" else "L")
            )
        else:
            form_chars.append(
                "W" if row["FTR"] == "A" else ("D" if row["FTR"] == "D" else "L")
            )

    recent_form = "".join(form_chars) if form_chars else None

    return TeamResponse(
        name=team_name,
        matches_played=total_matches,
        home_record=home_record,
        away_record=away_record,
        recent_form=recent_form,
    )
