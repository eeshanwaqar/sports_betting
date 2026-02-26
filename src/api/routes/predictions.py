"""
Predictions Routes - Match prediction API endpoints.

Endpoints:
    POST /predict        - Single match prediction
    POST /predict/batch  - Batch predictions (up to 20 matches)
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ProbabilityResponse,
    OddsResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from src.api.dependencies import get_predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Predictions"])


def _raw_to_response(raw: dict) -> PredictionResponse:
    """Convert predictor dict output to typed Pydantic response."""
    return PredictionResponse(
        home_team=raw["home_team"],
        away_team=raw["away_team"],
        prediction=raw["prediction"],
        probabilities=ProbabilityResponse(**raw["probabilities"]),
        odds=OddsResponse(**raw["odds"]),
        confidence=raw["confidence"],
    )


@router.post("", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest):
    """
    Predict a single match outcome.

    Returns probabilities, decimal odds, and the predicted result.
    """
    predictor = get_predictor()

    # Validate teams exist in historical data
    available_teams = predictor.get_available_teams()
    for team, label in [
        (request.home_team, "Home team"),
        (request.away_team, "Away team"),
    ]:
        if team not in available_teams:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{label} '{team}' not found. "
                    f"Available teams: {', '.join(available_teams[:10])}..."
                ),
            )

    if request.home_team == request.away_team:
        raise HTTPException(
            status_code=422,
            detail="Home team and away team must be different",
        )

    match_date = (
        datetime.combine(request.match_date, datetime.min.time())
        if request.match_date
        else None
    )

    try:
        raw = predictor.predict(
            request.home_team,
            request.away_team,
            match_date,
        )
        return _raw_to_response(raw)
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict multiple matches in a single request.

    Accepts up to 20 matches. Each match is predicted independently.
    """
    predictor = get_predictor()
    available_teams = predictor.get_available_teams()
    results = []

    for match in request.matches:
        # Validate
        for team in [match.home_team, match.away_team]:
            if team not in available_teams:
                raise HTTPException(
                    status_code=422,
                    detail=f"Team '{team}' not found in historical data",
                )

        match_date = (
            datetime.combine(match.match_date, datetime.min.time())
            if match.match_date
            else None
        )

        try:
            raw = predictor.predict(match.home_team, match.away_team, match_date)
            results.append(_raw_to_response(raw))
        except Exception as exc:
            logger.error(f"Batch prediction failed for {match}: {exc}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed for {match.home_team} vs {match.away_team}: {exc}",
            )

    return BatchPredictionResponse(predictions=results, count=len(results))
