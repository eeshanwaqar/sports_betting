"""
API Schemas - Pydantic request/response models for the REST API.

Defines typed contracts for all API endpoints.
"""

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# --- Request Models ---

class PredictionRequest(BaseModel):
    """Request body for a single match prediction."""
    home_team: str = Field(..., description="Home team name", examples=["Arsenal"])
    away_team: str = Field(..., description="Away team name", examples=["Chelsea"])
    match_date: Optional[date] = Field(
        None, description="Match date (defaults to today)"
    )


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""
    matches: List[PredictionRequest] = Field(
        ..., min_length=1, max_length=20, description="List of matches to predict"
    )


# --- Response Models ---

class ProbabilityResponse(BaseModel):
    """Predicted probabilities for each outcome."""
    home_win: float = Field(..., ge=0, le=1)
    draw: float = Field(..., ge=0, le=1)
    away_win: float = Field(..., ge=0, le=1)


class OddsResponse(BaseModel):
    """Decimal betting odds for each outcome."""
    home_win: float = Field(..., gt=1)
    draw: float = Field(..., gt=1)
    away_win: float = Field(..., gt=1)


class PredictionResponse(BaseModel):
    """Full prediction response for a single match."""
    home_team: str
    away_team: str
    prediction: str = Field(..., description="Predicted outcome: H, D, or A")
    probabilities: ProbabilityResponse
    odds: OddsResponse
    confidence: float = Field(..., ge=0, le=1)


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    count: int


class TeamResponse(BaseModel):
    """Team information response."""
    name: str
    matches_played: int
    home_record: Dict[str, int]
    away_record: Dict[str, int]
    recent_form: Optional[str] = None


class TeamListResponse(BaseModel):
    """List of available teams."""
    teams: List[str]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    error_code: Optional[str] = None
