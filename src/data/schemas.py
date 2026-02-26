"""
Data Schemas - Pydantic models for data validation.

Used for API request/response validation and data contracts.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class MatchInput(BaseModel):
    """Input schema for a match prediction request."""
    home_team: str = Field(..., description="Home team name", examples=["Arsenal"])
    away_team: str = Field(..., description="Away team name", examples=["Chelsea"])
    date: Optional[date] = Field(None, description="Match date (optional)")


class ProbabilityOutput(BaseModel):
    """Predicted probabilities for each outcome."""
    home_win: float = Field(..., ge=0, le=1)
    draw: float = Field(..., ge=0, le=1)
    away_win: float = Field(..., ge=0, le=1)


class OddsOutput(BaseModel):
    """Decimal betting odds for each outcome."""
    home_win: float = Field(..., gt=1)
    draw: float = Field(..., gt=1)
    away_win: float = Field(..., gt=1)


class PredictionOutput(BaseModel):
    """Full prediction response."""
    home_team: str
    away_team: str
    prediction: str = Field(..., description="Predicted outcome: H, D, or A")
    probabilities: ProbabilityOutput
    odds: OddsOutput
    confidence: float = Field(..., ge=0, le=1)


class MatchRecord(BaseModel):
    """Schema for a historical match record."""
    date: date
    home_team: str
    away_team: str
    home_goals: int = Field(..., ge=0)
    away_goals: int = Field(..., ge=0)
    result: str = Field(..., pattern="^[HDA]$")
    season: str
