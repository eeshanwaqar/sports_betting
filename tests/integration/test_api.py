"""
API Integration Tests.

Uses FastAPI's TestClient (via httpx) to exercise API endpoints
without starting a real server. The MatchPredictor is mocked to
avoid needing actual model artifacts.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_predictor():
    """
    Create a mock MatchPredictor that returns deterministic predictions.

    This avoids loading real model files, historical data, or MLflow.
    """
    predictor = MagicMock()

    predictor.get_available_teams.return_value = [
        "Arsenal", "Chelsea", "Liverpool", "Man City", "Man United",
    ]

    predictor.predict.return_value = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "prediction": "H",
        "probabilities": {"home_win": 0.55, "draw": 0.25, "away_win": 0.20},
        "odds": {"home_win": 1.73, "draw": 3.81, "away_win": 4.76},
        "confidence": 0.55,
    }

    # Mock the assembler with a small DataFrame for teams/matches routes
    import pandas as pd
    mock_df = pd.DataFrame({
        "Date": pd.to_datetime(["2023-10-01", "2023-10-08", "2023-10-15"]),
        "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
        "AwayTeam": ["Chelsea", "Liverpool", "Liverpool"],
        "FTHG": [2, 1, 3],
        "FTAG": [1, 1, 0],
        "FTR": ["H", "D", "H"],
    })
    predictor.assembler.historical_df = mock_df

    # Mock model loader info
    predictor.loader.model_info = {"model_type": "GradientBoosting"}

    return predictor


@pytest.fixture
def client(mock_predictor):
    """
    FastAPI TestClient with the MatchPredictor dependency overridden.
    """
    with patch("src.api.dependencies._predictor_instance", mock_predictor):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_returns_api_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "EPL" in data["name"]
        assert "endpoints" in data


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    def test_health_liveness(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_readiness(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["model_type"] == "GradientBoosting"


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class TestPredictionEndpoints:
    def test_predict_single_match(self, client):
        response = client.post("/predict", json={
            "home_team": "Arsenal",
            "away_team": "Chelsea",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ("H", "D", "A")
        assert "probabilities" in data
        assert "odds" in data
        assert 0 <= data["confidence"] <= 1

    def test_predict_unknown_team_returns_422(self, client):
        response = client.post("/predict", json={
            "home_team": "NonExistentFC",
            "away_team": "Chelsea",
        })
        assert response.status_code == 422

    def test_predict_same_team_returns_422(self, client):
        response = client.post("/predict", json={
            "home_team": "Arsenal",
            "away_team": "Arsenal",
        })
        assert response.status_code == 422

    def test_predict_batch(self, client):
        response = client.post("/predict/batch", json={
            "matches": [
                {"home_team": "Arsenal", "away_team": "Chelsea"},
                {"home_team": "Liverpool", "away_team": "Man City"},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_predict_with_date(self, client):
        response = client.post("/predict", json={
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "match_date": "2024-03-15",
        })
        assert response.status_code == 200

    def test_probabilities_structure(self, client):
        response = client.post("/predict", json={
            "home_team": "Arsenal",
            "away_team": "Chelsea",
        })
        probs = response.json()["probabilities"]
        assert "home_win" in probs
        assert "draw" in probs
        assert "away_win" in probs
        # Each probability should be between 0 and 1
        for key in probs:
            assert 0 <= probs[key] <= 1


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

class TestTeamEndpoints:
    def test_list_teams(self, client):
        response = client.get("/teams")
        assert response.status_code == 200
        data = response.json()
        assert "teams" in data
        assert data["count"] == len(data["teams"])
        assert "Arsenal" in data["teams"]

    def test_get_team_details(self, client):
        response = client.get("/teams/Arsenal")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Arsenal"
        assert "home_record" in data
        assert "away_record" in data
        assert "matches_played" in data

    def test_get_unknown_team_returns_404(self, client):
        response = client.get("/teams/NonExistentFC")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

class TestMatchEndpoints:
    def test_recent_matches(self, client):
        response = client.get("/matches/recent")
        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
        assert "count" in data

    def test_recent_matches_filter_by_team(self, client):
        response = client.get("/matches/recent?team=Arsenal")
        assert response.status_code == 200

    def test_recent_matches_unknown_team_returns_404(self, client):
        response = client.get("/matches/recent?team=NonExistentFC")
        assert response.status_code == 404

    def test_head_to_head(self, client):
        response = client.get("/matches/head-to-head?team_a=Arsenal&team_b=Chelsea")
        assert response.status_code == 200
        data = response.json()
        assert data["team_a"] == "Arsenal"
        assert data["team_b"] == "Chelsea"
        assert "summary" in data
        assert "recent_matches" in data

    def test_head_to_head_unknown_team(self, client):
        response = client.get(
            "/matches/head-to-head?team_a=Arsenal&team_b=NonExistent"
        )
        assert response.status_code == 404
