"""
Feature Assembler - Assemble features for a single match prediction.

This bridges the FeatureBuilder (training-oriented) with live inference.
It constructs a feature vector for a new match that hasn't been played yet.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from src.features.builder import FeatureBuilder
from src.utils.config import FeatureConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureAssembler:
    """
    Assembles features for live/upcoming match prediction.

    Wraps FeatureBuilder to create feature vectors for matches
    that haven't been played yet.

    Usage:
        assembler = FeatureAssembler(historical_df, feature_names, config)
        features = assembler.assemble("Arsenal", "Chelsea")
    """

    def __init__(
        self,
        historical_df: pd.DataFrame,
        feature_names: List[str],
        config: Optional[FeatureConfig] = None,
        train_medians: Optional[dict] = None,
    ):
        """
        Args:
            historical_df: All historical match data.
            feature_names: Ordered list of feature column names (from features.txt).
            config: Feature configuration.
            train_medians: Training set feature medians for NaN imputation.
                Features unavailable at inference (e.g. odds) are filled with
                training medians instead of 0.0 to avoid model distortion.
        """
        self.historical_df = historical_df
        self.feature_names = feature_names
        self.train_medians = train_medians or {}
        self.builder = FeatureBuilder(historical_df, config)

    def assemble(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[datetime] = None,
    ) -> np.ndarray:
        """
        Assemble a feature vector for an upcoming match.

        Creates a synthetic row and runs it through the FeatureBuilder.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            match_date: Match date (defaults to now).

        Returns:
            1D numpy array of feature values in the correct order.

        Raises:
            ValueError: If a team is not found in historical data.
        """
        if match_date is None:
            match_date = datetime.now()

        date = pd.Timestamp(match_date)

        # Validate teams exist in data
        all_teams = set(self.historical_df["HomeTeam"].unique()) | set(
            self.historical_df["AwayTeam"].unique()
        )
        for team in [home_team, away_team]:
            if team not in all_teams:
                raise ValueError(
                    f"Unknown team: '{team}'. Available: {sorted(all_teams)}"
                )

        # Build a synthetic match row
        row = pd.Series({
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Date": date,
            "Season": self._infer_season(date),
            "FTHG": 0,  # Not played yet
            "FTAG": 0,
            "FTR": None,
            "B365H": np.nan,  # No pre-match odds for future match
            "B365D": np.nan,
            "B365A": np.nan,
        })

        # Build features
        features = self.builder.build_match(row)

        # Extract in correct order. Use training medians as fallback for
        # missing/NaN features (e.g. betting odds unavailable for future
        # matches). This prevents sending 0.0 which is far out-of-distribution
        # and causes systematic prediction bias.
        vector = np.array([
            features.get(f, self.train_medians.get(f, 0.0))
            for f in self.feature_names
        ], dtype=float)

        # Replace any remaining NaN with training median, then 0.0 as last resort
        for i, fname in enumerate(self.feature_names):
            if np.isnan(vector[i]):
                vector[i] = self.train_medians.get(fname, 0.0)

        return vector

    def _infer_season(self, date: pd.Timestamp) -> str:
        """Infer season label from date."""
        if date.month >= 8:
            return f"{date.year}-{str(date.year + 1)[-2:]}"
        return f"{date.year - 1}-{str(date.year)[-2:]}"
