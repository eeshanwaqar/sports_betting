"""
Predictor - Core prediction logic for live match predictions.

This is the main entry point for the API.
Orchestrates: model loading → feature assembly → inference → odds conversion.
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

from src.inference.model_loader import ModelLoader
from src.inference.feature_assembler import FeatureAssembler
from src.inference.odds_calculator import probs_to_odds
from src.data.loader import load_raw_matches
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MatchPredictor:
    """
    End-to-end match prediction pipeline.

    Usage:
        predictor = MatchPredictor(config)
        result = predictor.predict("Arsenal", "Chelsea")

    The predictor loads models and data lazily on first use.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            from src.utils.config import load_config
            config = load_config()

        self.config = config
        self._loader: Optional[ModelLoader] = None
        self._assembler: Optional[FeatureAssembler] = None

    @property
    def loader(self) -> ModelLoader:
        """Model artifacts (lazy loaded)."""
        if self._loader is None:
            self._loader = ModelLoader(self.config.model.models_dir)
        return self._loader

    @property
    def assembler(self) -> FeatureAssembler:
        """Feature assembler (lazy loaded with historical data)."""
        if self._assembler is None:
            historical_df = load_raw_matches(self.config.data.raw)
            self._assembler = FeatureAssembler(
                historical_df,
                self.loader.feature_names,
                self.config.features,
                self.loader.train_medians,
            )
        return self._assembler

    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[datetime] = None,
    ) -> dict:
        """
        Predict match outcome with probabilities and odds.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            match_date: Match date (defaults to now).

        Returns:
            Dict with prediction, probabilities, odds, confidence.
        """
        logger.info(f"Predicting: {home_team} vs {away_team}")

        # 1. Assemble features
        features = self.assembler.assemble(home_team, away_team, match_date)
        features_2d = features.reshape(1, -1)

        # 2. Scale if model is linear
        model_type = self.loader.model_info.get("model_type", "")
        if "Logistic" in model_type:
            features_2d = self.loader.scaler.transform(features_2d)

        # 3. Predict with calibrated probabilities (falls back to raw if no calibrator)
        if self.loader.calibrator is not None:
            proba = self.loader.calibrator.predict_proba(features_2d)[0]
        else:
            proba = self.loader.model.predict_proba(features_2d)[0]
        pred_idx = np.argmax(proba)
        pred_label = self.loader.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba.max())

        # Map probabilities to H/D/A
        classes = self.loader.label_encoder.classes_
        prob_map = {cls: float(proba[i]) for i, cls in enumerate(classes)}

        # 4. Convert to odds
        odds = probs_to_odds(prob_map["H"], prob_map["D"], prob_map["A"])

        logger.info(
            f"  Prediction: {pred_label} ({confidence:.1%}), "
            f"H={prob_map['H']:.1%}, D={prob_map['D']:.1%}, A={prob_map['A']:.1%}"
        )

        return {
            "home_team": home_team,
            "away_team": away_team,
            "prediction": pred_label,
            "probabilities": {
                "home_win": prob_map["H"],
                "draw": prob_map["D"],
                "away_win": prob_map["A"],
            },
            "odds": odds,
            "confidence": confidence,
        }

    def get_available_teams(self) -> list:
        """Return list of all teams in historical data."""
        df = self.assembler.historical_df
        teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
        return sorted(teams)
