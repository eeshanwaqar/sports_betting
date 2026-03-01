"""
Feature Builder - Orchestrator that combines all feature modules.

Maps to: notebooks/03_feature_engineering.ipynb (build_match_features + build loop)

This is the central module that:
1. Pre-computes expensive state (Elo ratings) once
2. Builds features for a single match (used by both training and inference)
3. Builds features for all matches (training mode)
4. Selects model-ready feature columns
5. Cleans and saves output

Key Principle: No data leakage — every feature uses only pre-match data.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from src.features.elo import build_elo_ratings, calc_elo_features
from src.features.form import calc_exp_form, calc_form, calc_streaks, calc_venue_form
from src.features.h2h import calc_h2h
from src.features.odds import calc_odds_features
from src.features.team_stats import calc_season_stats, calc_shooting_stats
from src.features.temporal import calc_temporal_features
from src.utils.config import AppConfig, FeatureConfig
from src.utils.constants import META_COLUMNS, TARGET_COLUMNS
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


# --- Model Feature Selection ---
# These are the columns selected for the model (from notebook analysis)
MODEL_FEATURES: List[str] = [
    # Elo (3)
    "home_elo", "away_elo", "elo_diff",
    # Betting Odds (5)
    "odds_prob_home", "odds_prob_draw", "odds_prob_away",
    "odds_home_away_diff", "odds_fav_is_home",
    # Form - Last 5 (6)
    "home_form5_points", "home_form5_win_rate", "home_form5_goal_diff",
    "away_form5_points", "away_form5_win_rate", "away_form5_goal_diff",
    # Form - Last 3 (4)
    "home_form3_win_rate", "home_form3_goal_diff",
    "away_form3_win_rate", "away_form3_goal_diff",
    # Exponential Form (2)
    "home_exp_form", "away_exp_form",
    # Venue Form (6)
    "home_venue_win_rate", "home_venue_avg_goals_for", "home_venue_clean_sheets",
    "away_venue_win_rate", "away_venue_avg_goals_for", "away_venue_clean_sheets",
    # Season Stats (8)
    "home_season_win_rate", "home_season_ppg",
    "home_season_avg_goals_for", "home_season_clean_sheets",
    "away_season_win_rate", "away_season_ppg",
    "away_season_avg_goals_for", "away_season_clean_sheets",
    # Streaks (4)
    "home_win_streak", "home_unbeaten_streak",
    "away_win_streak", "away_unbeaten_streak",
    # Shooting (4)
    "home_shot_accuracy", "home_shot_conversion",
    "away_shot_accuracy", "away_shot_conversion",
    # H2H (2)
    "h2h_win_rate", "h2h_avg_gf",
    # Difference / Mismatch (12)
    "diff_form5_points", "diff_form5_goal_diff", "diff_form5_win_rate",
    "diff_season_win_rate", "diff_season_ppg", "diff_season_goals_for",
    "diff_venue_win_rate", "diff_venue_goals",
    "home_attack_vs_away_defense", "away_attack_vs_home_defense",
    "diff_exp_form", "diff_win_streak",
    # Temporal (3)
    "is_weekend", "month", "matchweek",
]


class FeatureBuilder:
    """
    Builds features for match outcome prediction.

    Usage (training):
        builder = FeatureBuilder(df, config)
        features_df = builder.build_all()
        model_df = builder.get_model_ready(features_df)

    Usage (single match inference):
        builder = FeatureBuilder(df, config)
        features = builder.build_match(row)
    """

    def __init__(self, df: pd.DataFrame, config: Optional[FeatureConfig] = None):
        """
        Args:
            df: Full match DataFrame (sorted by date).
            config: Feature configuration. Uses defaults if None.
        """
        self.df = df.sort_values("Date").reset_index(drop=True)
        self.config = config or FeatureConfig()

        # Pre-compute Elo ratings (expensive, do once)
        logger.info("Pre-computing Elo ratings...")
        self.elo_history, self.current_elo_ratings = build_elo_ratings(
            self.df,
            k=self.config.elo.k_factor,
            home_advantage=self.config.elo.home_advantage,
            initial_elo=self.config.elo.initial_rating,
        )

    def build_match(self, row: pd.Series) -> dict:
        """
        Build all features for a single match. Only uses pre-match data.

        This is the core function used by both training and inference.

        Args:
            row: A single match row from the DataFrame.

        Returns:
            Dict of all features for this match.
        """
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]
        df = self.df
        cfg = self.config

        features: dict = {
            "match_id": f"{home}_vs_{away}_{date.strftime('%Y%m%d')}",
            "date": date,
            "home_team": home,
            "away_team": away,
            "season": row["Season"],
        }

        # 1. Elo Ratings (falls back to latest known rating for future matches)
        features.update(calc_elo_features(
            row, self.elo_history, cfg.elo.initial_rating, self.current_elo_ratings
        ))

        # 2. Betting Odds
        features.update(calc_odds_features(row))

        # 3. Multi-Window Form
        for n in cfg.form_windows:
            home_form = calc_form(df, home, date, n=n)
            for k, v in home_form.items():
                features[f"home_{k}"] = v

            away_form = calc_form(df, away, date, n=n)
            for k, v in away_form.items():
                features[f"away_{k}"] = v

        # 4. Exponential Form
        home_exp = calc_exp_form(df, home, date, n=5, decay=cfg.exp_decay)
        features["home_exp_form"] = home_exp["exp_form"]
        away_exp = calc_exp_form(df, away, date, n=5, decay=cfg.exp_decay)
        features["away_exp_form"] = away_exp["exp_form"]

        # 5. Venue-Specific Form
        home_venue = calc_venue_form(df, home, date, is_home=True, n=cfg.venue_window)
        for k, v in home_venue.items():
            features[f"home_{k}"] = v

        away_venue = calc_venue_form(df, away, date, is_home=False, n=cfg.venue_window)
        for k, v in away_venue.items():
            features[f"away_{k}"] = v

        # 6. Season Stats
        home_season = calc_season_stats(df, home, date)
        for k, v in home_season.items():
            features[f"home_{k}"] = v

        away_season = calc_season_stats(df, away, date)
        for k, v in away_season.items():
            features[f"away_{k}"] = v

        # 7. Streaks
        home_streaks = calc_streaks(df, home, date)
        for k, v in home_streaks.items():
            features[f"home_{k}"] = v

        away_streaks = calc_streaks(df, away, date)
        for k, v in away_streaks.items():
            features[f"away_{k}"] = v

        # 8. Shooting Stats
        home_shots = calc_shooting_stats(df, home, date, n=cfg.shooting_window)
        for k, v in home_shots.items():
            features[f"home_{k}"] = v

        away_shots = calc_shooting_stats(df, away, date, n=cfg.shooting_window)
        for k, v in away_shots.items():
            features[f"away_{k}"] = v

        # 9. Head-to-Head
        h2h = calc_h2h(df, home, away, date, n=cfg.h2h_window)
        features.update(h2h)

        # 10. Difference / Mismatch Features
        features.update(self._calc_differences(features))

        # 11. Temporal
        features.update(calc_temporal_features(row))

        # 12. Target (only available for training, not inference)
        if "FTR" in row.index:
            features["target"] = row["FTR"]
            features["home_goals"] = row["FTHG"]
            features["away_goals"] = row["FTAG"]

        return features

    def _calc_differences(self, features: dict) -> dict:
        """
        Compute difference/mismatch features from already-computed features.

        These are the strongest predictors (EDA: top 4 correlated features are all diffs).
        """
        diffs: dict = {}

        if pd.isna(features.get("home_form5_points")) or pd.isna(features.get("away_form5_points")):
            return diffs

        # Form differences
        diffs["diff_form5_points"] = features["home_form5_points"] - features["away_form5_points"]
        diffs["diff_form5_goal_diff"] = features["home_form5_goal_diff"] - features["away_form5_goal_diff"]
        diffs["diff_form5_win_rate"] = features["home_form5_win_rate"] - features["away_form5_win_rate"]

        # Season differences
        diffs["diff_season_win_rate"] = features.get("home_season_win_rate", 0) - features.get("away_season_win_rate", 0)
        diffs["diff_season_ppg"] = features.get("home_season_ppg", 0) - features.get("away_season_ppg", 0)
        diffs["diff_season_goals_for"] = features.get("home_season_avg_goals_for", 0) - features.get("away_season_avg_goals_for", 0)

        # Venue differences
        diffs["diff_venue_win_rate"] = features.get("home_venue_win_rate", 0) - features.get("away_venue_win_rate", 0)
        diffs["diff_venue_goals"] = features.get("home_venue_avg_goals_for", 0) - features.get("away_venue_avg_goals_for", 0)

        # Attack vs Defense mismatch
        diffs["home_attack_vs_away_defense"] = features.get("home_season_avg_goals_for", 0) - features.get("away_season_avg_goals_against", 0)
        diffs["away_attack_vs_home_defense"] = features.get("away_season_avg_goals_for", 0) - features.get("home_season_avg_goals_against", 0)

        # Exponential form diff
        diffs["diff_exp_form"] = features.get("home_exp_form", 0) - features.get("away_exp_form", 0)

        # Streak diff
        diffs["diff_win_streak"] = features.get("home_win_streak", 0) - features.get("away_win_streak", 0)

        return diffs

    def build_all(self) -> pd.DataFrame:
        """
        Build features for ALL matches in the dataset (training mode).

        Returns:
            DataFrame with all features.
        """
        total = len(self.df)
        logger.info(f"Building features for {total} matches...")

        all_features = []
        for i, (_, row) in enumerate(self.df.iterrows()):
            features = self.build_match(row)
            all_features.append(features)

            if (i + 1) % 500 == 0:
                logger.info(f"  Progress: {i + 1}/{total} ({(i + 1) / total * 100:.0f}%)")

        features_df = pd.DataFrame(all_features)
        logger.info(f"Feature building complete: {features_df.shape}")
        return features_df

    def get_model_ready(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean features and select model-ready columns.

        1. Drop rows with missing core features (early season)
        2. Fill remaining NaN with median
        3. Select model features + meta + target

        Args:
            features_df: Raw features from build_all().

        Returns:
            Clean, model-ready DataFrame.
        """
        core = ["home_form5_points", "away_form5_points", "home_season_win_rate"]
        clean_df = features_df.dropna(subset=core).copy()

        logger.info(f"Dropped {len(features_df) - len(clean_df)} incomplete rows (early season)")

        # Fill remaining NaN with median
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if clean_df[col].isna().any():
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())

        # Select available model features
        available = [f for f in MODEL_FEATURES if f in clean_df.columns]
        missing = [f for f in MODEL_FEATURES if f not in clean_df.columns]
        if missing:
            logger.warning(f"Missing {len(missing)} features: {missing}")

        # Build output
        out_cols = META_COLUMNS + available + TARGET_COLUMNS
        out_cols = [c for c in out_cols if c in clean_df.columns]

        model_df = clean_df[out_cols]
        logger.info(f"Model-ready dataset: {model_df.shape[0]} matches, {len(available)} features")
        return model_df

    def save(
        self,
        features_df: pd.DataFrame,
        model_df: pd.DataFrame,
        output_dir: str = "data/features",
    ) -> None:
        """Save features and model-ready datasets to CSV."""
        ensure_dir(output_dir)

        features_path = os.path.join(output_dir, "features.csv")
        features_df.to_csv(features_path, index=False)
        logger.info(f"Saved all features: {features_path}")

        model_path = os.path.join(output_dir, "model_ready.csv")
        model_df.to_csv(model_path, index=False)
        logger.info(f"Saved model-ready: {model_path}")


def run_pipeline(config: Optional[AppConfig] = None) -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline.

    Loads clean matches → builds features → saves outputs.
    """
    from src.data.loader import load_raw_matches

    if config is None:
        from src.utils.config import load_config
        config = load_config()

    df = load_raw_matches(config.data.raw)
    builder = FeatureBuilder(df, config.features)

    features_df = builder.build_all()
    model_df = builder.get_model_ready(features_df)
    builder.save(features_df, model_df, config.data.features)

    return model_df
