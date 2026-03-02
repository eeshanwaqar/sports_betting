"""
Tests for src.features.team_stats module.

Covers: calc_season_stats, calc_shooting_stats.
"""

import numpy as np
import pandas as pd

from src.features.team_stats import calc_season_stats, calc_shooting_stats

# ---------------------------------------------------------------------------
# calc_season_stats
# ---------------------------------------------------------------------------

class TestCalcSeasonStats:
    """Tests for season-to-date statistic features."""

    def test_no_prior_matches_returns_nan(self, sample_matches_df):
        """Before the first match, everything should be NaN/0."""
        result = calc_season_stats(
            sample_matches_df, "Arsenal",
            before_date=pd.Timestamp("2023-08-01"),
        )
        assert result["season_played"] == 0
        assert np.isnan(result["season_win_rate"])
        assert np.isnan(result["season_ppg"])

    def test_returns_expected_keys(self, sample_matches_df, cutoff_date):
        result = calc_season_stats(sample_matches_df, "Arsenal", cutoff_date)
        expected_keys = {
            "season_played", "season_win_rate", "season_ppg",
            "season_avg_goals_for", "season_avg_goals_against",
            "season_clean_sheets",
        }
        assert set(result.keys()) == expected_keys

    def test_win_rate_bounded(self, sample_matches_df, cutoff_date):
        result = calc_season_stats(sample_matches_df, "Arsenal", cutoff_date)
        if not np.isnan(result["season_win_rate"]):
            assert 0.0 <= result["season_win_rate"] <= 1.0

    def test_ppg_bounded(self, sample_matches_df, cutoff_date):
        """Points per game must be between 0 and 3."""
        result = calc_season_stats(sample_matches_df, "Arsenal", cutoff_date)
        if not np.isnan(result["season_ppg"]):
            assert 0.0 <= result["season_ppg"] <= 3.0

    def test_clean_sheets_bounded(self, sample_matches_df, cutoff_date):
        result = calc_season_stats(sample_matches_df, "Arsenal", cutoff_date)
        if not np.isnan(result["season_clean_sheets"]):
            assert 0.0 <= result["season_clean_sheets"] <= 1.0

    def test_season_played_increases(self, sample_matches_df):
        """More matches should be counted at a later cutoff."""
        early = calc_season_stats(
            sample_matches_df, "Arsenal", pd.Timestamp("2023-09-01"),
        )
        late = calc_season_stats(
            sample_matches_df, "Arsenal", pd.Timestamp("2023-12-01"),
        )
        assert late["season_played"] >= early["season_played"]


# ---------------------------------------------------------------------------
# calc_shooting_stats
# ---------------------------------------------------------------------------

class TestCalcShootingStats:
    """Tests for shot-based feature calculations."""

    def test_no_prior_matches_returns_nan(self, sample_matches_df):
        result = calc_shooting_stats(
            sample_matches_df, "Arsenal",
            before_date=pd.Timestamp("2023-08-01"),
        )
        assert np.isnan(result["avg_shots"])
        assert np.isnan(result["shot_accuracy"])

    def test_returns_expected_keys(self, sample_matches_df, cutoff_date):
        result = calc_shooting_stats(sample_matches_df, "Arsenal", cutoff_date)
        expected_keys = {"avg_shots", "avg_sot", "shot_accuracy", "shot_conversion"}
        assert set(result.keys()) == expected_keys

    def test_shot_accuracy_bounded(self, sample_matches_df, cutoff_date):
        result = calc_shooting_stats(sample_matches_df, "Arsenal", cutoff_date)
        if not np.isnan(result["shot_accuracy"]):
            assert 0.0 <= result["shot_accuracy"] <= 1.0

    def test_shot_conversion_bounded(self, sample_matches_df, cutoff_date):
        result = calc_shooting_stats(sample_matches_df, "Arsenal", cutoff_date)
        if not np.isnan(result["shot_conversion"]):
            assert 0.0 <= result["shot_conversion"] <= 1.0

    def test_avg_shots_positive(self, sample_matches_df, cutoff_date):
        result = calc_shooting_stats(sample_matches_df, "Arsenal", cutoff_date)
        if not np.isnan(result["avg_shots"]):
            assert result["avg_shots"] > 0

    def test_without_shot_columns(self):
        """DataFrame without HS/AS/HST/AST columns should return NaN."""
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-09-01", "2023-09-15"]),
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Arsenal"],
            "FTHG": [2, 1], "FTAG": [0, 0], "FTR": ["H", "H"],
        })
        result = calc_shooting_stats(df, "Arsenal", pd.Timestamp("2023-10-01"))
        assert np.isnan(result["avg_shots"])
