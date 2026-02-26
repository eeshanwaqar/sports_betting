"""
Tests for src.features.h2h module.

Covers: calc_h2h (head-to-head statistics from home team's perspective).
"""

import pytest
import numpy as np
import pandas as pd

from src.features.h2h import calc_h2h


class TestCalcH2H:
    """Tests for head-to-head feature calculations."""

    def test_no_prior_meetings(self, sample_matches_df):
        """Two teams that never met should return h2h_played=0."""
        result = calc_h2h(
            sample_matches_df,
            home_team="Arsenal",
            away_team="Chelsea",
            before_date=pd.Timestamp("2023-08-01"),
        )
        assert result["h2h_played"] == 0
        assert np.isnan(result["h2h_win_rate"])
        assert np.isnan(result["h2h_avg_gf"])

    def test_with_prior_meetings(self, sample_matches_df, late_cutoff_date):
        """Arsenal vs Chelsea have several meetings in the sample data."""
        result = calc_h2h(
            sample_matches_df,
            home_team="Arsenal",
            away_team="Chelsea",
            before_date=late_cutoff_date,
        )
        assert result["h2h_played"] > 0
        assert 0.0 <= result["h2h_win_rate"] <= 1.0
        assert result["h2h_avg_gf"] >= 0

    def test_win_rate_is_from_home_team_perspective(self, sample_matches_df, late_cutoff_date):
        """Swapping home/away should give complementary results."""
        result_a_home = calc_h2h(
            sample_matches_df, "Arsenal", "Chelsea", late_cutoff_date,
        )
        result_c_home = calc_h2h(
            sample_matches_df, "Chelsea", "Arsenal", late_cutoff_date,
        )
        # Same number of meetings regardless of perspective
        assert result_a_home["h2h_played"] == result_c_home["h2h_played"]

        # Win rates should sum to <= 1 (draws account for the remainder)
        total = result_a_home["h2h_win_rate"] + result_c_home["h2h_win_rate"]
        assert total <= 1.0 + 1e-9  # Small float tolerance

    def test_window_limits_matches(self, sample_matches_df, late_cutoff_date):
        """n=2 should only consider the last 2 meetings."""
        result_full = calc_h2h(
            sample_matches_df, "Arsenal", "Chelsea", late_cutoff_date, n=50,
        )
        result_2 = calc_h2h(
            sample_matches_df, "Arsenal", "Chelsea", late_cutoff_date, n=2,
        )
        assert result_2["h2h_played"] <= 2
        assert result_2["h2h_played"] <= result_full["h2h_played"]

    def test_no_future_leakage(self, sample_matches_df):
        """
        Using a cutoff between matches should exclude matches after that date.
        First Arsenal-Chelsea match is Aug 12, second is Sep 23.
        """
        result = calc_h2h(
            sample_matches_df, "Arsenal", "Chelsea",
            before_date=pd.Timestamp("2023-09-01"),
        )
        # Only the Aug 12 match should be included
        assert result["h2h_played"] == 1

    def test_unknown_teams_return_zero_played(self, sample_matches_df, late_cutoff_date):
        """Teams not in the data should get h2h_played=0."""
        result = calc_h2h(
            sample_matches_df, "Tottenham", "Everton", late_cutoff_date,
        )
        assert result["h2h_played"] == 0
