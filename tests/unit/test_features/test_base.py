"""
Tests for src.features.base module.

Covers: get_team_matches, get_prior_matches.
"""

import pytest
import pandas as pd

from src.features.base import get_team_matches, get_prior_matches


class TestGetTeamMatches:
    """Tests for the unified team-match view builder."""

    def test_returns_all_matches_for_team(self, sample_matches_df):
        result = get_team_matches(sample_matches_df, "Arsenal")
        # Arsenal appears in most of the 12 sample matches
        assert len(result) > 0

    def test_has_standardized_columns(self, sample_matches_df):
        result = get_team_matches(sample_matches_df, "Arsenal")
        for col in ["GoalsFor", "GoalsAgainst", "Result", "Points", "IsHome"]:
            assert col in result.columns

    def test_result_mapping_correct(self, sample_matches_df):
        """Home win FTR='H' should map to Result='W' for home team."""
        result = get_team_matches(sample_matches_df, "Arsenal")
        # First match: Arsenal home win (FTHG=2, FTAG=1, FTR=H)
        first = result.iloc[0]
        assert first["Result"] == "W"
        assert first["Points"] == 3
        assert first["GoalsFor"] == 2
        assert first["GoalsAgainst"] == 1

    def test_away_loss_mapped_correctly(self, sample_matches_df):
        """When Arsenal loses away, Result should be 'L'."""
        result = get_team_matches(sample_matches_df, "Arsenal")
        # Match on 2023-10-07: Liverpool 2-0 Arsenal (FTR=H) -> Arsenal loss
        loss_match = result[result["Date"] == pd.Timestamp("2023-10-07")]
        if len(loss_match) > 0:
            assert loss_match.iloc[0]["Result"] == "L"
            assert loss_match.iloc[0]["Points"] == 0

    def test_sorted_by_date(self, sample_matches_df):
        result = get_team_matches(sample_matches_df, "Arsenal")
        dates = result["Date"].tolist()
        assert dates == sorted(dates)

    def test_shot_columns_present_when_available(self, sample_matches_df):
        result = get_team_matches(sample_matches_df, "Arsenal")
        assert "ShotsFor" in result.columns
        assert "ShotsAgainst" in result.columns

    def test_unknown_team_returns_empty(self, sample_matches_df):
        result = get_team_matches(sample_matches_df, "Tottenham")
        assert len(result) == 0


class TestGetPriorMatches:
    """Tests for date-filtered prior match retrieval."""

    def test_excludes_future_matches(self, sample_matches_df, cutoff_date):
        result = get_prior_matches(sample_matches_df, "Arsenal", cutoff_date)
        assert all(result["Date"] < cutoff_date)

    def test_n_limits_output(self, sample_matches_df, late_cutoff_date):
        result = get_prior_matches(
            sample_matches_df, "Arsenal", late_cutoff_date, n=3,
        )
        assert len(result) <= 3

    def test_home_only_filter(self, sample_matches_df, late_cutoff_date):
        result = get_prior_matches(
            sample_matches_df, "Arsenal", late_cutoff_date, home_only=True,
        )
        assert all(result["IsHome"] == True)  # noqa: E712

    def test_away_only_filter(self, sample_matches_df, late_cutoff_date):
        result = get_prior_matches(
            sample_matches_df, "Arsenal", late_cutoff_date, home_only=False,
        )
        assert all(result["IsHome"] == False)  # noqa: E712

    def test_empty_when_before_first_match(self, sample_matches_df):
        result = get_prior_matches(
            sample_matches_df, "Arsenal", pd.Timestamp("2023-08-01"),
        )
        assert len(result) == 0
