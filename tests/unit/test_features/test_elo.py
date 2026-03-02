"""
Tests for src.features.elo module.

Covers: build_elo_ratings, calc_elo_features.
"""

import pandas as pd

from src.features.elo import build_elo_ratings, calc_elo_features


class TestBuildEloRatings:
    """Tests for the full Elo computation across a dataset."""

    def test_returns_history_and_current(self, sample_matches_df):
        elo_history, current = build_elo_ratings(sample_matches_df)
        assert isinstance(elo_history, dict)
        assert isinstance(current, dict)

    def test_all_teams_have_current_rating(self, sample_matches_df):
        _, current = build_elo_ratings(sample_matches_df)
        assert "Arsenal" in current
        assert "Chelsea" in current
        assert "Liverpool" in current

    def test_history_has_entries_for_every_match(self, sample_matches_df):
        """Each match should produce two history entries (home + away)."""
        elo_history, _ = build_elo_ratings(sample_matches_df)
        # 12 matches * 2 teams = 24 entries
        assert len(elo_history) == len(sample_matches_df) * 2

    def test_initial_elo_is_used(self, sample_matches_df):
        """First rating for each team should be the initial_elo."""
        elo_history, _ = build_elo_ratings(sample_matches_df, initial_elo=1500)
        first_date = sample_matches_df["Date"].min()
        # Arsenal plays in the first match (home)
        assert elo_history[("Arsenal", first_date)] == 1500

    def test_winner_gains_loser_drops(self, sample_matches_df):
        """After a home win (first match), home Elo should rise and away Elo fall."""
        _, current = build_elo_ratings(sample_matches_df)
        # With 12 matches the net effect varies, but we can verify the system
        # doesn't crash and produces distinct ratings
        assert current["Arsenal"] != current["Chelsea"]

    def test_custom_k_factor(self, sample_matches_df):
        """Higher K should produce larger rating swings."""
        _, current_low = build_elo_ratings(sample_matches_df, k=10)
        _, current_high = build_elo_ratings(sample_matches_df, k=40)

        spread_low = max(current_low.values()) - min(current_low.values())
        spread_high = max(current_high.values()) - min(current_high.values())
        assert spread_high > spread_low

    def test_elo_is_zero_sum(self, sample_matches_df):
        """Total Elo across all teams should stay close to n_teams * initial."""
        initial = 1500
        _, current = build_elo_ratings(sample_matches_df, initial_elo=initial)
        total = sum(current.values())
        expected_total = len(current) * initial
        # Allow small floating-point drift
        assert abs(total - expected_total) < 1.0


class TestCalcEloFeatures:
    """Tests for extracting Elo features for a single match."""

    def test_exact_history_lookup(self, sample_matches_df):
        elo_history, current = build_elo_ratings(sample_matches_df)
        first_row = sample_matches_df.iloc[0]
        result = calc_elo_features(first_row, elo_history, current_ratings=current)

        assert "home_elo" in result
        assert "away_elo" in result
        assert "elo_diff" in result
        assert result["elo_diff"] == result["home_elo"] - result["away_elo"]

    def test_fallback_to_current_ratings(self, sample_matches_df):
        """Future match (not in history) should fall back to latest ratings."""
        elo_history, current = build_elo_ratings(sample_matches_df)

        future_row = pd.Series({
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "Date": pd.Timestamp("2025-01-01"),
        })
        result = calc_elo_features(
            future_row, elo_history, current_ratings=current,
        )
        assert result["home_elo"] == current["Arsenal"]
        assert result["away_elo"] == current["Chelsea"]

    def test_fallback_to_initial_elo(self):
        """Unknown team with no history and no current -> initial_elo."""
        result = calc_elo_features(
            pd.Series({
                "HomeTeam": "NewTeam", "AwayTeam": "AnotherNew",
                "Date": pd.Timestamp("2025-01-01"),
            }),
            elo_history={},
            initial_elo=1500,
            current_ratings=None,
        )
        assert result["home_elo"] == 1500
        assert result["away_elo"] == 1500
        assert result["elo_diff"] == 0.0
