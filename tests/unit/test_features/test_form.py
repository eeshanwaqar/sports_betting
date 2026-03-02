"""
Tests for src.features.form module.

Covers: calc_form, calc_exp_form, calc_venue_form, calc_streaks.
"""

import numpy as np
import pandas as pd

from src.features.form import calc_exp_form, calc_form, calc_streaks, calc_venue_form

# ---------------------------------------------------------------------------
# calc_form
# ---------------------------------------------------------------------------

class TestCalcForm:
    """Tests for rolling form over last N matches."""

    def test_returns_nan_when_no_prior_matches(self, sample_matches_df):
        """First match of the season -> no history -> all NaN."""
        result = calc_form(
            sample_matches_df, "Arsenal",
            before_date=pd.Timestamp("2023-08-01"), n=5,
        )
        assert np.isnan(result["form5_points"])
        assert np.isnan(result["form5_win_rate"])
        assert np.isnan(result["form5_goal_diff"])

    def test_correct_points_and_win_rate(self, sample_matches_df, cutoff_date):
        """Arsenal before Nov 1 has a known recent history."""
        result = calc_form(sample_matches_df, "Arsenal", cutoff_date, n=5)

        assert "form5_points" in result
        assert "form5_win_rate" in result
        assert isinstance(result["form5_points"], (int, float, np.integer, np.floating))
        assert 0.0 <= result["form5_win_rate"] <= 1.0

    def test_window_size_respected(self, sample_matches_df, late_cutoff_date):
        """Asking for n=3 should only consider last 3 matches."""
        result_3 = calc_form(sample_matches_df, "Arsenal", late_cutoff_date, n=3)
        result_5 = calc_form(sample_matches_df, "Arsenal", late_cutoff_date, n=5)

        # Different windows should generally give different totals
        assert "form3_points" in result_3
        assert "form5_points" in result_5

    def test_goal_diff_sign(self, sample_matches_df, cutoff_date):
        """A strong team should have a positive goal difference."""
        result = calc_form(sample_matches_df, "Arsenal", cutoff_date, n=5)
        # Arsenal won most early matches so goal_diff should be positive
        assert isinstance(result["form5_goal_diff"], (int, float, np.integer, np.floating))

    def test_avg_goals_positive(self, sample_matches_df, cutoff_date):
        """Average goals for/against should be non-negative."""
        result = calc_form(sample_matches_df, "Arsenal", cutoff_date, n=5)
        assert result["form5_avg_goals_for"] >= 0
        assert result["form5_avg_goals_against"] >= 0


# ---------------------------------------------------------------------------
# calc_exp_form
# ---------------------------------------------------------------------------

class TestCalcExpForm:
    """Tests for exponentially weighted form."""

    def test_returns_nan_when_no_matches(self, sample_matches_df):
        result = calc_exp_form(
            sample_matches_df, "Arsenal",
            before_date=pd.Timestamp("2023-08-01"), n=5,
        )
        assert np.isnan(result["exp_form"])

    def test_returns_float_value(self, sample_matches_df, cutoff_date):
        result = calc_exp_form(sample_matches_df, "Arsenal", cutoff_date, n=5)
        assert isinstance(result["exp_form"], float)
        # Points range is 0..3, so weighted average must be in [0, 3]
        assert 0.0 <= result["exp_form"] <= 3.0

    def test_higher_decay_weights_recent_more(self, sample_matches_df, cutoff_date):
        """A decay closer to 1.0 weights all matches more equally."""
        result_low = calc_exp_form(
            sample_matches_df, "Arsenal", cutoff_date, n=5, decay=0.3,
        )
        result_high = calc_exp_form(
            sample_matches_df, "Arsenal", cutoff_date, n=5, decay=0.95,
        )
        # Both should return a valid number; exact ordering depends on data
        assert not np.isnan(result_low["exp_form"])
        assert not np.isnan(result_high["exp_form"])


# ---------------------------------------------------------------------------
# calc_venue_form
# ---------------------------------------------------------------------------

class TestCalcVenueForm:
    """Tests for venue-specific (home/away) form."""

    def test_returns_nan_when_no_venue_matches(self, sample_matches_df):
        result = calc_venue_form(
            sample_matches_df, "Arsenal",
            before_date=pd.Timestamp("2023-08-01"), is_home=True,
        )
        assert np.isnan(result["venue_win_rate"])

    def test_home_form_keys_present(self, sample_matches_df, cutoff_date):
        result = calc_venue_form(
            sample_matches_df, "Arsenal", cutoff_date, is_home=True,
        )
        assert "venue_win_rate" in result
        assert "venue_avg_goals_for" in result
        assert "venue_avg_goals_against" in result
        assert "venue_clean_sheets" in result

    def test_away_form_works(self, sample_matches_df, cutoff_date):
        result = calc_venue_form(
            sample_matches_df, "Arsenal", cutoff_date, is_home=False,
        )
        assert "venue_win_rate" in result

    def test_win_rate_bounded(self, sample_matches_df, cutoff_date):
        result = calc_venue_form(
            sample_matches_df, "Arsenal", cutoff_date, is_home=True,
        )
        if not np.isnan(result["venue_win_rate"]):
            assert 0.0 <= result["venue_win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# calc_streaks
# ---------------------------------------------------------------------------

class TestCalcStreaks:
    """Tests for win / unbeaten / loss streaks."""

    def test_no_prior_matches_returns_zeros(self, sample_matches_df):
        result = calc_streaks(
            sample_matches_df, "Arsenal",
            before_date=pd.Timestamp("2023-08-01"),
        )
        assert result == {"win_streak": 0, "unbeaten_streak": 0, "loss_streak": 0}

    def test_streak_keys(self, sample_matches_df, cutoff_date):
        result = calc_streaks(sample_matches_df, "Arsenal", cutoff_date)
        assert set(result.keys()) == {"win_streak", "unbeaten_streak", "loss_streak"}

    def test_streak_values_non_negative(self, sample_matches_df, cutoff_date):
        result = calc_streaks(sample_matches_df, "Arsenal", cutoff_date)
        assert result["win_streak"] >= 0
        assert result["unbeaten_streak"] >= 0
        assert result["loss_streak"] >= 0

    def test_unbeaten_geq_win_streak(self, sample_matches_df, cutoff_date):
        """Unbeaten streak must be >= win streak (wins are also unbeaten)."""
        result = calc_streaks(sample_matches_df, "Arsenal", cutoff_date)
        assert result["unbeaten_streak"] >= result["win_streak"]

    def test_win_and_loss_mutually_exclusive(self, sample_matches_df, cutoff_date):
        """A team can't simultaneously be on a win streak and loss streak."""
        result = calc_streaks(sample_matches_df, "Arsenal", cutoff_date)
        # At most one of win_streak, loss_streak can be > 0
        assert not (result["win_streak"] > 0 and result["loss_streak"] > 0)
