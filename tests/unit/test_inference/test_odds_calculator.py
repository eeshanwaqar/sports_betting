"""
Tests for src.inference.odds_calculator module.

Covers: prob_to_odds, probs_to_odds, calc_value, is_value_bet.
"""

import pytest
import math
import numpy as np

from src.inference.odds_calculator import (
    prob_to_odds,
    probs_to_odds,
    calc_value,
    is_value_bet,
)


# ---------------------------------------------------------------------------
# prob_to_odds
# ---------------------------------------------------------------------------

class TestProbToOdds:
    """Tests for single-probability to decimal-odds conversion."""

    def test_even_money_no_margin(self):
        """50% probability with 0% margin should give odds of 2.0."""
        result = prob_to_odds(0.5, margin=0.0)
        assert result == 2.0

    def test_even_money_with_margin(self):
        """50% probability with 5% margin -> 1/(0.5*1.05) = 1.905."""
        result = prob_to_odds(0.5, margin=0.05)
        assert result == round(1 / (0.5 * 1.05), 2)

    def test_high_probability(self):
        """90% probability should give low odds (favourite)."""
        result = prob_to_odds(0.9, margin=0.05)
        assert result < 2.0
        assert result > 1.0

    def test_low_probability(self):
        """10% probability should give high odds (long shot)."""
        result = prob_to_odds(0.1, margin=0.05)
        assert result > 5.0

    def test_zero_probability_returns_inf(self):
        result = prob_to_odds(0.0)
        assert result == float("inf")

    def test_negative_probability_returns_inf(self):
        result = prob_to_odds(-0.1)
        assert result == float("inf")

    def test_result_is_rounded_to_two_decimals(self):
        result = prob_to_odds(0.333, margin=0.0)
        # 1/0.333 = 3.003003... rounded to 3.0
        assert result == round(1 / 0.333, 2)


# ---------------------------------------------------------------------------
# probs_to_odds
# ---------------------------------------------------------------------------

class TestProbsToOdds:
    """Tests for converting all three probabilities to odds."""

    def test_returns_three_keys(self):
        result = probs_to_odds(0.5, 0.3, 0.2)
        assert set(result.keys()) == {"home_win", "draw", "away_win"}

    def test_higher_prob_gives_lower_odds(self):
        """The favourite (highest prob) should have the lowest odds."""
        result = probs_to_odds(0.6, 0.25, 0.15)
        assert result["home_win"] < result["draw"]
        assert result["draw"] < result["away_win"]

    def test_all_odds_above_one(self):
        """Decimal odds must always be > 1."""
        result = probs_to_odds(0.5, 0.3, 0.2, margin=0.0)
        for key in result:
            assert result[key] > 1.0


# ---------------------------------------------------------------------------
# calc_value
# ---------------------------------------------------------------------------

class TestCalcValue:
    """Tests for value calculation."""

    def test_positive_value(self):
        """Model prob > bookmaker prob -> positive value."""
        assert calc_value(0.6, 0.5) == pytest.approx(0.1)

    def test_negative_value(self):
        """Model prob < bookmaker prob -> negative value."""
        assert calc_value(0.4, 0.5) == pytest.approx(-0.1)

    def test_zero_value(self):
        assert calc_value(0.5, 0.5) == pytest.approx(0.0)

    def test_nan_inputs_return_zero(self):
        assert calc_value(np.nan, 0.5) == 0.0
        assert calc_value(0.5, np.nan) == 0.0
        assert calc_value(np.nan, np.nan) == 0.0


# ---------------------------------------------------------------------------
# is_value_bet
# ---------------------------------------------------------------------------

class TestIsValueBet:
    """Tests for value bet detection."""

    def test_positive_edge_is_value(self):
        assert is_value_bet(0.6, 0.5, threshold=0.0) is True

    def test_negative_edge_is_not_value(self):
        assert is_value_bet(0.4, 0.5, threshold=0.0) is False

    def test_threshold_filters_marginal_value(self):
        """0.51 vs 0.50 has value of 0.01, but threshold 0.05 should reject."""
        assert is_value_bet(0.51, 0.50, threshold=0.05) is False
        assert is_value_bet(0.56, 0.50, threshold=0.05) is True

    def test_equal_probs_not_value(self):
        """Exactly equal should NOT be flagged (value = 0 is not > threshold)."""
        assert is_value_bet(0.5, 0.5, threshold=0.0) is False
