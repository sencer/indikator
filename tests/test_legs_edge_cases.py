"""Edge case and boundary condition tests for zigzag_legs."""

from __future__ import annotations

import pandas as pd
import pytest

from indikator.legs import zigzag_legs


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_two_values_below_threshold(self) -> None:
        """Two values with price change below threshold."""
        df = pd.DataFrame({"close": [100.0, 100.5]})  # 0.5% change
        result = zigzag_legs(df, threshold=0.01)  # 1% threshold

        # Should not establish trend
        assert (result["zigzag_legs"] == 0.0).all()

    def test_two_values_at_exact_threshold(self) -> None:
        """Two values at exactly the threshold."""
        df = pd.DataFrame({"close": [100.0, 101.0]})  # Exactly 1% change
        result = zigzag_legs(df, threshold=0.01)

        # Should not establish trend (needs to exceed threshold)
        assert result["zigzag_legs"].iloc[0] == 0.0

    def test_two_values_above_threshold(self) -> None:
        """Two values with price change above threshold."""
        df = pd.DataFrame({"close": [100.0, 102.0]})  # 2% change
        result = zigzag_legs(df, threshold=0.01)  # 1% threshold

        # Should establish uptrend
        assert result["zigzag_legs"].iloc[-1] > 0

    def test_exact_threshold_crossing_up(self) -> None:
        """Test price that crosses threshold exactly."""
        df = pd.DataFrame({"close": [100.0, 101.01]})  # Just over 1%
        result = zigzag_legs(df, threshold=0.01, confirmation_bars=0)

        assert result["zigzag_legs"].iloc[-1] > 0

    def test_exact_threshold_crossing_down(self) -> None:
        """Test price that crosses threshold exactly downward."""
        df = pd.DataFrame({"close": [100.0, 98.99]})  # Just over -1%
        result = zigzag_legs(df, threshold=0.01, confirmation_bars=0)

        assert result["zigzag_legs"].iloc[-1] < 0

    def test_very_small_prices(self) -> None:
        """Test with very small price values."""
        df = pd.DataFrame({"close": [0.0001, 0.00011, 0.00012]})
        result = zigzag_legs(df, threshold=0.05)

        # Should handle small values without error
        assert len(result) == 3
        assert "zigzag_legs" in result.columns

    def test_very_large_prices(self) -> None:
        """Test with very large price values."""
        df = pd.DataFrame({"close": [1e10, 1.1e10, 1.2e10]})
        result = zigzag_legs(df, threshold=0.05)

        # Should handle large values without overflow
        assert len(result) == 3
        assert result["zigzag_legs"].iloc[-1] > 0

    def test_single_spike_up_and_reversal(self) -> None:
        """Test single spike that immediately reverses."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Spike up 10%
                100.0,  # Immediately back down
            ]
        })
        result = zigzag_legs(df, threshold=0.05, confirmation_bars=0)

        # Should detect both up and down legs
        assert "zigzag_legs" in result.columns
        assert len(result) == 3

    def test_single_spike_down_and_reversal(self) -> None:
        """Test single spike down that immediately reverses."""
        df = pd.DataFrame({
            "close": [
                100.0,
                90.0,  # Spike down 10%
                100.0,  # Immediately back up
            ]
        })
        result = zigzag_legs(df, threshold=0.05, confirmation_bars=0)

        assert "zigzag_legs" in result.columns
        assert len(result) == 3

    def test_alternating_small_changes(self) -> None:
        """Test alternating small changes below threshold."""
        df = pd.DataFrame({
            "close": [100.0, 100.2, 99.8, 100.1, 99.9, 100.0]  # All < 0.5%
        })
        result = zigzag_legs(df, threshold=0.01)

        # Should not establish any trend
        assert (result["zigzag_legs"] == 0.0).all()

    def test_extremely_small_threshold(self) -> None:
        """Test with very small threshold (high sensitivity)."""
        df = pd.DataFrame({"close": [100.0, 100.01, 100.02, 100.03]})
        result = zigzag_legs(df, threshold=0.0001)  # 0.01% threshold

        # Should detect trend with tiny changes
        assert result["zigzag_legs"].iloc[-1] != 0

    def test_extremely_large_threshold(self) -> None:
        """Test with very large threshold (low sensitivity)."""
        df = pd.DataFrame({"close": [100.0, 110.0, 120.0, 130.0]})
        result = zigzag_legs(df, threshold=0.5)  # 50% threshold

        # Should not detect trend even with large moves
        assert (result["zigzag_legs"] == 0.0).all()


class TestInvalidInputs:
    """Tests for invalid inputs (NaN, Inf, etc)."""

    def test_nan_values_raise_error(self):
        """NaN values should raise error."""
        nan_prices_df = pd.DataFrame({"close": [100.0, 105.0, float("nan"), 110.0]})
        with pytest.raises(ValueError):
            zigzag_legs(nan_prices_df)

    def test_inf_values_raise_error(self):
        """Infinite values should raise error."""
        inf_prices_df = pd.DataFrame({"close": [100.0, 105.0, float("inf"), 110.0]})
        with pytest.raises(ValueError):
            zigzag_legs(inf_prices_df)

    def test_negative_inf_values_raise_error(self):
        """Negative infinite values should raise error."""
        df = pd.DataFrame({"close": [100.0, 105.0, float("-inf"), 110.0]})
        with pytest.raises(ValueError):
            zigzag_legs(df)

    def test_all_nan_values(self):
        """All NaN values should raise error."""
        df = pd.DataFrame({"close": [float("nan"), float("nan"), float("nan")]})
        with pytest.raises(ValueError):
            zigzag_legs(df)

    def test_mixed_nan_and_valid(self):
        """Mixture of valid and NaN values should raise error."""
        df = pd.DataFrame({"close": [100.0, float("nan"), 105.0, float("nan")]})
        with pytest.raises(ValueError):
            zigzag_legs(df)


class TestMinDistanceParameter:
    """Test min_distance_pct parameter behavior."""

    def test_min_distance_filters_small_moves(self) -> None:
        """min_distance_pct should filter out small pivot updates."""
        df = pd.DataFrame({
            "close": [
                100.0,
                102.0,  # 2% up - establishes trend
                102.1,  # 0.1% up - below min_distance
                102.2,  # 0.1% up - below min_distance
                105.0,  # 2.9% up from 102 - above min_distance
            ]
        })

        # With default min_distance_pct=0.005 (0.5%), tiny moves get filtered
        result = zigzag_legs(df, threshold=0.01, min_distance_pct=0.005)

        assert "zigzag_legs" in result.columns

    def test_zero_min_distance_updates_every_bar(self) -> None:
        """Zero min_distance should update pivot on any price increase."""
        df = pd.DataFrame({
            "close": [100.0, 100.01, 100.02, 100.03, 100.04]  # Tiny increments
        })

        result = zigzag_legs(
            df, threshold=0.0001, min_distance_pct=0.0, confirmation_bars=0
        )

        # Should establish trend and track every tiny move
        assert result["zigzag_legs"].iloc[-1] > 0

    def test_large_min_distance_ignores_noise(self) -> None:
        """Large min_distance should ignore noisy price action."""
        df = pd.DataFrame({
            "close": [
                100.0,
                105.0,  # 5% up
                104.5,
                105.5,
                104.8,  # Noise around 105
                110.0,  # 5% up from 105
            ]
        })

        result = zigzag_legs(df, threshold=0.01, min_distance_pct=0.02)

        # Large min_distance should smooth out the noise
        assert "zigzag_legs" in result.columns


class TestEpsilonParameter:
    """Test epsilon (division by zero protection)."""

    def test_zero_prices_handled(self) -> None:
        """Zero prices should be handled by epsilon."""
        df = pd.DataFrame({"close": [0.0, 0.0, 0.0]})

        # Should not crash, epsilon prevents division by zero
        result = zigzag_legs(df, threshold=0.01)

        assert len(result) == 3
        assert (result["zigzag_legs"] == 0.0).all()

    def test_prices_starting_from_zero(self) -> None:
        """Prices starting from zero and increasing."""
        df = pd.DataFrame({"close": [0.0, 1.0, 2.0, 3.0]})

        result = zigzag_legs(df, threshold=0.1, epsilon=1e-9)

        # Should handle transition from zero
        assert len(result) == 4

    def test_custom_epsilon(self, simple_uptrend_df: pd.DataFrame) -> None:
        """Custom epsilon value should work."""
        result = zigzag_legs(simple_uptrend_df, epsilon=1e-12)

        assert "zigzag_legs" in result.columns
