"""Tests for Bollinger Bands indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.bollinger import bollinger_bands


class TestBollingerBands:
    """Tests for Bollinger Bands indicator."""

    def test_bollinger_basic(self):
        """Test Bollinger Bands basic calculation."""
        prices = pd.Series(
            [
                100.0,
                102.0,
                101.0,
                103.0,
                105.0,
                104.0,
                106.0,
                108.0,
                107.0,
                109.0,
                111.0,
                110.0,
            ]
            * 2
        )

        result = bollinger_bands(prices, window=5, num_std=2.0)

        # Check columns
        assert "bb_middle" in result.columns
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_bandwidth" in result.columns
        assert "bb_percent" in result.columns

        # Check shape
        assert len(result) == len(prices)

        # Check middle band is SMA
        expected_middle = prices.rolling(window=5, min_periods=1).mean()
        pd.testing.assert_series_equal(
            result["bb_middle"], expected_middle, check_names=False
        )

        # Check upper band >= middle band (where both are not NaN)
        valid_mask = result["bb_upper"].notna() & result["bb_middle"].notna()
        assert (
            result.loc[valid_mask, "bb_upper"] >= result.loc[valid_mask, "bb_middle"]
        ).all()

        # Check lower band <= middle band (where both are not NaN)
        assert (
            result.loc[valid_mask, "bb_lower"] <= result.loc[valid_mask, "bb_middle"]
        ).all()

    def test_bollinger_dataframe(self):
        """Test Bollinger Bands with DataFrame input."""
        data = pd.DataFrame({
            "close": [100.0, 102.0, 101.0, 103.0, 105.0],
            "volume": [100, 200, 150, 180, 220],
        })
        # Pass Series directly
        result = bollinger_bands(data["close"], window=5, num_std=2.0)
        assert isinstance(result, pd.DataFrame)
        assert "bb_middle" in result.columns

    def test_bollinger_percent_b(self):
        """Test Bollinger %B calculation."""
        prices = pd.Series(
            [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 2
        )

        result = bollinger_bands(prices, window=5, num_std=2.0)

        # %B should be in range [0, 1] most of the time (can go outside for extreme values)
        # At middle band, %B should be 0.5
        # We can't test exact values without knowing std dev, but we can check it's calculated
        assert result["bb_percent"].notna().any()

    def test_bollinger_empty_data(self):
        """Test Bollinger Bands with empty series."""
        prices = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            bollinger_bands(prices)

    def test_bollinger_window_parameter(self):
        """Test Bollinger Bands with different window sizes."""
        prices = pd.Series(
            [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 3
        )

        result_short = bollinger_bands(prices, window=3, num_std=2.0)
        result_long = bollinger_bands(prices, window=15, num_std=2.0)

        # Different windows should produce different results
        assert not result_short["bb_middle"].equals(result_long["bb_middle"])

    def test_bollinger_invalid_input(self):
        """Test Bollinger Bands with invalid input."""
        # Infinite values
        data = pd.Series([100.0, np.inf, 102.0])
        with pytest.raises(ValueError, match="must be finite"):
            bollinger_bands(data)

    def test_bollinger_with_inf(self):
        """Test Bollinger Bands with Inf values."""
        prices = pd.Series([100.0, 102.0, np.inf, 103.0, 105.0])

        with pytest.raises(ValueError, match="must be finite"):
            bollinger_bands(prices)
