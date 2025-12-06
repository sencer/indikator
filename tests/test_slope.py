"""Tests for slope indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indikator.slope import slope


class TestBasicFunctionality:
    """Test core functionality of slope."""

    def test_returns_series_from_dataframe(
        self, simple_uptrend_df: pd.DataFrame
    ) -> None:
        """Should return a Series when input is a DataFrame."""
        result = slope(simple_uptrend_df["close"], window=3)

        assert isinstance(result, pd.Series)
        assert result.name == "close"
        assert len(result) == len(simple_uptrend_df)

    def test_returns_series_from_series(self, simple_uptrend_df: pd.DataFrame) -> None:
        """Should return a Series when input is a Series."""
        s = simple_uptrend_df["close"]
        result = slope(s, window=3)

        assert isinstance(result, pd.Series)
        assert result.name == "close"
        assert len(result) == len(s)

    def test_custom_column_selection(self) -> None:
        """Should select correct column from DataFrame."""
        df = pd.DataFrame({
            "close": [100.0, 102.0, 105.0, 108.0],
            "vwap": [101.0, 103.0, 106.0, 109.0],
        })

        result = slope(df["vwap"], window=3)
        assert result.name == "vwap"
        # Should match slopes of vwap data
        assert not pd.isna(result.iloc[2])

    def test_output_length_matches_input(self, simple_uptrend_df: pd.DataFrame) -> None:
        """Output should have same length as input."""
        result = slope(simple_uptrend_df["close"], window=3)
        assert len(result) == len(simple_uptrend_df)

    def test_empty_dataframe_returns_empty_series(self) -> None:
        """Should raise ValueError if input is empty."""
        with pytest.raises(ValueError, match="Data must not be empty"):
            slope(pd.Series(dtype=float))


class TestSlopeCalculation:
    """Test slope calculation logic."""

    def test_uptrend_has_positive_slope(self, simple_uptrend_df: pd.DataFrame) -> None:
        """Uptrend should produce positive slope values."""
        result = slope(simple_uptrend_df["close"], window=3)

        # Get non-NaN values (after window is satisfied)
        slopes = result.dropna()
        assert (slopes > 0).all()

    def test_downtrend_has_negative_slope(
        self, simple_downtrend_df: pd.DataFrame
    ) -> None:
        """Downtrend should produce negative slope values."""
        result = slope(simple_downtrend_df["close"], window=3)

        slopes = result.dropna()
        assert (slopes < 0).all()

    def test_flat_prices_have_zero_slope(self, flat_prices_df: pd.DataFrame) -> None:
        """Constant prices should result in zero slope."""
        result = slope(flat_prices_df["close"], window=3)

        slopes = result.dropna()
        # Allow small numerical errors
        assert np.allclose(slopes, 0.0, atol=1e-10)

    def test_linear_growth_has_constant_slope(self) -> None:
        """Perfect linear growth should have constant slope."""
        # Linear: 100, 102, 104, 106, 108, 110
        data = pd.Series(np.linspace(100, 110, 6), name="close")
        result = slope(data, window=3)

        slopes = result.dropna()
        # All slopes should be approximately equal
        assert np.allclose(slopes, slopes.iloc[0], rtol=1e-10)

    def test_steeper_trend_has_larger_slope_magnitude(self) -> None:
        """Steeper trends should have larger absolute slope values."""
        # Gentle trend
        gentle = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], name="close")
        result_gentle = slope(gentle, window=3)

        # Steep trend
        steep = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], name="close")
        result_steep = slope(steep, window=3)

        avg_slope_gentle = result_gentle.dropna().mean()
        avg_slope_steep = result_steep.dropna().mean()

        assert avg_slope_steep > avg_slope_gentle

    def test_window_parameter_affects_smoothness(self) -> None:
        """Larger window should produce smoother slope values."""
        # Noisy data
        np.random.seed(42)
        data = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5 + 0.5), name="close")

        result_small = slope(data, window=3)
        result_large = slope(data, window=10)

        # Larger window should have less variation (smoother)
        std_small = result_small.dropna().std()
        std_large = result_large.dropna().std()

        assert std_small > std_large


class TestNaNHandling:
    """Test NaN value handling."""

    def test_insufficient_data_returns_nan(self) -> None:
        """Should return NaN for bars before window is satisfied."""
        data = pd.Series([100.0, 102.0, 105.0, 108.0], name="close")
        result = slope(data, window=3)

        # First 2 values should be NaN (window=3 needs 3 bars)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # Third value onward should be valid
        assert not pd.isna(result.iloc[2])

    def test_single_value_returns_nan(self, single_value_df: pd.DataFrame) -> None:
        """Single value should return NaN (insufficient for slope)."""
        result = slope(single_value_df["close"], window=3)
        assert pd.isna(result.iloc[0])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_column_raises_error(self, simple_uptrend_df: pd.DataFrame) -> None:
        """Should raise ValueError if column doesn't exist."""
        with pytest.raises(KeyError):
            slope(simple_uptrend_df["nonexistent"], window=3)

    def test_minimum_window_size(self) -> None:
        """Should work with minimum window size of 2."""
        data = pd.Series([100.0, 105.0, 110.0], name="close")
        result = slope(data, window=2)

        # Should have valid slopes starting from index 1
        assert not pd.isna(result.iloc[1])

    def test_large_dataset_performance(self, large_dataset_df: pd.DataFrame) -> None:
        """Should handle large datasets efficiently (Numba optimization)."""
        result = slope(large_dataset_df["close"], window=20)

        assert len(result) == len(large_dataset_df)

    def test_infinite_values_raise_error(self) -> None:
        """Should raise SchemaError if data contains infinite values."""
        data = pd.Series([100.0, np.inf, 105.0], name="close")
        with pytest.raises(ValueError):
            slope(data)


class TestParameterValidation:
    """Test parameter validation and constraints."""

    def test_default_parameters_work(self, simple_uptrend_df: pd.DataFrame) -> None:
        """Should work with all default parameters."""
        result = slope(simple_uptrend_df["close"])
        assert isinstance(result, pd.Series)

    def test_custom_window(self, simple_uptrend_df: pd.DataFrame) -> None:
        """Should accept custom window parameter."""
        result = slope(simple_uptrend_df["close"], window=2)
        assert len(result) == len(simple_uptrend_df)


class TestTrendStrength:
    """Test interpretation of slope magnitude."""

    def test_accelerating_trend_increases_slope(self) -> None:
        """Accelerating uptrend should show increasing slope values."""
        # Accelerating: 100, 101, 103, 106, 110
        data = pd.Series([100.0, 101.0, 103.0, 106.0, 110.0], name="close")
        result = slope(data, window=3)

        slopes = result.dropna().values
        # Later slopes should be larger (accelerating)
        assert slopes[-1] > slopes[0]

    def test_decelerating_trend_decreases_slope(self) -> None:
        """Decelerating uptrend should show decreasing slope values."""
        # Decelerating: 100, 105, 108, 110, 111
        data = pd.Series([100.0, 105.0, 108.0, 110.0, 111.0], name="close")
        result = slope(data, window=3)

        slopes = result.dropna().values
        # Later slopes should be smaller (decelerating)
        assert slopes[-1] < slopes[0]


class TestOHLCVData:
    """Test with full OHLCV data."""

    @pytest.fixture
    def ohlcv_df(self) -> pd.DataFrame:
        """Sample OHLCV data."""
        return pd.DataFrame({
            "open": [100.0, 102.0, 105.0, 108.0, 110.0],
            "high": [103.0, 106.0, 108.0, 111.0, 113.0],
            "low": [99.0, 101.0, 104.0, 107.0, 109.0],
            "close": [102.0, 105.0, 107.0, 110.0, 112.0],
            "volume": [1000, 1500, 1200, 1300, 1100],
        })

    def test_works_with_ohlcv_data(self, ohlcv_df: pd.DataFrame) -> None:
        """Should work with full OHLCV DataFrame."""
        result = slope(ohlcv_df["close"], window=3)
        assert isinstance(result, pd.Series)
        assert result.name == "close"

    def test_can_calculate_slope_for_high(self, ohlcv_df: pd.DataFrame) -> None:
        """Should work with high column."""
        result = slope(ohlcv_df["high"], window=3)

        assert result.name == "high"
        slopes = result.dropna()
        assert (slopes > 0).all()  # High prices trending up
