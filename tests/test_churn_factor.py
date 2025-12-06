"""Tests for churn_factor indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indikator.churn_factor import churn_factor


class TestBasicFunctionality:
    """Test core functionality of churn_factor."""

    @pytest.fixture
    def basic_ohlcv_df(self) -> pd.DataFrame:
        """Basic OHLCV data for testing."""
        return pd.DataFrame({
            "high": [105.0, 110.0, 115.0],
            "low": [100.0, 105.0, 110.0],
            "volume": [1000, 2000, 1500],
        })

    def test_returns_dataframe_with_churn_factor_column(
        self, basic_ohlcv_df: pd.DataFrame
    ) -> None:
        """Should add churn_factor column to DataFrame."""
        result = churn_factor(basic_ohlcv_df)

        assert isinstance(result, pd.DataFrame)
        assert "churn_factor" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "volume" in result.columns

    def test_does_not_modify_original_dataframe(
        self, basic_ohlcv_df: pd.DataFrame
    ) -> None:
        """Should not modify the input DataFrame."""
        original_cols = basic_ohlcv_df.columns.tolist()
        _ = churn_factor(basic_ohlcv_df)

        assert basic_ohlcv_df.columns.tolist() == original_cols
        assert "churn_factor" not in basic_ohlcv_df.columns

    def test_output_length_matches_input(self, basic_ohlcv_df: pd.DataFrame) -> None:
        """Output should have same length as input."""
        result = churn_factor(basic_ohlcv_df)
        assert len(result) == len(basic_ohlcv_df)

    def test_empty_dataframe_returns_empty_with_column(self) -> None:
        """Empty input should return empty output with churn_factor column."""
        empty_df = pd.DataFrame({"high": [], "low": [], "volume": []})
        result = churn_factor(empty_df)

        assert len(result) == 0
        assert "churn_factor" in result.columns


class TestChurnFactorCalculation:
    """Test churn factor calculation logic."""

    def test_basic_calculation(self) -> None:
        """Should correctly calculate volume / (high - low)."""
        data = pd.DataFrame({
            "high": [110.0],
            "low": [100.0],
            "volume": [1000.0],
        })
        result = churn_factor(data)

        # 1000 / (110 - 100) = 1000 / 10 = 100
        assert result["churn_factor"].iloc[0] == 100.0

    def test_higher_churn_for_narrow_range(self) -> None:
        """Narrow price range with high volume should produce high churn."""
        # Narrow range, high volume
        narrow = pd.DataFrame({
            "high": [101.0],
            "low": [100.0],
            "volume": [1000.0],
        })

        # Wide range, same volume
        wide = pd.DataFrame({
            "high": [110.0],
            "low": [100.0],
            "volume": [1000.0],
        })

        result_narrow = churn_factor(narrow)
        result_wide = churn_factor(wide)

        assert (
            result_narrow["churn_factor"].iloc[0] > result_wide["churn_factor"].iloc[0]
        )

    def test_churn_increases_with_volume(self) -> None:
        """Higher volume should produce higher churn for same range."""
        # Low volume
        low_vol = pd.DataFrame({
            "high": [110.0],
            "low": [100.0],
            "volume": [1000.0],
        })

        # High volume
        high_vol = pd.DataFrame({
            "high": [110.0],
            "low": [100.0],
            "volume": [5000.0],
        })

        result_low = churn_factor(low_vol)
        result_high = churn_factor(high_vol)

        assert result_high["churn_factor"].iloc[0] > result_low["churn_factor"].iloc[0]


class TestZeroRangeHandling:
    """Test handling of zero price range (high == low)."""

    @pytest.fixture
    def zero_range_df(self) -> pd.DataFrame:
        """Data with zero price range bar."""
        return pd.DataFrame({
            "high": [105.0, 100.0, 110.0],
            "low": [100.0, 100.0, 105.0],
            "volume": [1000, 2000, 1500],
        })

    def test_nan_strategy_returns_nan_for_zero_range(
        self, zero_range_df: pd.DataFrame
    ) -> None:
        """Should return NaN for zero range bars when using 'nan' strategy."""
        result = churn_factor(zero_range_df, fill_strategy="nan")

        # Second bar has zero range (high == low)
        assert not pd.isna(result["churn_factor"].iloc[0])
        assert pd.isna(result["churn_factor"].iloc[1])
        assert not pd.isna(result["churn_factor"].iloc[2])

    def test_forward_fill_strategy_fills_from_previous(
        self, zero_range_df: pd.DataFrame
    ) -> None:
        """Should forward fill NaN values when using 'forward_fill' strategy."""
        result = churn_factor(zero_range_df, fill_strategy="forward_fill")

        # Should have valid value at index 1 (forward filled)
        assert not pd.isna(result["churn_factor"].iloc[1])
        # Should be same as previous bar
        assert result["churn_factor"].iloc[1] == result["churn_factor"].iloc[0]

    def test_zero_strategy_uses_epsilon(self, zero_range_df: pd.DataFrame) -> None:
        """Should use epsilon to prevent division by zero with 'zero' strategy."""
        result = churn_factor(zero_range_df, fill_strategy="zero", epsilon=1e-9)

        # Should not contain NaN values
        assert not result["churn_factor"].isna().any()
        # Should not contain inf values
        assert not np.isinf(result["churn_factor"]).any()

    def test_zero_strategy_with_custom_fill_value(self) -> None:
        """Should use custom fill_value for remaining NaN with 'zero' strategy."""
        data = pd.DataFrame({
            "high": [100.0],
            "low": [100.0],
            "volume": [1000.0],
        })

        result = churn_factor(data, fill_strategy="zero", fill_value=999.0)

        # Should have a valid value (not NaN)
        assert not pd.isna(result["churn_factor"].iloc[0])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_high_column_raises_error(self) -> None:
        """Should raise SchemaError if 'high' column is missing."""
        data = pd.DataFrame({"low": [100.0], "volume": [1000.0]})

        with pytest.raises(ValueError):
            churn_factor(data)

    def test_missing_low_column_raises_error(self) -> None:
        """Should raise SchemaError if 'low' column is missing."""
        data = pd.DataFrame({"high": [105.0], "volume": [1000.0]})

        with pytest.raises(ValueError):
            churn_factor(data)

    def test_missing_volume_column_raises_error(self) -> None:
        """Should raise SchemaError if 'volume' column is missing."""
        data = pd.DataFrame({"high": [105.0], "low": [100.0]})

        with pytest.raises(ValueError):
            churn_factor(data)

    def test_all_zero_range_bars(self) -> None:
        """Should handle data where all bars have zero range."""
        data = pd.DataFrame({
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "volume": [1000, 1500, 1200],
        })

        result = churn_factor(data, fill_strategy="nan")
        assert result["churn_factor"].isna().all()

    def test_low_greater_than_high_raises_error(self) -> None:
        """Should raise SchemaError if data contains low > high."""
        data = pd.DataFrame({
            "high": [100.0, 100.0],
            "low": [95.0, 105.0],  # Second bar invalid
            "volume": [1000, 1500],
        })

        with pytest.raises(ValueError):
            churn_factor(data)


class TestParameterValidation:
    """Test parameter validation and constraints."""

    @pytest.fixture
    def basic_ohlcv_df(self) -> pd.DataFrame:
        """Basic OHLCV data for testing."""
        return pd.DataFrame({
            "high": [105.0, 110.0, 115.0],
            "low": [100.0, 105.0, 110.0],
            "volume": [1000, 2000, 1500],
        })

    def test_default_parameters_work(self, basic_ohlcv_df: pd.DataFrame) -> None:
        """Should work with all default parameters."""
        result = churn_factor(basic_ohlcv_df)
        assert "churn_factor" in result.columns

    def test_custom_epsilon(self, basic_ohlcv_df: pd.DataFrame) -> None:
        """Should accept custom epsilon parameter."""
        result = churn_factor(basic_ohlcv_df, epsilon=1e-6)
        assert "churn_factor" in result.columns

    def test_custom_fill_value(self, basic_ohlcv_df: pd.DataFrame) -> None:
        """Should accept custom fill_value parameter."""
        result = churn_factor(basic_ohlcv_df, fill_value=1.0)
        assert "churn_factor" in result.columns


class TestInterpretation:
    """Test churn factor interpretation scenarios."""

    def test_high_churn_indicates_tight_range_heavy_volume(self) -> None:
        """High churn should indicate lots of trading in narrow range."""
        # Heavy volume in 1-point range
        high_churn = pd.DataFrame({
            "high": [100.5],
            "low": [100.0],
            "volume": [10000.0],
        })

        result = churn_factor(high_churn)

        # Churn = 10000 / 0.5 = 20000 (very high)
        assert result["churn_factor"].iloc[0] > 10000

    def test_low_churn_indicates_wide_range_low_volume(self) -> None:
        """Low churn should indicate little trading across wide range."""
        # Low volume in 10-point range
        low_churn = pd.DataFrame({
            "high": [110.0],
            "low": [100.0],
            "volume": [100.0],
        })

        result = churn_factor(low_churn)

        # Churn = 100 / 10 = 10 (low)
        assert result["churn_factor"].iloc[0] < 50


class TestFullOHLCVData:
    """Test with complete OHLCV data."""

    @pytest.fixture
    def full_ohlcv_df(self) -> pd.DataFrame:
        """Full OHLCV DataFrame."""
        return pd.DataFrame({
            "open": [100.0, 102.0, 105.0],
            "high": [105.0, 110.0, 115.0],
            "low": [99.0, 101.0, 104.0],
            "close": [102.0, 108.0, 112.0],
            "volume": [1000, 2000, 1500],
        })

    def test_preserves_all_original_columns(self, full_ohlcv_df: pd.DataFrame) -> None:
        """Should preserve all original OHLCV columns."""
        result = churn_factor(full_ohlcv_df)

        assert all(col in result.columns for col in full_ohlcv_df.columns)
        assert "churn_factor" in result.columns

    def test_churn_calculation_with_full_ohlcv(
        self, full_ohlcv_df: pd.DataFrame
    ) -> None:
        """Should correctly calculate churn with full OHLCV data."""
        result = churn_factor(full_ohlcv_df)

        # First bar: 1000 / (105 - 99) = 1000 / 6 ≈ 166.67
        assert np.isclose(result["churn_factor"].iloc[0], 1000 / 6, rtol=1e-10)

        # Second bar: 2000 / (110 - 101) = 2000 / 9 ≈ 222.22
        assert np.isclose(result["churn_factor"].iloc[1], 2000 / 9, rtol=1e-10)

        # Third bar: 1500 / (115 - 104) = 1500 / 11 ≈ 136.36
        assert np.isclose(result["churn_factor"].iloc[2], 1500 / 11, rtol=1e-10)
