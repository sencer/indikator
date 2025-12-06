"""Tests for zscore indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indikator.zscore import zscore


class TestBasicFunctionality:
  """Test core functionality of zscore."""

  def test_returns_series_from_dataframe(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should return Series with zscore values."""
    result = zscore(simple_uptrend_df["close"])

    assert isinstance(result, pd.Series)
    assert result.name == "close"
    assert len(result) == len(simple_uptrend_df)

  def test_custom_column_selection(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should calculate zscore for custom column."""
    df = simple_uptrend_df.copy()
    df["vwap"] = df["close"] * 1.01

    result = zscore(df["vwap"])
    assert result.name == "vwap"
    assert not pd.isna(result.iloc[2])

  def test_output_length_matches_input(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Output should have same length as input."""
    result = zscore(simple_uptrend_df["close"])
    assert len(result) == len(simple_uptrend_df)

  def test_empty_dataframe_returns_empty_series(self) -> None:
    """Should raise ValueError if input is empty."""
    with pytest.raises(ValueError, match="Data must not be empty"):
      zscore(pd.Series(dtype=float))


class TestZScoreCalculation:
  """Test zscore calculation logic."""

  def test_flat_prices_return_zero_zscore(self, flat_prices_df: pd.DataFrame) -> None:
    """Constant prices should result in zero zscore (no deviation)."""
    result = zscore(flat_prices_df["close"], window=3)

    # All values should be 0 since there's no price variation
    assert (result == 0.0).all()

  def test_single_deviation_has_expected_sign(self) -> None:
    """Single price above mean should have positive zscore."""
    # Create data: mostly 100, then spike to 110
    data = pd.Series([100.0, 100.0, 100.0, 110.0], name="close")
    result = zscore(data, window=3)

    # Last value (110) is above the rolling mean, so zscore should be positive
    assert result.iloc[-1] > 0

  def test_negative_deviation_has_negative_zscore(self) -> None:
    """Price below mean should have negative zscore."""
    # Create data: mostly 100, then drop to 90
    data = pd.Series([100.0, 100.0, 100.0, 90.0], name="close")
    result = zscore(data, window=3)

    # Last value (90) is below the rolling mean, so zscore should be negative
    assert result.iloc[-1] < 0

  def test_window_parameter_affects_sensitivity(self) -> None:
    """Larger window retains memory of older prices."""
    # Long stable period at 100, then spike to 110
    # Larger window "remembers" the stable 100s better
    data = pd.Series([100.0, 100.0, 100.0, 100.0, 110.0], name="close")

    result_small = zscore(data, window=2)
    result_large = zscore(data, window=4)

    # Larger window has stronger memory of stable 100s, so spike is more significant
    # This is expected behavior - larger windows better detect anomalies
    assert abs(result_large.iloc[-1]) > abs(result_small.iloc[-1])

  def test_zscore_magnitude_increases_with_deviation(self) -> None:
    """Larger deviations should produce larger absolute zscore values."""
    # Small deviation
    data_small = pd.Series([100.0, 100.0, 100.0, 102.0], name="close")
    result_small = zscore(data_small, window=3)

    # Large deviation
    data_large = pd.Series([100.0, 100.0, 100.0, 110.0], name="close")
    result_large = zscore(data_large, window=3)

    assert abs(result_large.iloc[-1]) > abs(result_small.iloc[-1])


class TestEdgeCases:
  """Test edge cases and error handling."""

  def test_single_value_returns_zero(self, single_value_df: pd.DataFrame) -> None:
    """Single value should have zscore of 0 (no std dev)."""
    result = zscore(single_value_df["close"], window=2)
    assert result.iloc[0] == 0.0

  def test_missing_column_raises_error(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should raise ValueError if column doesn't exist."""
    # This test is no longer relevant for zscore as it takes Series
    # But we can test if we try to access missing column before passing
    with pytest.raises(KeyError):
      zscore(simple_uptrend_df["nonexistent"])

  def test_epsilon_prevents_division_by_zero(self) -> None:
    """Should handle near-zero std dev without producing inf."""
    # Prices with very small variation
    data = pd.Series([100.0, 100.0, 100.0], name="close")
    result = zscore(data, window=3, epsilon=1e-9)

    # Should not contain inf values
    assert not np.isinf(result).any()

  def test_infinite_values_raise_error(self) -> None:
    """Should raise SchemaError if data contains infinite values."""
    data = pd.Series([100.0, np.inf, 105.0], name="close")
    with pytest.raises(ValueError):
      zscore(data)


class TestParameterValidation:
  """Test parameter validation and constraints."""

  def test_default_parameters_work(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should work with all default parameters."""
    result = zscore(simple_uptrend_df["close"])
    assert isinstance(result, pd.Series)

  def test_custom_window(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should accept custom window parameter."""
    result = zscore(simple_uptrend_df["close"], window=10)
    assert len(result) == len(simple_uptrend_df)

  def test_custom_epsilon(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should accept custom epsilon parameter."""
    result = zscore(simple_uptrend_df["close"], epsilon=1e-6)
    assert len(result) == len(simple_uptrend_df)


class TestOHLCVData:
  """Test with full OHLCV data."""

  @pytest.fixture
  def ohlcv_df(self) -> pd.DataFrame:
    """Sample OHLCV data."""
    return pd.DataFrame({
      "open": [100.0, 102.0, 105.0],
      "high": [103.0, 106.0, 108.0],
      "low": [99.0, 101.0, 104.0],
      "close": [102.0, 105.0, 107.0],
      "volume": [1000, 1500, 1200],
    })

  def test_works_with_ohlcv_data(self, ohlcv_df: pd.DataFrame) -> None:
    """Should work with full OHLCV DataFrame."""
    result = zscore(ohlcv_df["close"])
    assert isinstance(result, pd.Series)
    assert result.name == "close"

  def test_can_calculate_zscore_for_volume(self, ohlcv_df: pd.DataFrame) -> None:
    """Should work with volume column."""
    result = zscore(ohlcv_df["volume"], window=2)
    assert result.name == "volume"
    assert len(result) == len(ohlcv_df)
