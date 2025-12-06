"""Tests for MACD indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.macd import macd


class TestMACD:
  """Tests for MACD indicator."""

  def test_macd_basic(self):
    """Test MACD basic calculation."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 4
    )

    result = macd(prices, fast_period=5, slow_period=10, signal_period=3)

    # Check columns
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_histogram" in result.columns

    # Check shape
    assert len(result) == len(prices)

    # Check MACD is calculated
    assert not result["macd"].isna().all()
    assert not result["macd_signal"].isna().all()
    assert not result["macd_histogram"].isna().all()

  def test_macd_dataframe(self):
    """Test MACD with DataFrame input."""
    data = pd.DataFrame({
      "close": [
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
      ],
      "volume": [100, 200, 150, 180, 220, 190, 210, 230, 200, 240],
    })
    # Pass Series directly
    result = macd(data["close"])
    assert isinstance(result, pd.DataFrame)
    assert "macd" in result.columns

  def test_macd_validation_fast_slow(self):
    """Test MACD validation of fast/slow periods."""
    prices = pd.Series([100.0, 102.0, 101.0])

    with pytest.raises(ValueError, match="must be <"):
      macd(prices, fast_period=20, slow_period=10)

  def test_macd_empty_data(self) -> None:
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Data must not be empty"):
      macd(empty_data)

  def test_macd_histogram(self):
    """Test MACD histogram calculation."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 5
    )

    result = macd(prices, fast_period=5, slow_period=10, signal_period=3)

    # Histogram should equal MACD - Signal (where both are not NaN)
    valid_mask = result["macd"].notna() & result["macd_signal"].notna()
    expected_histogram = (
      result.loc[valid_mask, "macd"] - result.loc[valid_mask, "macd_signal"]
    )
    pd.testing.assert_series_equal(
      result.loc[valid_mask, "macd_histogram"],
      expected_histogram,
      check_names=False,
      atol=1e-10,
    )

  def test_macd_invalid_input(self):
    """Test MACD with invalid input."""
    # Infinite values
    data = pd.Series([100.0, np.inf, 102.0])
    with pytest.raises(ValueError, match="must be finite"):
      macd(data)

  def test_macd_with_inf(self):
    """Test MACD with Inf values."""
    prices = pd.Series([100.0, 102.0, np.inf, 103.0, 105.0] * 10)

    with pytest.raises(ValueError, match="must be finite"):
      macd(prices)
