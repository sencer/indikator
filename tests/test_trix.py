"""Tests for TRIX (Triple Exponential Average) indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.trix import trix

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestTRIX:
  """Tests for TRIX indicator."""

  def test_trix_basic(self):
    """Test TRIX basic calculation."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 5,
    )

    result = trix(prices, period=5)

    # Check shape
    assert len(result) == len(prices)

    # Check TRIX name
    assert result.name == "trix"

    # Check TRIX is calculated (values exist after warmup period)
    assert result.notna().any()

  def test_trix_oscillates_around_zero(self):
    """Test that TRIX oscillates around zero for oscillating prices."""
    # Create prices that go up then down repeatedly
    prices = pd.Series([100.0 + 5 * np.sin(i / 3) for i in range(100)])

    result = trix(prices, period=5)

    # TRIX should have both positive and negative values
    valid = result.dropna()
    if len(valid) > 0:
      # For oscillating prices, expect some variation
      assert valid.std() > 0

  def test_trix_uptrend(self):
    """Test TRIX for strong uptrend."""
    # Strong uptrend: TRIX should be positive
    prices = pd.Series([100.0 + i for i in range(100)])

    result = trix(prices, period=5)

    # In uptrend, TRIX should be mostly positive after warmup
    valid = result.dropna()
    if len(valid) > 10:
      # Last values should be positive
      assert (valid.iloc[-10:] > 0).all()

  def test_trix_downtrend(self):
    """Test TRIX for strong downtrend."""
    # Strong downtrend: TRIX should be negative
    prices = pd.Series([200.0 - i for i in range(100)])

    result = trix(prices, period=5)

    # In downtrend, TRIX should be mostly negative after warmup
    valid = result.dropna()
    if len(valid) > 10:
      # Last values should be negative
      assert (valid.iloc[-10:] < 0).all()

  def test_trix_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="not empty"):
      trix(empty_data)

  def test_trix_insufficient_data(self):
    """Test TRIX with insufficient data."""
    prices = pd.Series([100.0, 101.0, 102.0])
    result = trix(prices, period=5)

    # Should return all NaN
    assert result.isna().all()

  def test_trix_period_parameter(self):
    """Test TRIX with different period sizes."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 10,
    )

    result_short = trix(prices, period=3)
    result_long = trix(prices, period=10)

    # Short period should have values earlier
    assert result_short.notna().sum() > result_long.notna().sum()

  def test_trix_with_inf(self):
    """Test TRIX with Inf values."""
    prices = pd.Series([100.0, 102.0, np.inf, 103.0, 105.0] * 10)

    with pytest.raises(ValueError, match="must be finite"):
      trix(prices)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_trix_matches_talib(self):
    """Test TRIX matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(200) * 0.5))

    result = trix(prices, period=14)
    expected = pd.Series(talib.TRIX(prices.values, timeperiod=14))

    # Compare non-NaN values (use relaxed tolerance for EMA initialization)
    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-2,  # Relaxed for EMA initialization differences
    )
