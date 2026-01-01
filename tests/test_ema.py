"""Tests for EMA (Exponential Moving Average) indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.ema import ema

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestEMA:
  """Tests for EMA indicator."""

  def test_ema_basic(self):
    """Test EMA basic calculation."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 2
    )

    result = ema(prices, period=5)

    # Check shape
    assert len(result) == len(prices)

    # Check EMA is calculated after period
    assert result.isna().iloc[: 5 - 1].all()  # First 4 bars should be NaN
    assert not result.isna().iloc[4:].any()  # From bar 5 should have values

  def test_ema_more_weight_to_recent(self):
    """Test that EMA gives more weight to recent prices."""
    # Prices with sudden spike at the end
    prices_spike = pd.Series([100.0] * 9 + [110.0])

    result = ema(prices_spike, period=5)

    # EMA should react to spike (more than SMA would)
    # After 9 bars of 100, EMA should be close to 100
    # After spike to 110, EMA should jump up noticeably
    assert result.iloc[-1] > result.iloc[-2]

  def test_ema_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="non-empty"):
      ema(empty_data)

  def test_ema_insufficient_data(self):
    """Test EMA with insufficient data."""
    prices = pd.Series([100.0, 101.0])
    result = ema(prices, period=5)

    # Should return all NaN
    assert result.isna().all()

  def test_ema_period_parameter(self):
    """Test EMA with different period sizes."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 3
    )

    result_short = ema(prices, period=3)
    result_long = ema(prices, period=10)

    # Short period should have values earlier
    assert result_short.notna().sum() > result_long.notna().sum()

    # Short period should be more responsive (higher variance)
    short_std = result_short.dropna().std()
    long_std = result_long.dropna().std()
    assert short_std > long_std

  def test_ema_with_inf(self):
    """Test EMA with Inf values."""
    prices = pd.Series([100.0, 102.0, np.inf, 103.0, 105.0] * 5)

    with pytest.raises(ValueError, match="must be finite"):
      ema(prices)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_ema_matches_talib(self):
    """Test EMA matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = ema(prices, period=14)
    expected = pd.Series(talib.EMA(prices.values, timeperiod=14))

    # Compare non-NaN values (TA-Lib may have different NaN handling)
    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
