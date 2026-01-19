"""Tests for DEMA (Double Exponential Moving Average) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.dema import dema

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestDEMA:
  """Tests for DEMA indicator."""

  def test_dema_basic(self):
    """Test DEMA basic calculation."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(50) * 0.5))

    result = dema(prices, period=10).to_pandas()

    assert len(result) == len(prices)
    # Check warmup period is NaN
    assert result.isna().iloc[:18].all()  # 2*period - 2
    # Values after warmup should not be NaN
    assert not result.isna().iloc[19:].any()

  def test_dema_trend_following(self):
    """Test DEMA follows trend faster than EMA."""
    # Strong uptrend
    prices = pd.Series([100.0 + i * 2 for i in range(30)])

    result = dema(prices, period=5).to_pandas()

    # DEMA should be closer to current price than simple average
    valid_results = result.dropna()
    assert len(valid_results) > 0
    # Last DEMA should be close to last price (less lag)
    assert valid_results.iloc[-1] > prices.iloc[-3]

  def test_dema_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      dema(empty_data)

  def test_dema_insufficient_data(self):
    """Test DEMA with insufficient data for period."""
    prices = pd.Series([100.0, 101.0, 102.0])
    result = dema(prices, period=10).to_pandas()

    assert result.isna().all()

  def test_dema_with_nan(self):
    """Test DEMA rejects NaN input."""
    prices = pd.Series([100.0, 102.0, np.nan, 104.0, 105.0])
    with pytest.raises((ValueError, ValidationError), match="Finite"):
      dema(prices)

  def test_dema_with_inf(self):
    """Test DEMA rejects Inf input."""
    prices = pd.Series([100.0, 102.0, np.inf, 104.0, 105.0])
    with pytest.raises((ValueError, ValidationError), match="Finite"):
      dema(prices)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_dema_matches_talib(self):
    """Test DEMA matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = dema(prices, period=20).to_pandas()
    expected = pd.Series(talib.DEMA(prices.values, timeperiod=20))

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
