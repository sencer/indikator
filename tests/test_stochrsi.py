"""Tests for STOCHRSI (Stochastic RSI) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.stochrsi import stochrsi

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestStochRSI:
  """Tests for Stochastic RSI indicator."""

  def test_stochrsi_basic(self):
    """Test StochRSI basic calculation."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = stochrsi(prices, rsi_period=14, stoch_period=14, k_period=3, d_period=3)
    k = result.stochrsi_k
    d = result.stochrsi_d

    assert len(k) == len(prices)
    assert len(d) == len(prices)

    # Check valid values exist after lookback
    lookback = 14 + 14 + 3 + 3 - 3
    assert not np.isnan(k[lookback:]).all()
    assert not np.isnan(d[lookback + 2 :]).all()

  def test_stochrsi_range(self):
    """Test StochRSI stays in 0-100 range."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(150) * 0.5))

    result = stochrsi(prices)
    k = result.stochrsi_k
    d = result.stochrsi_d

    valid_k = k[~np.isnan(k)]
    valid_d = d[~np.isnan(d)]

    assert (valid_k >= 0).all() and (valid_k <= 100.001).all()
    assert (valid_d >= 0).all() and (valid_d <= 100.001).all()

  def test_stochrsi_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      stochrsi(empty_data)

  def test_stochrsi_insufficient_data(self):
    """Test StochRSI with insufficient data."""
    prices = pd.Series([100.0] * 20)
    result = stochrsi(prices)

    # All should be NaN with insufficient data
    assert np.isnan(result.stochrsi_k).all()
    assert np.isnan(result.stochrsi_d).all()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_stochrsi_matches_talib(self):
    """Test StochRSI matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = stochrsi(prices, rsi_period=14, stoch_period=14, k_period=14, d_period=3)

    # TA-Lib STOCHRSI
    expected_k, expected_d = talib.STOCHRSI(
      prices.values, timeperiod=14, fastk_period=14, fastd_period=3, fastd_matype=0
    )

    # Compare fastk values
    valid_k = ~np.isnan(result.stochrsi_k) & ~np.isnan(expected_k)
    np.testing.assert_allclose(
      result.stochrsi_k[valid_k],
      expected_k[valid_k],
      rtol=1e-10,
    )

    # Compare fastd values
    valid_d = ~np.isnan(result.stochrsi_d) & ~np.isnan(expected_d)
    np.testing.assert_allclose(
      result.stochrsi_d[valid_d],
      expected_d[valid_d],
      rtol=1e-10,
    )
