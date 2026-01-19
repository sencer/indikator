"""Tests for WMA (Weighted Moving Average) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.wma import wma

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestWMA:
  """Tests for Weighted Moving Average indicator."""

  def test_wma_basic(self):
    """Test WMA basic calculation."""
    prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = wma(prices, period=3).to_pandas()

    # WMA[2] = (1*1 + 2*2 + 3*3) / (1+2+3) = (1+4+9) / 6 = 14/6 = 2.333...
    assert len(result) == 5
    assert result.isna().iloc[:2].all()
    assert pytest.approx(result.iloc[2], rel=1e-6) == 14.0 / 6.0
    # WMA[3] = (2*1 + 3*2 + 4*3) / 6 = (2+6+12) / 6 = 20/6
    assert pytest.approx(result.iloc[3], rel=1e-6) == 20.0 / 6.0

  def test_wma_weights_recent_more(self):
    """Test WMA weights recent prices more heavily."""
    # Sharp increase at end
    prices = pd.Series([100.0, 100.0, 100.0, 100.0, 150.0])
    result = wma(prices, period=3).to_pandas()

    # Last WMA should be heavily influenced by 150
    # WMA = (100*1 + 100*2 + 150*3) / 6 = (100+200+450)/6 = 125
    assert pytest.approx(result.iloc[4], rel=1e-6) == 125.0

  def test_wma_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      wma(empty_data)

  def test_wma_insufficient_data(self):
    """Test WMA with insufficient data."""
    prices = pd.Series([100.0, 101.0])
    result = wma(prices, period=5).to_pandas()

    assert result.isna().all()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_wma_matches_talib(self):
    """Test WMA matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = wma(prices, period=20).to_pandas()
    expected = pd.Series(talib.WMA(prices.values, timeperiod=20))

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
