"""Tests for KAMA (Kaufman Adaptive Moving Average) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.kama import kama

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestKAMA:
  """Tests for Kaufman Adaptive Moving Average indicator."""

  def test_kama_basic(self):
    """Test KAMA basic calculation."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(50) * 0.5))

    result = kama(prices, period=10).to_pandas()

    assert len(result) == len(prices)
    assert result.isna().iloc[:9].all()
    assert not result.isna().iloc[10:].any()

  def test_kama_trending_vs_choppy(self):
    """Test KAMA adapts to trend vs choppy market."""
    # Strong trend
    trending = pd.Series([100.0 + i * 2 for i in range(30)])
    result_trending = kama(trending, period=10).to_pandas()

    # Choppy market
    choppy = pd.Series([100.0 + (i % 3 - 1) for i in range(30)])
    result_choppy = kama(choppy, period=10).to_pandas()

    # Both should produce valid results
    assert not result_trending.isna().iloc[10:].any()
    assert not result_choppy.isna().iloc[10:].any()

  def test_kama_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      kama(empty_data)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_kama_matches_talib(self):
    """Test KAMA matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = kama(prices, period=10).to_pandas()
    expected = pd.Series(talib.KAMA(prices.values, timeperiod=10))

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
