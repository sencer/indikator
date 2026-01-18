"""Tests for CCI (Commodity Channel Index) indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.cci import cci

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestCCI:
  """Tests for CCI indicator."""

  def test_cci_basic(self):
    """Test CCI basic calculation."""
    np.random.seed(42)
    n = 50
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2

    result = cci(high, low, close, period=20)

    # Check shape
    assert len(result) == len(close)

    # Check CCI has values after period
    assert result.isna().iloc[: 20 - 1].all()
    assert not result.isna().iloc[19:].any()

  def test_cci_flat_prices(self):
    """Test CCI with flat prices."""
    high = pd.Series([105.0] * 30)
    low = pd.Series([100.0] * 30)
    close = pd.Series([102.5] * 30)  # Middle of range

    result = cci(high, low, close, period=5)

    # CCI should be near 0 for flat typical price
    valid = result.dropna()
    assert (abs(valid) < 1).all()

  def test_cci_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="not empty"):
      cci(empty, empty, empty)

  def test_cci_insufficient_data(self):
    """Test CCI with insufficient data."""
    high = pd.Series([105.0, 106.0])
    low = pd.Series([100.0, 101.0])
    close = pd.Series([102.0, 103.0])

    result = cci(high, low, close, period=20)

    # Should return all NaN
    assert result.isna().all()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_cci_matches_talib(self):
    """Test CCI matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2

    result = cci(high, low, close, period=20)
    expected = pd.Series(
      talib.CCI(high.values, low.values, close.values, timeperiod=20),
    )

    # Compare non-NaN values
    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
