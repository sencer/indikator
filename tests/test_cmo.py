"""Tests for CMO (Chande Momentum Oscillator) indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.cmo import cmo

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestCMO:
  """Tests for CMO indicator."""

  def test_cmo_basic(self):
    """Test CMO basic calculation."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 3,
    )

    result = cmo(prices, period=5)

    assert len(result) == len(prices)
    assert result.name == "cmo"
    assert result.notna().any()

  def test_cmo_range(self):
    """Test CMO is in valid range."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = cmo(prices, period=14)

    valid = result.dropna()
    assert (valid >= -100).all()
    assert (valid <= 100).all()

  def test_cmo_uptrend(self):
    """Test CMO for strong uptrend."""
    prices = pd.Series([100.0 + i for i in range(50)])

    result = cmo(prices, period=5)

    valid = result.dropna()
    # Strong uptrend should give high positive CMO
    assert (valid > 50).all()

  def test_cmo_downtrend(self):
    """Test CMO for strong downtrend."""
    prices = pd.Series([100.0 - i * 0.5 for i in range(50)])

    result = cmo(prices, period=5)

    valid = result.dropna()
    # Strong downtrend should give low negative CMO (near -100)
    assert (valid < -50).all()

  def test_cmo_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="not empty"):
      cmo(empty_data)

  def test_cmo_insufficient_data(self):
    """Test CMO with insufficient data."""
    prices = pd.Series([100.0, 101.0, 102.0])
    result = cmo(prices, period=5)

    assert result.isna().all()

  def test_cmo_with_inf(self):
    """Test CMO with Inf values."""
    prices = pd.Series([100.0, 102.0, np.inf, 103.0, 105.0] * 5)

    with pytest.raises(ValueError, match="must be finite"):
      cmo(prices)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_cmo_matches_talib(self):
    """Test CMO matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = cmo(prices, period=14)
    expected = pd.Series(talib.CMO(prices.values, timeperiod=14))

    # Compare non-NaN values
    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
