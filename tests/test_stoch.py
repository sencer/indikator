"""Tests for Stochastic Oscillator indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.stoch import stoch

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestStoch:
  """Tests for Stochastic Oscillator indicator."""

  def test_stoch_basic(self):
    """Test Stochastic basic calculation."""
    np.random.seed(42)
    n = 50
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2

    result = stoch(high, low, close, k_period=14, k_slowing=3, d_period=3)

    # Check columns
    assert "stoch_k" in result.columns
    assert "stoch_d" in result.columns

    # Check shape
    assert len(result) == len(close)

    # Check %K is in range [0, 100] where valid
    valid_k = result["stoch_k"].dropna()
    assert (valid_k >= 0).all()
    assert (valid_k <= 100).all()

    # Check %D is in range [0, 100] where valid
    valid_d = result["stoch_d"].dropna()
    assert (valid_d >= 0).all()
    assert (valid_d <= 100).all()

  def test_stoch_overbought_oversold(self):
    """Test Stochastic with extreme price levels."""
    # Price at high of range - should show overbought
    high = pd.Series([110.0] * 20)
    low = pd.Series([100.0] * 20)
    close = pd.Series([109.0] * 20)  # Close near high

    result = stoch(high, low, close, k_period=5, k_slowing=1, d_period=1)

    # %K should be high (close to 100)
    valid_k = result["stoch_k"].dropna()
    assert (valid_k > 80).all()

  def test_stoch_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="not empty"):
      stoch(empty, empty, empty)

  def test_stoch_insufficient_data(self):
    """Test Stochastic with insufficient data."""
    high = pd.Series([105.0, 106.0])
    low = pd.Series([100.0, 101.0])
    close = pd.Series([102.0, 103.0])

    result = stoch(high, low, close, k_period=14)

    # Should return all NaN
    assert result["stoch_k"].isna().all()
    assert result["stoch_d"].isna().all()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_stoch_matches_talib(self):
    """Test Stochastic matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2

    result = stoch(high, low, close, k_period=14, k_slowing=3, d_period=3)
    slowk, _slowd = talib.STOCH(
      high.values,
      low.values,
      close.values,
      fastk_period=14,
      slowk_period=3,
      slowd_period=3,
    )

    # Compare non-NaN values
    valid_mask = result["stoch_k"].notna() & ~np.isnan(slowk)
    np.testing.assert_allclose(
      result["stoch_k"][valid_mask].values,
      slowk[valid_mask],
      rtol=1e-5,
    )
