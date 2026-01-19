"""Tests for SAR (Parabolic SAR) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.sar import sar

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestSAR:
  """Tests for Parabolic SAR indicator."""

  def test_sar_basic(self):
    """Test SAR basic calculation."""
    np.random.seed(42)
    n = 50
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)

    high_s = pd.Series(high)
    low_s = pd.Series(low)

    result = sar(high_s, low_s).to_pandas()

    assert len(result) == n
    assert result.isna().iloc[0]  # First is NaN
    assert not result.isna().iloc[2:].any()

  def test_sar_uptrend(self):
    """Test SAR tracks below price in uptrend."""
    # Create clear uptrend
    n = 30
    close = np.array([100.0 + i * 2 for i in range(n)])
    high = close + 1
    low = close - 1

    result = sar(pd.Series(high), pd.Series(low)).to_pandas()

    # SAR should be below lows in uptrend
    valid_sar = result.dropna()
    valid_low = low[result.notna()]
    assert (valid_sar.values < valid_low).all()

  def test_sar_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      sar(empty, empty)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_sar_matches_talib(self):
    """Test SAR matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)

    high_s = pd.Series(high)
    low_s = pd.Series(low)

    result = sar(high_s, low_s, acceleration=0.02, maximum=0.2).to_pandas()
    expected = pd.Series(talib.SAR(high, low, acceleration=0.02, maximum=0.2))

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
