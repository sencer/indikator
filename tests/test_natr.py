"""Tests for NATR (Normalized ATR) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.natr import natr

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestNATR:
  """Tests for Normalized ATR indicator."""

  def test_natr_basic(self):
    """Test NATR basic calculation."""
    np.random.seed(42)
    n = 50
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)

    result = natr(
      pd.Series(high), pd.Series(low), pd.Series(close), period=14
    ).to_pandas()

    assert len(result) == n
    assert result.isna().iloc[:14].all()
    assert not result.isna().iloc[15:].any()
    # NATR should be positive (it's a percentage)
    assert (result.dropna() >= 0).all()

  def test_natr_is_percentage(self):
    """Test NATR returns percentage values."""
    # Create data with known volatility relative to price
    n = 30
    close = np.array([100.0] * n)
    high = close + 1.0  # 1% range
    low = close - 1.0

    result = natr(
      pd.Series(high), pd.Series(low), pd.Series(close), period=14
    ).to_pandas()

    # NATR should be around 2% (range is 2 out of 100)
    valid = result.dropna()
    assert (valid > 0).all() and (valid < 10).all()

  def test_natr_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      natr(empty, empty, empty)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_natr_matches_talib(self):
    """Test NATR matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)

    result = natr(
      pd.Series(high), pd.Series(low), pd.Series(close), period=14
    ).to_pandas()
    expected = pd.Series(talib.NATR(high, low, close, timeperiod=14))

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
