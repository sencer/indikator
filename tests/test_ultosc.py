"""Tests for ULTOSC (Ultimate Oscillator) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.ultosc import ultosc

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestULTOSC:
  """Tests for Ultimate Oscillator indicator."""

  def test_ultosc_basic(self):
    """Test ULTOSC basic calculation."""
    np.random.seed(42)
    n = 60
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)

    result = ultosc(
      pd.Series(high),
      pd.Series(low),
      pd.Series(close),
      period1=7,
      period2=14,
      period3=28,
    ).to_pandas()

    assert len(result) == n
    # Need period3 bars for first valid value
    assert result.isna().iloc[:28].all()
    assert not result.isna().iloc[29:].any()

  def test_ultosc_range(self):
    """Test ULTOSC stays in 0-100 range."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)

    result = ultosc(pd.Series(high), pd.Series(low), pd.Series(close)).to_pandas()

    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()

  def test_ultosc_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      ultosc(empty, empty, empty)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_ultosc_matches_talib(self):
    """Test ULTOSC matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)

    result = ultosc(
      pd.Series(high),
      pd.Series(low),
      pd.Series(close),
      period1=7,
      period2=14,
      period3=28,
    ).to_pandas()
    expected = pd.Series(
      talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    )

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-6,
    )
