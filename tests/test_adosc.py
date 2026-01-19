"""Tests for ADOSC (Chaikin Oscillator) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.adosc import adosc

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestADOSC:
  """Tests for Accumulation/Distribution Oscillator indicator."""

  def test_adosc_basic(self):
    """Test ADOSC basic calculation."""
    np.random.seed(42)
    n = 50
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000 + 10000)

    result = adosc(
      pd.Series(high),
      pd.Series(low),
      pd.Series(close),
      pd.Series(volume),
      fast_period=3,
      slow_period=10,
    ).to_pandas()

    assert len(result) == n
    # Need slow_period bars for first valid value
    assert result.isna().iloc[:9].all()
    assert not result.isna().iloc[10:].any()

  def test_adosc_oscillates(self):
    """Test ADOSC oscillates around zero."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000 + 10000)

    result = adosc(
      pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume)
    ).to_pandas()

    valid = result.dropna()
    # Should have both positive and negative values
    assert (valid > 0).any() and (valid < 0).any()

  def test_adosc_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      adosc(empty, empty, empty, empty)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_adosc_matches_talib(self):
    """Test ADOSC matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000 + 10000)

    result = adosc(
      pd.Series(high),
      pd.Series(low),
      pd.Series(close),
      pd.Series(volume),
      fast_period=3,
      slow_period=10,
    ).to_pandas()
    expected = pd.Series(
      talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    )

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
