"""Tests for TEMA (Triple Exponential Moving Average) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.tema import tema

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestTEMA:
  """Tests for TEMA indicator."""

  def test_tema_basic(self):
    """Test TEMA basic calculation."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(80) * 0.5))

    result = tema(prices, period=10).to_pandas()

    assert len(result) == len(prices)
    # TEMA needs 3*period - 3 bars for first valid value
    lookback = 3 * 10 - 3
    assert result.isna().iloc[:lookback].all()
    assert not result.isna().iloc[lookback:].any()

  def test_tema_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      tema(empty_data)

  def test_tema_insufficient_data(self):
    """Test TEMA with insufficient data."""
    prices = pd.Series([100.0] * 10)
    result = tema(prices, period=10).to_pandas()

    # Need 3*10-3 = 27 bars, but only have 10
    assert result.isna().all()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_tema_matches_talib(self):
    """Test TEMA matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(150) * 0.5))

    result = tema(prices, period=20).to_pandas()
    expected = pd.Series(talib.TEMA(prices.values, timeperiod=20))

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
