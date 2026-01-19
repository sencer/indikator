"""Tests for AD (Accumulation/Distribution Line) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.ad import ad

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestAD:
  """Tests for Accumulation/Distribution Line indicator."""

  def test_ad_basic(self):
    """Test AD basic calculation."""
    np.random.seed(42)
    n = 50
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000 + 10000)

    result = ad(
      pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume)
    ).to_pandas()

    assert len(result) == n
    # AD has no warmup - all values valid
    assert not result.isna().any()

  def test_ad_cumulative(self):
    """Test AD is cumulative."""
    # Simple case: close at high = CLV = 1, close at low = CLV = -1
    high = pd.Series([110.0, 120.0, 130.0])
    low = pd.Series([100.0, 100.0, 100.0])
    close = pd.Series([110.0, 120.0, 130.0])  # Close at high
    volume = pd.Series([1000.0, 1000.0, 1000.0])

    result = ad(high, low, close, volume).to_pandas()

    # CLV = (2*close - high - low) / (high - low) = 1.0 when close = high
    # AD should be cumulative: 1000, 2000, 3000
    assert pytest.approx(result.iloc[0]) == 1000.0
    assert pytest.approx(result.iloc[1]) == 2000.0
    assert pytest.approx(result.iloc[2]) == 3000.0

  def test_ad_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      ad(empty, empty, empty, empty)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_ad_matches_talib(self):
    """Test AD matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000 + 10000)

    result = ad(
      pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume)
    ).to_pandas()
    expected = pd.Series(talib.AD(high, low, close, volume))

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
