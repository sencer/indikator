"""Tests for ADX (Average Directional Index) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.adx import adx, adx_with_di

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestADX:
  """Tests for ADX indicator."""

  def test_adx_basic(self):
    """Test ADX basic calculation."""
    np.random.seed(42)
    n = 50
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2

    result = adx_with_di(high, low, close, period=14).to_pandas()

    # Check columns
    assert "adx" in result.columns
    assert "plus_di" in result.columns
    assert "minus_di" in result.columns

    # Check shape
    assert len(result) == len(close)

    # Check ADX and DIs are in range [0, 100] where valid
    valid_adx = result["adx"].dropna()
    assert (valid_adx >= 0).all()
    assert (valid_adx <= 100).all()

    valid_plus_di = result["plus_di"].dropna()
    assert (valid_plus_di >= 0).all()

    valid_minus_di = result["minus_di"].dropna()
    assert (valid_minus_di >= 0).all()

  def test_adx_strong_uptrend(self):
    """Test ADX with strong uptrend."""
    # Strong uptrend should show high ADX and +DI > -DI
    n = 50
    close = pd.Series([100.0 + i * 2 for i in range(n)])
    high = close + 1.0
    low = close - 0.5

    result = adx_with_di(high, low, close, period=14).to_pandas()

    valid = result.dropna()
    if len(valid) > 0:
      # +DI should be greater than -DI in uptrend
      assert (valid["plus_di"] > valid["minus_di"]).any()

  def test_adx_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      adx(empty, empty, empty)

  def test_adx_insufficient_data(self):
    """Test ADX with insufficient data."""
    high = pd.Series([105.0, 106.0])
    low = pd.Series([100.0, 101.0])
    close = pd.Series([102.0, 103.0])

    result = adx(high, low, close, period=14).to_pandas()

    # Should return all NaN for ADX (Series result)
    assert result.isna().all()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_adx_matches_talib(self):
    """Test ADX matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2

    result = adx_with_di(high, low, close, period=14).to_pandas()
    expected_adx = pd.Series(
      talib.ADX(high.values, low.values, close.values, timeperiod=14),
    )
    expected_plus_di = pd.Series(
      talib.PLUS_DI(high.values, low.values, close.values, timeperiod=14),
    )
    expected_minus_di = pd.Series(
      talib.MINUS_DI(high.values, low.values, close.values, timeperiod=14),
    )

    # Compare ADX (non-NaN values)
    valid_mask = result["adx"].notna() & expected_adx.notna()
    np.testing.assert_allclose(
      result["adx"][valid_mask].values,
      expected_adx[valid_mask].values,
      rtol=1e-1,
    )

    # Compare +DI
    valid_mask = result["plus_di"].notna() & expected_plus_di.notna()
    np.testing.assert_allclose(
      result["plus_di"][valid_mask].values,
      expected_plus_di[valid_mask].values,
      rtol=1e-1,
    )

    # Compare -DI
    valid_mask = result["minus_di"].notna() & expected_minus_di.notna()
    np.testing.assert_allclose(
      result["minus_di"][valid_mask].values,
      expected_minus_di[valid_mask].values,
      rtol=1e-1,
    )
