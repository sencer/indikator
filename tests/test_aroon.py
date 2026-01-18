"""Tests for AROON indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.aroon import aroon

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestAROON:
  """Tests for AROON indicator."""

  def test_aroon_basic(self):
    """Test AROON basic calculation."""
    high = pd.Series([105, 106, 104, 108, 107, 109, 108, 110] * 5)
    low = pd.Series([100, 101, 99, 103, 102, 104, 103, 105] * 5)

    result = aroon(high, low, period=5)

    assert "aroon_up" in result.columns
    assert "aroon_down" in result.columns
    assert "aroon_osc" in result.columns
    assert len(result) == len(high)

  def test_aroon_range(self):
    """Test AROON is in valid range."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = pd.Series(close + np.random.rand(n) * 2)
    low = pd.Series(close - np.random.rand(n) * 2)

    result = aroon(high, low, period=14)

    valid_up = result["aroon_up"].dropna()
    valid_down = result["aroon_down"].dropna()
    valid_osc = result["aroon_osc"].dropna()

    assert (valid_up >= 0).all() and (valid_up <= 100).all()
    assert (valid_down >= 0).all() and (valid_down <= 100).all()
    assert (valid_osc >= -100).all() and (valid_osc <= 100).all()

  def test_aroon_uptrend(self):
    """Test AROON for strong uptrend."""
    high = pd.Series([100.0 + i for i in range(50)])
    low = pd.Series([99.0 + i for i in range(50)])

    result = aroon(high, low, period=10)

    valid = result.dropna()
    # Strong uptrend: aroon_up should be high
    assert (valid["aroon_up"].iloc[-10:] > 70).all()

  def test_aroon_downtrend(self):
    """Test AROON for strong downtrend."""
    high = pd.Series([150.0 - i for i in range(50)])
    low = pd.Series([149.0 - i for i in range(50)])

    result = aroon(high, low, period=10)

    valid = result.dropna()
    # Strong downtrend: aroon_down should be high
    assert (valid["aroon_down"].iloc[-10:] > 70).all()

  def test_aroon_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="not empty"):
      aroon(empty, empty)

  def test_aroon_insufficient_data(self):
    """Test AROON with insufficient data."""
    high = pd.Series([105.0, 106.0, 104.0])
    low = pd.Series([100.0, 101.0, 99.0])

    result = aroon(high, low, period=10)
    assert result["aroon_up"].isna().all()

  def test_aroon_with_inf(self):
    """Test AROON with Inf values."""
    high = pd.Series([100.0, 102.0, np.inf, 103.0, 105.0] * 10)
    low = pd.Series([99.0, 101.0, 98.0, 102.0, 104.0] * 10)

    with pytest.raises(ValueError, match="must be finite"):
      aroon(high, low)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_aroon_matches_talib(self):
    """Test AROON matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = pd.Series(close + np.random.rand(n) * 2)
    low = pd.Series(close - np.random.rand(n) * 2)

    result = aroon(high, low, period=14)
    expected_down, expected_up = talib.AROON(high.values, low.values, timeperiod=14)

    # Compare non-NaN values
    valid_mask = result["aroon_up"].notna() & ~np.isnan(expected_up)
    np.testing.assert_allclose(
      result["aroon_up"][valid_mask].values,
      expected_up[valid_mask],
      rtol=1e-10,
    )
    np.testing.assert_allclose(
      result["aroon_down"][valid_mask].values,
      expected_down[valid_mask],
      rtol=1e-10,
    )
