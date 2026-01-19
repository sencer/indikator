"""Tests for MOM (Momentum) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.mom import mom

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestMOM:
  """Tests for Momentum indicator."""

  def test_mom_basic(self):
    """Test MOM basic calculation."""
    prices = pd.Series([100.0, 102.0, 105.0, 103.0, 108.0, 110.0, 107.0, 112.0])

    result = mom(prices, period=3).to_pandas()

    assert len(result) == len(prices)
    # First 3 values should be NaN
    assert result.isna().iloc[:3].all()
    # MOM[3] = 103 - 100 = 3
    assert pytest.approx(result.iloc[3]) == 3.0
    # MOM[4] = 108 - 102 = 6
    assert pytest.approx(result.iloc[4]) == 6.0

  def test_mom_uptrend(self):
    """Test MOM in uptrend."""
    prices = pd.Series([100.0 + i * 2 for i in range(20)])
    result = mom(prices, period=5).to_pandas()

    # In steady uptrend of 2 per bar, MOM should be 10 (5 * 2)
    valid_results = result.dropna()
    assert all(valid_results == 10.0)

  def test_mom_downtrend(self):
    """Test MOM in downtrend."""
    prices = pd.Series([100.0 - i * 2 for i in range(20)])
    result = mom(prices, period=5).to_pandas()

    valid_results = result.dropna()
    assert all(valid_results == -10.0)

  def test_mom_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      mom(empty_data)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_mom_matches_talib(self):
    """Test MOM matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = mom(prices, period=10).to_pandas()
    expected = pd.Series(talib.MOM(prices.values, timeperiod=10))

    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
