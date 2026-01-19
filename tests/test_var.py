"""Tests for VAR indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.var import var

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


@pytest.fixture
def price_data():
  """Generate price test data."""
  np.random.seed(42)
  return pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))


class TestVar:
  """Tests for VAR indicator."""

  def test_var_basic(self, price_data):
    """Test VAR calculation."""
    result = var(price_data, period=5).to_pandas()

    assert len(result) == len(price_data)
    assert result.isna().sum() == 4  # period - 1

  def test_var_formula(self):
    """Verify VAR formula."""
    data = pd.Series([100.0, 102.0, 104.0, 103.0, 105.0])

    result = var(data, period=3, nbdev=1.0).to_pandas()

    # Manual calc at index 2 (100, 102, 104)
    expected_var = data[:3].var(ddof=0)
    assert result.iloc[2] == pytest.approx(expected_var)

  def test_var_nbdev(self):
    """Test VAR with nbdev multiplier."""
    data = pd.Series([100.0, 102.0, 104.0, 103.0, 105.0])

    result1 = var(data, period=3, nbdev=1.0).to_pandas()
    result2 = var(data, period=3, nbdev=2.0).to_pandas()

    # result2 should be 2x result1
    valid = result1.notna()
    np.testing.assert_allclose(result2[valid].values, result1[valid].values * 2)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_var_matches_talib(self, price_data):
    """Test VAR matches TA-Lib."""
    period = 5
    result = var(price_data, period=period, nbdev=1.0).to_pandas()
    expected = talib.VAR(price_data.values, timeperiod=period, nbdev=1.0)

    valid = ~np.isnan(expected)
    np.testing.assert_allclose(result.values[valid], expected[valid], rtol=1e-9)
