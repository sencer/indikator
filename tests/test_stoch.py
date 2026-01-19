"""Tests for Stochastic Oscillator."""

import numpy as np
import pandas as pd
import pytest

from indikator.stoch import stoch, stochf

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestStoch:
  """Tests for Stochastic Oscillator (STOCH and STOCHF)."""

  def test_stoch_basic(self):
    """Test STOCH basic calculation."""
    high = pd.Series([10.0, 12.0, 15.0, 14.0, 13.0])
    low = pd.Series([8.0, 9.0, 10.0, 11.0, 10.0])
    close = pd.Series([9.0, 11.0, 14.0, 12.0, 11.0])

    result = stoch(high, low, close).to_pandas()
    assert "stoch_k" in result.columns
    assert "stoch_d" in result.columns
    # Basic check: stoch should be between 0 and 100
    valid_k = result["stoch_k"].dropna()
    assert (valid_k >= 0).all() and (valid_k <= 100).all()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_stoch_matches_talib(self):
    """Test STOCH matches TA-Lib."""
    np.random.seed(42)
    high = pd.Series(np.random.uniform(105, 110, 100))
    low = pd.Series(np.random.uniform(95, 100, 100))
    close = pd.Series(np.random.uniform(95, 110, 100))

    result = stoch(high, low, close, k_period=5, k_slowing=3, d_period=3).to_pandas()

    exp_k, exp_d = talib.STOCH(
      high.values,
      low.values,
      close.values,
      fastk_period=5,
      slowk_period=3,
      slowk_matype=0,
      slowd_period=3,
      slowd_matype=0,
    )

    valid_mask_k = result["stoch_k"].notna() & np.isfinite(exp_k)
    np.testing.assert_allclose(
      result["stoch_k"][valid_mask_k].values, exp_k[valid_mask_k], rtol=1e-10
    )

    valid_mask_d = result["stoch_d"].notna() & np.isfinite(exp_d)
    np.testing.assert_allclose(
      result["stoch_d"][valid_mask_d].values, exp_d[valid_mask_d], rtol=1e-10
    )

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_stochf_matches_talib(self):
    """Test STOCHF (Fast Stochastic) matches TA-Lib."""
    np.random.seed(42)
    high = pd.Series(np.random.uniform(105, 110, 100))
    low = pd.Series(np.random.uniform(95, 100, 100))
    close = pd.Series(np.random.uniform(95, 110, 100))

    result = stochf(high, low, close, fastk_period=5, fastd_period=3).to_pandas()

    exp_k, exp_d = talib.STOCHF(
      high.values,
      low.values,
      close.values,
      fastk_period=5,
      fastd_period=3,
      fastd_matype=0,
    )

    valid_mask_k = result["stoch_k"].notna() & np.isfinite(exp_k)
    np.testing.assert_allclose(
      result["stoch_k"][valid_mask_k].values, exp_k[valid_mask_k], rtol=1e-10
    )

    valid_mask_d = result["stoch_d"].notna() & np.isfinite(exp_d)
    np.testing.assert_allclose(
      result["stoch_d"][valid_mask_d].values, exp_d[valid_mask_d], rtol=1e-10
    )
