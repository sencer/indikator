import numpy as np
import pandas as pd
import pytest
import talib

from indikator.cci import cci
from indikator.cmo import cmo
from indikator.mfi import mfi
from indikator.willr import willr


@pytest.mark.parametrize("period", [14, 30])
def test_willr_matches_talib(period):
  np.random.seed(42)
  # WILLR inputs: High, Low, Close
  high = pd.Series(np.random.randn(100).cumsum() + 100, name="high")
  low = high - np.abs(np.random.randn(100))
  close = (high + low) / 2 + np.random.randn(100) * 0.1

  # Ensure High >= Low
  high = np.maximum(high, low)
  low = np.minimum(high, low)

  result = willr(high, low, close, period=period)
  expected = talib.WILLR(high.values, low.values, close.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.to_pandas(), pd.Series(expected, index=high.index, name="willr"), atol=1e-10
  )


@pytest.mark.parametrize("period", [14, 30])
def test_cci_matches_talib(period):
  np.random.seed(42)
  high = pd.Series(np.random.randn(100).cumsum() + 100, name="high")
  low = high - np.abs(np.random.randn(100))
  close = (high + low) / 2 + np.random.randn(100) * 0.1

  high = np.maximum(high, low)
  low = np.minimum(high, low)

  result = cci(high, low, close, period=period)
  expected = talib.CCI(high.values, low.values, close.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.to_pandas(), pd.Series(expected, index=high.index, name="cci"), atol=1e-10
  )


@pytest.mark.parametrize("period", [14, 30])
def test_cmo_matches_talib(period):
  np.random.seed(42)
  close = pd.Series(np.random.randn(100).cumsum() + 100, name="close")

  result = cmo(close, period=period)
  expected = talib.CMO(close.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.to_pandas(), pd.Series(expected, index=close.index, name="cmo"), atol=1e-10
  )


@pytest.mark.parametrize("period", [14, 30])
def test_mfi_matches_talib(period):
  np.random.seed(42)
  high = pd.Series(np.random.randn(100).cumsum() + 100, name="high")
  low = high - np.abs(np.random.randn(100))
  close = (high + low) / 2
  volume = pd.Series(np.abs(np.random.randn(100) * 1000 + 10000), name="volume")

  high = np.maximum(high, low)
  low = np.minimum(high, low)

  result = mfi(high, low, close, volume, period=period)
  expected = talib.MFI(
    high.values, low.values, close.values, volume.values, timeperiod=period
  )

  pd.testing.assert_series_equal(
    result.to_pandas(), pd.Series(expected, index=high.index, name="mfi"), atol=1e-10
  )
