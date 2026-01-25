"""Stochastic RSI (STOCHRSI) indicator module.

Applies Stochastic formula to RSI for more sensitive momentum readings.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import StochRSIResult
from indikator._stochrsi_numba import compute_stochrsi_numba
from indikator.utils import to_numpy


@configurable
@validate
def stochrsi(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  rsi_period: Hyper[int, Ge[2]] = 14,
  stoch_period: Hyper[int, Ge[2]] = 14,
  k_period: Hyper[int, Ge[1]] = 3,
  d_period: Hyper[int, Ge[1]] = 3,
) -> StochRSIResult:
  """Calculate Stochastic RSI (STOCHRSI).

  StochRSI applies the Stochastic oscillator formula to RSI values,
  creating a more sensitive indicator that oscillates between 0-100.

  Formula:
  StochRSI = (RSI - min(RSI, n)) / (max(RSI, n) - min(RSI, n)) * 100
  %K = SMA(StochRSI, k_period)
  %D = SMA(%K, d_period)

  Interpretation:
  - StochRSI > 80: Overbought
  - StochRSI < 20: Oversold
  - More sensitive than RSI alone
  - K/D crossovers for signals

  Features:
  - Fused RSI + Stochastic computation
  - Lazy rescan for min/max (amortized O(N))

  Args:
    data: Input price Series (typically close prices)
    rsi_period: RSI lookback period (default: 14)
    stoch_period: Stochastic lookback on RSI (default: 14)
    k_period: %K SMA smoothing period (default: 3)
    d_period: %D SMA smoothing period (default: 3)

  Returns:
    StochRSIResult with %K and %D values

  Example:
    >>> prices = pd.Series([...])
    >>> result = stochrsi(prices, rsi_period=14, stoch_period=14)
  """
  values = to_numpy(data)

  k_values, d_values = compute_stochrsi_numba(
    values, rsi_period, stoch_period, k_period, d_period
  )

  return StochRSIResult(data_index=data.index, stochrsi_k=k_values, stochrsi_d=d_values)
