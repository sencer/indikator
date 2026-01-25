"""Weighted Moving Average (WMA) indicator module.

WMA gives more weight to recent prices, with linearly increasing weights.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import WMAResult
from indikator._wma_numba import compute_wma_numba
from indikator.utils import to_numpy


@configurable
@validate
def wma(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> WMAResult:
  """Calculate Weighted Moving Average (WMA).

  WMA assigns linearly increasing weights to prices, with the most
  recent price having the highest weight.

  Formula:
  WMA = (P1*1 + P2*2 + ... + Pn*n) / (1 + 2 + ... + n)

  Interpretation:
  - More responsive than SMA, less than EMA
  - Smooth trend following
  - Good for identifying trend direction

  Features:
  - O(1) rolling update per step (not O(period))
  - Uses weighted/unweighted sum trick for efficiency

  Args:
    data: Input price Series (typically close prices)
    period: Lookback period (default: 20)

  Returns:
    WMAResult with weighted moving average values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = wma(prices, period=5)
  """
  values = to_numpy(data)

  wma_values = compute_wma_numba(values, period)

  return WMAResult(data_index=data.index, wma=wma_values)
