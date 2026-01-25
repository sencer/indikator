"""Double Exponential Moving Average (DEMA) indicator module.

DEMA reduces lag compared to traditional EMA by using a combination
of single and double-smoothed EMAs.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._dema_numba import compute_dema_numba
from indikator._results import DEMAResult
from indikator.utils import to_numpy


@configurable
@validate
def dema(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> DEMAResult:
  """Calculate Double Exponential Moving Average (DEMA).

  DEMA reduces the lag inherent in EMAs by applying a correction factor
  based on the difference between EMA and EMA of EMA.

  Formula:
  DEMA = 2 * EMA(P, n) - EMA(EMA(P, n), n)

  Interpretation:
  - Faster response to price changes than standard EMA
  - Less whipsaw in trending markets
  - Better for crossover strategies

  Features:
  - Fused Numba kernel: computes both EMA stages in single loop
  - Values stay in registers, no intermediate arrays

  Args:
    data: Input price Series (typically close prices)
    period: Lookback period (default: 20)

  Returns:
    DEMAResult with DEMA values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = dema(prices, period=5)
  """
  values = to_numpy(data)

  dema_values = compute_dema_numba(values, period)

  return DEMAResult(data_index=data.index, dema=dema_values)
