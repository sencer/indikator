"""MIDPRICE - Midpoint Price over period.

MIDPRICE = (highest high + lowest low) / 2
"""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.rolling import compute_midprice_numba
from indikator.utils import to_numpy


@configurable
@validate
def midprice(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
  """Calculate Midpoint Price over period.

  MIDPRICE = (highest high + lowest low) / 2

  Args:
    high: High prices
    low: Low prices
    period: Lookback period (default 14)

  Returns:
    IndicatorResult
  """
  h = to_numpy(high)
  low_np = to_numpy(low)

  result = compute_midprice_numba(h, low_np, period)

  return IndicatorResult(data_index=high.index, value=result, name="midprice")
