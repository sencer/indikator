"""WCLPRICE - Weighted Close Price indicator.

WCLPRICE = (High + Low + 2*Close) / 4
"""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.price_transform import compute_wclprice_numba
from indikator.utils import to_numpy


@configurable
@validate
def wclprice(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Calculate Weighted Close Price.

  WCLPRICE = (High + Low + 2*Close) / 4

  Args:
    high: High prices
    low: Low prices
    close: Close prices

  Returns:
    IndicatorResult
  """
  h = to_numpy(high)
  low_np = to_numpy(low)
  c = to_numpy(close)

  result = compute_wclprice_numba(h, low_np, c)

  return IndicatorResult(data_index=high.index, value=result, name="wclprice")
