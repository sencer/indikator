"""AVGPRICE - Average Price indicator.

AVGPRICE = (Open + High + Low + Close) / 4
"""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.price_transform import compute_avgprice_numba
from indikator.utils import to_numpy


@configurable
@validate
def avgprice(
  open: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Calculate Average Price.

  AVGPRICE = (Open + High + Low + Close) / 4

  Args:
    open: Open prices
    high: High prices
    low: Low prices
    close: Close prices

  Returns:
    IndicatorResult
  """
  o = to_numpy(open)
  h = to_numpy(high)
  low_arr = to_numpy(low)
  c = to_numpy(close)

  result = compute_avgprice_numba(o, h, low_arr, c)

  return IndicatorResult(data_index=high.index, value=result, name="avgprice")
