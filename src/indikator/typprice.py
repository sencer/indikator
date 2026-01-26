"""TYPPRICE - Typical Price indicator.

TYPPRICE = (High + Low + Close) / 3
"""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.price_transform import compute_typprice_numba
from indikator.utils import to_numpy


@configurable
@validate
def typprice(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Calculate Typical Price.

  TYPPRICE = (High + Low + Close) / 3

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

  result = compute_typprice_numba(h, low_np, c)

  return IndicatorResult(data_index=high.index, value=result, name="typprice")
