"""MEDPRICE - Median Price indicator.

MEDPRICE = (High + Low) / 2
"""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.price_transform import compute_medprice_numba
from indikator.utils import to_numpy


@configurable
@validate
def medprice(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Calculate Median Price.

  MEDPRICE = (High + Low) / 2

  Args:
    high: High prices
    low: Low prices

  Returns:
    IndicatorResult
  """
  h = to_numpy(high)
  low_np = to_numpy(low)

  result = compute_medprice_numba(h, low_np)

  return IndicatorResult(data_index=high.index, value=result, name="medprice")
