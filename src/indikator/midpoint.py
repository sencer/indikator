"""MIDPOINT - Midpoint over period.

MIDPOINT = (highest value + lowest value) / 2
"""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.rolling import compute_midpoint_numba
from indikator.utils import to_numpy


@configurable
@validate
def midpoint(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
  """Calculate Midpoint over period.

  MIDPOINT = (highest value + lowest value) / 2

  Args:
    data: Input prices
    period: Lookback period (default 14)

  Returns:
    IndicatorResult
  """
  values = to_numpy(data)

  result = compute_midpoint_numba(values, period)

  return IndicatorResult(data_index=data.index, value=result, name="midpoint")
