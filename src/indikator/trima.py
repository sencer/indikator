from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.trima import compute_trima_numba
from indikator.utils import to_numpy


@configurable
@validate
def trima(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> IndicatorResult:
  """Calculate Triangular Moving Average (TRIMA).

  TRIMA is a smoothed version of SMA, calculated as SMA of SMA.
  The weights form a triangular shape.

  Args:
    data: Input price Series.
    period: Lookback period (default: 30).

  Returns:
    IndicatorResult
  """
  values = to_numpy(data)
  result = compute_trima_numba(values, period)
  return IndicatorResult(data_index=data.index, value=result, name="trima")
