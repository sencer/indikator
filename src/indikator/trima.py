from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import TRIMAResult
from indikator._trima_numba import compute_trima_numba
from indikator.utils import to_numpy


@configurable
@validate
def trima(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> TRIMAResult:
  """Calculate Triangular Moving Average (TRIMA).

  TRIMA is a smoothed version of SMA, calculated as SMA of SMA.
  The weights form a triangular shape.

  Args:
    data: Input price Series.
    period: Lookback period (default: 30).

  Returns:
    TRIMAResult
  """
  values = to_numpy(data)
  result = compute_trima_numba(values, period)
  return TRIMAResult(data_index=data.index, trima=result)
