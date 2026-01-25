from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import T3Result
from indikator._t3_numba import compute_t3_numba
from indikator.utils import to_numpy


@configurable
@validate
def t3(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 5,
  vfactor: Hyper[float, Ge[0.0]] = 0.7,
) -> T3Result:
  """Calculate T3 Moving Average.

  T3 is a smooth, low-lag moving average developed by Tim Tilson.
  It uses a "volume factor" (vfactor) to control responsiveness.
  Default vfactor is 0.7.

  Args:
    data: Input price Series.
    period: EMA period (default 5).
    vfactor: Volume factor (default 0.7).

  Returns:
    T3Result
  """
  values = to_numpy(data)
  result = compute_t3_numba(values, period, vfactor)
  return T3Result(data_index=data.index, t3=result)
