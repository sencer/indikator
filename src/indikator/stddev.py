"""STDDEV - Standard Deviation over period."""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import STDDEVResult
from indikator._stats_numba import compute_stddev_numba
from indikator.utils import to_numpy


@configurable
@validate
def stddev(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 5,
  nbdev: Hyper[float, Ge[0.0]] = 1.0,
) -> STDDEVResult:
  """Calculate Standard Deviation over period.

  Args:
    data: Input prices
    period: Lookback period (default 5)
    nbdev: Number of deviations (multiplier, default 1.0)

  Returns:
    STDDEVResult
  """
  values = to_numpy(data)
  result = compute_stddev_numba(values, period, nbdev)
  return STDDEVResult(data_index=data.index, stddev=result)
