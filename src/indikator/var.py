"""VAR - Variance over period."""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.stats import compute_var_numba
from indikator.utils import to_numpy


@configurable
@validate
def var(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 5,
  nbdev: Hyper[float, Ge[0.0]] = 1.0,
) -> IndicatorResult:
  """Calculate Variance over period.

  Args:
    data: Input prices
    period: Lookback period (default 5)
    nbdev: Number of deviations (multiplier, default 1.0)

  Returns:
    IndicatorResult
  """
  values = to_numpy(data)
  result = compute_var_numba(values, period, nbdev)
  return IndicatorResult(data_index=data.index, value=result, name="var")
