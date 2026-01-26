"""MESA Indicators module.

Includes MAMA (MESA Adaptive Moving Average) and FAMA (Following Adaptive Moving Average).
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Gt, Hyper, Le, configurable
import pandas as pd

from indikator._results import MAMAResult
from indikator.numba.mesa import compute_mama_numba
from indikator.utils import to_numpy


@configurable
@validate
def mama(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  fastlimit: Hyper[float, Gt[0.0], Le[1.0]] = 0.5,
  slowlimit: Hyper[float, Gt[0.0], Le[1.0]] = 0.05,
) -> MAMAResult:
  """Calculate MESA Adaptive Moving Average (MAMA) and FAMA.

  MAMA and FAMA provide a smoothed, responsive trendline that avoids
  whipsaws in consolidation but adapts quickly to new trends.

  Args:
    data: Input price Series.
    fastlimit: MESA fast limit (default: 0.5).
    slowlimit: MESA slow limit (default: 0.05).

  Returns:
    MAMAResult(index, mama, fama)
  """
  values = to_numpy(data)

  m, f = compute_mama_numba(values, fastlimit, slowlimit)

  return MAMAResult(data_index=data.index, mama=m, fama=f)
