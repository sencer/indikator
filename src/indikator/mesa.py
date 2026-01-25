"""MESA Indicators module.

Includes MAMA (MESA Adaptive Moving Average) and FAMA (Following Adaptive Moving Average).
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Gt, Hyper, Le, configurable
import numpy as np
import pandas as pd

from indikator._mesa_numba import compute_mama_numba
from indikator._results import MAMAResult

if TYPE_CHECKING:
  from numpy.typing import NDArray


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
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore
  )

  m, f = compute_mama_numba(values, fastlimit, slowlimit)

  return MAMAResult(data_index=data.index, mama=m, fama=f)
