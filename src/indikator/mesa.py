"""MESA Indicators module.

Includes MAMA (MESA Adaptive Moving Average) and FAMA (Following Adaptive Moving Average).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import (
  validate,
)
from nonfig import configurable
import numpy as np

if TYPE_CHECKING:
  from datawarden import (
    Finite,
    NotEmpty,
    Validated,
  )
  from nonfig import Gt, Hyper, Le
  from numpy.typing import NDArray
  import pandas as pd

from indikator._mesa_numba import compute_mama_numba
from indikator._results import MAMAResult


@configurable
@validate
def mama(
  data: Validated[pd.Series, Finite, NotEmpty],
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

  return MAMAResult(index=data.index, mama=m, fama=f)
