from __future__ import annotations

from typing import TYPE_CHECKING

from datawarden import (
  validate,
)
from nonfig import configurable
import numpy as np

from indikator._results import T3Result
from indikator._t3_numba import compute_t3_numba

if TYPE_CHECKING:
  from datawarden import (
    Finite,
    NotEmpty,
    Validated,
  )
  from nonfig import Ge, Hyper
  import pandas as pd


@configurable
@validate
def t3(
  data: Validated[pd.Series, Finite, NotEmpty],
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
  values = data.to_numpy(dtype=np.float64, copy=False)
  result = compute_t3_numba(values, period, vfactor)
  return T3Result(index=data.index, t3=result)
