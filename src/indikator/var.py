"""VAR - Variance over period."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import VARResult
from indikator._stats_numba import compute_var_numba


@configurable
@validate
def var(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 5,
  nbdev: Hyper[float, Ge[0.0]] = 1.0,
) -> VARResult:
  """Calculate Variance over period.

  Args:
    data: Input prices
    period: Lookback period (default 5)
    nbdev: Number of deviations (multiplier, default 1.0)

  Returns:
    VARResult
  """
  values = cast("NDArray[np.float64]", data.to_numpy(dtype=np.float64, copy=False))
  result = compute_var_numba(values, period, nbdev)
  return VARResult(index=data.index, var=result)
