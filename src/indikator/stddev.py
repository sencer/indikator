"""STDDEV - Standard Deviation over period."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import validate
from nonfig import configurable
import numpy as np

if TYPE_CHECKING:
  from datawarden import Finite, NotEmpty, Validated
  from nonfig import Ge, Hyper
  from numpy.typing import NDArray
  import pandas as pd

from indikator._results import STDDEVResult
from indikator._stats_numba import compute_stddev_numba


@configurable
@validate
def stddev(
  data: Validated[pd.Series, Finite, NotEmpty],
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
  values = cast("NDArray[np.float64]", data.to_numpy(dtype=np.float64, copy=False))
  result = compute_stddev_numba(values, period, nbdev)
  return STDDEVResult(index=data.index, stddev=result)
