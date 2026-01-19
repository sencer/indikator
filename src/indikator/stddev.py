"""STDDEV - Standard Deviation over period."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import STDDEVResult


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

  # Use pandas rolling with ddof=0 to match TA-Lib (population std)
  s = pd.Series(values)
  result = s.rolling(window=period).std(ddof=0).to_numpy() * nbdev

  return STDDEVResult(index=data.index, stddev=result)
