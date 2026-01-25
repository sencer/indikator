from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._results import TRIMAResult
from indikator._trima_numba import compute_trima_numba

if TYPE_CHECKING:
  from numpy.typing import NDArray


@configurable
@validate
def trima(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> TRIMAResult:
  """Calculate Triangular Moving Average (TRIMA).

  TRIMA is a smoothed version of SMA, calculated as SMA of SMA.
  The weights form a triangular shape.

  Args:
    data: Input price Series.
    period: Lookback period (default: 30).

  Returns:
    TRIMAResult
  """
  values = cast("NDArray[np.float64]", data.to_numpy(dtype=np.float64, copy=False))
  result = compute_trima_numba(values, period)
  return TRIMAResult(index=data.index, trima=result)
