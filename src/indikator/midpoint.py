"""MIDPOINT - Midpoint over period.

MIDPOINT = (highest value + lowest value) / 2
"""

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

from indikator._results import MIDPOINTResult
from indikator._rolling_numba import compute_midpoint_numba


@configurable
@validate
def midpoint(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> MIDPOINTResult:
  """Calculate Midpoint over period.

  MIDPOINT = (highest value + lowest value) / 2

  Args:
    data: Input prices
    period: Lookback period (default 14)

  Returns:
    MIDPOINTResult
  """
  values = cast("NDArray[np.float64]", data.to_numpy(dtype=np.float64, copy=False))

  result = compute_midpoint_numba(values, period)

  return MIDPOINTResult(index=data.index, midpoint=result)
