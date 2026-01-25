"""MIDPOINT - Midpoint over period.

MIDPOINT = (highest value + lowest value) / 2
"""

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import MIDPOINTResult
from indikator._rolling_numba import compute_midpoint_numba


@configurable
@validate
def midpoint(
  data: Validated[pd.Series[float], Finite, NotEmpty],
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
  values = cast("NDArray[np.float64]", data.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  result = compute_midpoint_numba(values, period)

  return MIDPOINTResult(data_index=data.index, midpoint=result)
