"""MIDPOINT - Midpoint over period.

MIDPOINT = (highest value + lowest value) / 2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import MIDPOINTResult


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

  # Use pandas rolling for simplicity
  s = pd.Series(values)

  highest = s.rolling(window=period).max().to_numpy()
  lowest = s.rolling(window=period).min().to_numpy()

  result = (highest + lowest) / 2.0

  return MIDPOINTResult(index=data.index, midpoint=result)
