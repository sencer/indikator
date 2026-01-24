"""MIDPRICE - Midpoint Price over period.

MIDPRICE = (highest high + lowest low) / 2
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

from indikator._results import MIDPRICEResult
from indikator._rolling_numba import compute_midprice_numba


@configurable
@validate
def midprice(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> MIDPRICEResult:
  """Calculate Midpoint Price over period.

  MIDPRICE = (highest high + lowest low) / 2

  Args:
    high: High prices
    low: Low prices
    period: Lookback period (default 14)

  Returns:
    MIDPRICEResult
  """
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))

  result = compute_midprice_numba(h, l, period)

  return MIDPRICEResult(index=high.index, midprice=result)
