"""MIDPRICE - Midpoint Price over period.

MIDPRICE = (highest high + lowest low) / 2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import MIDPRICEResult


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
  n = len(h)

  result = np.empty(n, dtype=np.float64)
  result[: period - 1] = np.nan

  # Use pandas rolling for simplicity and decent performance
  h_series = pd.Series(h)
  l_series = pd.Series(l)

  highest = h_series.rolling(window=period).max().to_numpy()
  lowest = l_series.rolling(window=period).min().to_numpy()

  result = (highest + lowest) / 2.0

  return MIDPRICEResult(index=high.index, midprice=result)
