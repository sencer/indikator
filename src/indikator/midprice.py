"""MIDPRICE - Midpoint Price over period.

MIDPRICE = (highest high + lowest low) / 2
"""

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import MIDPRICEResult
from indikator._rolling_numba import compute_midprice_numba


@configurable
@validate
def midprice(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
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
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_np = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  result = compute_midprice_numba(h, low_np, period)

  return MIDPRICEResult(data_index=high.index, midprice=result)
