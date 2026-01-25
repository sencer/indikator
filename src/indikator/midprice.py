"""MIDPRICE - Midpoint Price over period.

MIDPRICE = (highest high + lowest low) / 2
"""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import MIDPRICEResult
from indikator._rolling_numba import compute_midprice_numba
from indikator.utils import to_numpy


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
  h = to_numpy(high)
  low_np = to_numpy(low)

  result = compute_midprice_numba(h, low_np, period)

  return MIDPRICEResult(data_index=high.index, midprice=result)
