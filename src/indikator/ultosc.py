"""Ultimate Oscillator (ULTOSC) indicator module.

Multi-timeframe momentum oscillator using weighted average of BP/TR ratios.
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import ULTOSCResult
from indikator._ultosc_numba import compute_ultosc_numba


@configurable
@validate
def ultosc(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period1: Hyper[int, Ge[1]] = 7,
  period2: Hyper[int, Ge[1]] = 14,
  period3: Hyper[int, Ge[1]] = 28,
) -> ULTOSCResult:
  """Calculate Ultimate Oscillator (ULTOSC).

  ULTOSC combines momentum from three timeframes to reduce false
  signals common in single-period oscillators.

  Formula:
  BP = Close - min(Low, Prior Close)
  TR = max(High, Prior Close) - min(Low, Prior Close)
  ULTOSC = 100 * (4*Avg1 + 2*Avg2 + 1*Avg3) / 7

  Interpretation:
  - ULTOSC > 70: Overbought
  - ULTOSC < 30: Oversold
  - Divergence with price: Potential reversal

  Features:
  - O(1) rolling sums using circular buffers
  - Three weighted timeframes reduce noise

  Args:
    high: High prices
    low: Low prices
    close: Close prices
    period1: Short period (default: 7, weight 4)
    period2: Medium period (default: 14, weight 2)
    period3: Long period (default: 28, weight 1)

  Returns:
    ULTOSCResult with oscillator values (0-100)

  Example:
    >>> result = ultosc(high, low, close, period1=7, period2=14, period3=28)
  """
  h = cast(
    "NDArray[np.float64]",
    high.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  l = cast(
    "NDArray[np.float64]",
    low.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  c = cast(
    "NDArray[np.float64]",
    close.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  ultosc_values = compute_ultosc_numba(h, l, c, period1, period2, period3)

  return ULTOSCResult(index=close.index, ultosc=ultosc_values)
