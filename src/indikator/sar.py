"""Parabolic SAR indicator module.

Trend-following indicator that provides potential stop and reverse points.
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Gt, Hyper, Le, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import SARResult
from indikator._sar_numba import compute_sar_numba


@configurable
@validate
def sar(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  acceleration: Hyper[float, Gt[0.0], Le[1.0]] = 0.02,
  maximum: Hyper[float, Gt[0.0], Le[1.0]] = 0.2,
) -> SARResult:
  """Calculate Parabolic SAR (Stop and Reverse).

  Parabolic SAR provides potential entry and exit points by trailing
  price with an accelerating stop level.

  Formula:
  SAR(t+1) = SAR(t) + AF * (EP - SAR(t))

  Where:
  - AF: Acceleration Factor (starts at 0.02, increments by 0.02 on new EP)
  - EP: Extreme Point (highest high in uptrend, lowest low in downtrend)

  Interpretation:
  - Price above SAR: Uptrend (SAR is support)
  - Price below SAR: Downtrend (SAR is resistance)
  - SAR flip: Potential trend reversal

  Features:
  - State machine with register optimization
  - Handles trend reversals automatically

  Args:
    high: High prices
    low: Low prices
    acceleration: AF start and increment (default: 0.02)
    maximum: Maximum AF (default: 0.2)

  Returns:
    SARResult with SAR values

  Example:
    >>> result = sar(high, low, acceleration=0.02, maximum=0.2)
  """
  h = cast(
    "NDArray[np.float64]",
    high.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  l = cast(
    "NDArray[np.float64]",
    low.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  sar_values = compute_sar_numba(h, l, acceleration, acceleration, maximum)

  return SARResult(index=high.index, sar=sar_values)
