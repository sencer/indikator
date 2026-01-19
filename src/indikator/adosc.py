"""Accumulation/Distribution Oscillator (ADOSC/Chaikin) indicator module.

Measures momentum of A/D line using fast and slow EMAs.
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

from indikator._ad_numba import compute_adosc_numba
from indikator._results import ADOSCResult


@configurable
@validate
def adosc(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  volume: Validated[pd.Series, Finite, NotEmpty],
  fast_period: Hyper[int, Ge[2]] = 3,
  slow_period: Hyper[int, Ge[2]] = 10,
) -> ADOSCResult:
  """Calculate Accumulation/Distribution Oscillator (Chaikin Oscillator).

  ADOSC measures the momentum of the A/D line by taking the difference
  between fast and slow EMAs of the cumulative A/D values.

  Formula:
  ADOSC = EMA(AD, fast) - EMA(AD, slow)

  Interpretation:
  - Positive ADOSC: Bullish momentum (A/D accelerating up)
  - Negative ADOSC: Bearish momentum (A/D accelerating down)
  - Zero crossing: Momentum shift

  Features:
  - Fused computation: AD + dual EMA in single pass
  - O(N) complexity

  Args:
    high: High prices
    low: Low prices
    close: Close prices
    volume: Volume
    fast_period: Fast EMA period (default: 3)
    slow_period: Slow EMA period (default: 10)

  Returns:
    ADOSCResult with oscillator values

  Example:
    >>> result = adosc(high, low, close, volume, fast_period=3, slow_period=10)
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
  v = cast(
    "NDArray[np.float64]",
    volume.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  adosc_values = compute_adosc_numba(h, l, c, v, fast_period, slow_period)

  return ADOSCResult(index=close.index, adosc=adosc_values)
