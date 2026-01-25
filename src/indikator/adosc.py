"""Accumulation/Distribution Oscillator (ADOSC/Chaikin) indicator module.

Measures momentum of A/D line using fast and slow EMAs.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._ad_numba import compute_adosc_numba
from indikator._results import ADOSCResult
from indikator.utils import to_numpy


@configurable
@validate
def adosc(  # noqa: PLR0913, PLR0917
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  volume: Validated[pd.Series[float], Finite, NotEmpty],
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
  h = to_numpy(high)
  low_arr = to_numpy(low)
  c = to_numpy(close)
  v = to_numpy(volume)

  adosc_values = compute_adosc_numba(h, low_arr, c, v, fast_period, slow_period)

  return ADOSCResult(data_index=close.index, adosc=adosc_values)
