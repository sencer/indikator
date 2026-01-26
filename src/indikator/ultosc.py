"""Ultimate Oscillator (ULTOSC) indicator module.

Multi-timeframe momentum oscillator using weighted average of BP/TR ratios.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.ultosc import compute_ultosc_numba
from indikator.utils import to_numpy


@configurable
@validate
def ultosc(  # noqa: PLR0913, PLR0917
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period1: Hyper[int, Ge[1]] = 7,
  period2: Hyper[int, Ge[1]] = 14,
  period3: Hyper[int, Ge[1]] = 28,
) -> IndicatorResult:
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
    IndicatorResult with oscillator values (0-100)

  Example:
    >>> result = ultosc(high, low, close, period1=7, period2=14, period3=28)
  """
  h = to_numpy(high)
  low_np = to_numpy(low)
  c = to_numpy(close)

  ultosc_values = compute_ultosc_numba(h, low_np, c, period1, period2, period3)

  return IndicatorResult(data_index=close.index, value=ultosc_values, name="ultosc")
