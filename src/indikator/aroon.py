"""AROON indicator module.

This module provides AROON calculation, a trend indicator that measures
how many periods since the highest high and lowest low.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._aroon_numba import compute_aroon_numba, compute_aroonosc_numba
from indikator._results import AROONOSCResult, AROONResult
from indikator.utils import to_numpy


@configurable
@validate
def aroon(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 25,
) -> AROONResult:
  """Calculate AROON indicator.

    AROON measures trend strength by tracking how many periods since
    the highest high and lowest low occurred. It helps identify trend
    changes and strength.

    Formulas:
    Aroon Up = 100 * (period - periods_since_high) / period
    Aroon Down = 100 * (period - periods_since_low) / period
    Aroon Oscillator = Aroon Up - Aroon Down

    Interpretation:
    - Aroon Up > 70 and Aroon Down < 30: Strong uptrend
    - Aroon Down > 70 and Aroon Up < 30: Strong downtrend
    - Both low: Consolidation
    - Crossovers: Trend change signals
    - Aroon Osc > 0: Bullish, < 0: Bearish

    Features:
    - Numba-optimized for performance
    - Returns Up, Down, and Oscillator values
    - Range: 0 to 100 for Up/Down, -100 to +100 for Oscillator

    Args:
      high: High prices Series
      low: Low prices Series
      period: Lookback period (default: 25)

    Returns:
      DataFrame with columns: aroon_up, aroon_down, aroon_osc

    Raises:
      ValueError: If data contains NaN/Inf

    Example:
      >>> import pandas as pd
  from indikator.utils import to_numpy
      >>> high = pd.Series([105, 106, 104, 108, 107, 109, 108, 110] * 5)
      >>> low = pd.Series([100, 101, 99, 103, 102, 104, 103, 105] * 5)
      >>> result = aroon(high, low, period=5)
      >>> # Returns DataFrame with aroon_up, aroon_down, aroon_osc
  """
  # Convert to numpy for Numba
  high_vals = to_numpy(high)
  low_vals = to_numpy(low)

  # Calculate AROON using Numba-optimized function
  up, down, osc = compute_aroon_numba(high_vals, low_vals, period)

  return AROONResult(
    data_index=high.index,
    aroon_up=up,
    aroon_down=down,
    aroon_osc=osc,
  )


@configurable
@validate
def aroonosc(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 25,
) -> AROONOSCResult:
  """Calculate Aroon Oscillator.

  AROONOSC = Aroon Up - Aroon Down

  Range: -100 to +100
  - Positive: Bullish (uptrend)
  - Negative: Bearish (downtrend)

  Args:
    high: High prices Series
    low: Low prices Series
    period: Lookback period (default: 25)

  Returns:
    AROONOSCResult
  """
  high_vals = to_numpy(high)
  low_vals = to_numpy(low)

  osc = compute_aroonosc_numba(high_vals, low_vals, period)

  return AROONOSCResult(data_index=high.index, aroonosc=osc)
