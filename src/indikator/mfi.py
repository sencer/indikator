"""Money Flow Index (MFI) indicator module.

This module provides MFI calculation, a momentum indicator that uses both
price and volume to measure buying and selling pressure.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Gt, Hyper, configurable
import pandas as pd

from indikator._constants import DEFAULT_EPSILON
from indikator._results import IndicatorResult
from indikator.numba.momentum import compute_mfi_numba
from indikator.utils import to_numpy


@configurable
@validate
def mfi(  # noqa: PLR0913, PLR0917
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  volume: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
  epsilon: Hyper[float, Gt[0.0]] = DEFAULT_EPSILON,  # noqa: ARG001
) -> IndicatorResult:
  """Calculate Money Flow Index (MFI).

  MFI is a momentum indicator that uses both price and volume to measure
  buying and selling pressure. It is also known as volume-weighted RSI.

  Formula:
  1. Typical Price = (High + Low + Close) / 3
  2. Money Flow = Typical Price * Volume
  3. Positive Money Flow = sum of MF when typical price increases
  4. Negative Money Flow = sum of MF when typical price decreases
  5. Money Flow Ratio = Positive Money Flow / Negative Money Flow
  Typical Price = (High + Low + Close) / 3
  Raw Money Flow = Typical Price * Volume
  Money Ratio = Positive Money Flow / Negative Money Flow
  MFI = 100 - (100 / (1 + Money Ratio))

  Interpretation:
  - MFI > 80: Overbought
  - MFI < 20: Oversold
  - Divergence: Price makes new high but MFI doesn't = reversal signal

  Features:
  - Numba-optimized for performance
  - Handles edge cases (division by zero)
  - Uses standard 14 period default

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    volume: Volume Series.
    period: Lookback period (default: 14)

  Returns:
    IndicatorResult(index, mfi)
  """
  # Convert to numpy for Numba
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)
  vol_arr = to_numpy(volume)

  # Calculate MFI using Numba-optimized function
  # Note: compute_mfi_numba calculates Typical Price internally
  mfi_values = compute_mfi_numba(high_arr, low_arr, close_arr, vol_arr, period)

  return IndicatorResult(data_index=high.index, value=mfi_values, name="mfi")
