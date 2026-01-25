"""Money Flow Index (MFI) indicator module.

This module provides MFI calculation, a momentum indicator that uses both
price and volume to measure buying and selling pressure.
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Gt, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._constants import DEFAULT_EPSILON
from indikator._momentum_numba import compute_mfi_numba

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import MFIResult


@configurable
@validate
def mfi(  # noqa: PLR0913, PLR0917
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  volume: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
  epsilon: Hyper[float, Gt[0.0]] = DEFAULT_EPSILON,  # noqa: ARG001
) -> MFIResult:
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
    MFIResult(index, mfi)
  """
  # Convert to numpy for Numba
  high_arr = cast(
    "NDArray[np.float64]",
    high.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  low_arr = cast(
    "NDArray[np.float64]",
    low.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  close_arr = cast(
    "NDArray[np.float64]",
    close.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  vol_arr = cast(
    "NDArray[np.float64]",
    volume.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate MFI using Numba-optimized function
  # Note: compute_mfi_numba calculates Typical Price internally
  mfi_values = compute_mfi_numba(high_arr, low_arr, close_arr, vol_arr, period)

  return MFIResult(index=high.index, mfi=mfi_values)
