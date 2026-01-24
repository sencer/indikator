"""CCI (Commodity Channel Index) indicator module.

This module provides CCI calculation, a momentum-based oscillator
used to help determine when an asset is overbought or oversold.
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

from indikator._momentum_numba import compute_cci_numba
from indikator._results import CCIResult


@configurable
@validate
def cci(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
  constant: float = 0.015,
) -> CCIResult:
  """Calculate Commodity Channel Index (CCI).

  CCI is a momentum-based oscillator used to determine when an asset
  is overbought or oversold. It measures price variation from the mean.

  Formula:
  CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)

  Where:
  - Typical Price = (High + Low + Close) / 3
  - Mean Deviation = mean of absolute deviations from SMA
  - 0.015 constant scales to make 70-80% of values within +/- 100

  Interpretation:
  - CCI > +100: Overbought (potential reversal down)
  - CCI < -100: Oversold (potential reversal up)
  - CCI > 100: Overbought / strong uptrend
  - CCI < -100: Oversold / strong downtrend
  - CCI ~ 0: Ranging
  - Divergences often precede reversals

  Features:
  - Numba-optimized for performance
  - Standard constant 0.015 (Lambert's default)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)
    constant: Scaling constant (default: 0.015)

  Returns:
    CCIResult(index, cci)
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

  # Calculate CCI using Numba-optimized function
  cci_values = compute_cci_numba(high_arr, low_arr, close_arr, period, constant)

  return CCIResult(index=high.index, cci=cci_values)
