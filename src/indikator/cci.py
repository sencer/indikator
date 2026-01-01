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

from indikator._cci_numba import compute_cci_numba


@configurable
@validate
def cci(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> pd.Series:
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
  - CCI crossing 0: Momentum shift
  - Divergence: Price making new high but CCI doesn't = bearish

  Common strategies:
  - Buy when CCI moves from below -100 to above -100
  - Sell when CCI moves from above +100 to below +100
  - Trend following: Trade in direction when CCI > 0 or < 0

  Features:
  - Numba-optimized for performance
  - Standard 20-period default (Lambert's original)
  - Uses mean deviation (more stable than standard deviation)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 20)

  Returns:
    Series with CCI values (unbounded, typically -100 to +100)

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> high = pd.Series([105, 107, 106, 108, 110])
    >>> low = pd.Series([100, 102, 101, 103, 105])
    >>> close = pd.Series([102, 105, 104, 106, 108])
    >>> result = cci(high, low, close)
  """
  # Convert to numpy for Numba
  high_arr = cast(
    "NDArray[np.float64]",
    high.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )
  low_arr = cast(
    "NDArray[np.float64]",
    low.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )
  close_arr = cast(
    "NDArray[np.float64]",
    close.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate CCI using Numba-optimized function
  cci_values = compute_cci_numba(high_arr, low_arr, close_arr, period)

  return pd.Series(cci_values, index=close.index, name="cci")
