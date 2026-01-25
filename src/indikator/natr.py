"""Normalized Average True Range (NATR) indicator module.

NATR expresses ATR as a percentage of the closing price for
cross-instrument comparison.
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

from indikator._natr_numba import compute_natr_numba
from indikator._results import NATRResult


@configurable
@validate
def natr(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> NATRResult:
  """Calculate Normalized Average True Range (NATR).

  NATR normalizes ATR as a percentage of the closing price, allowing
  volatility comparison across different price levels and instruments.

  Formula:
  NATR = (ATR / Close) * 100

  Interpretation:
  - Higher NATR: More volatile relative to price
  - Lower NATR: Less volatile relative to price
  - Useful for position sizing across different instruments
  - Allows volatility comparison between $10 and $1000 stocks

  Features:
  - Fused Numba kernel: computes TR, ATR, and normalization in single loop
  - No intermediate arrays

  Args:
    high: High prices
    low: Low prices
    close: Close prices
    period: ATR lookback period (default: 14)

  Returns:
    NATRResult with normalized ATR values (percentage)

  Example:
    >>> result = natr(high, low, close, period=14)
  """
  h = cast(
    "NDArray[np.float64]",
    high.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  low_np = cast(
    "NDArray[np.float64]",
    low.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  c = cast(
    "NDArray[np.float64]",
    close.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  natr_values = compute_natr_numba(h, low_np, c, period)

  return NATRResult(data_index=close.index, natr=natr_values)
