"""Normalized Average True Range (NATR) indicator module.

NATR expresses ATR as a percentage of the closing price for
cross-instrument comparison.
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
from indikator.numba.natr import compute_natr_numba
from indikator.utils import to_numpy


@configurable
@validate
def natr(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> IndicatorResult:
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
    IndicatorResult with normalized ATR values (percentage)

  Example:
    >>> result = natr(high, low, close, period=14)
  """
  h = to_numpy(high)
  low_np = to_numpy(low)
  c = to_numpy(close)

  natr_values = compute_natr_numba(h, low_np, c, period)

  return IndicatorResult(data_index=close.index, value=natr_values, name="natr")
