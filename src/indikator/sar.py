"""Parabolic SAR indicator module.

Trend-following indicator that provides potential stop and reverse points.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Gt, Hyper, Le, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.sar import compute_sar_numba, compute_sarext_numba
from indikator.utils import to_numpy


@configurable
@validate
def sar(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  acceleration: Hyper[float, Gt[0.0], Le[1.0]] = 0.02,
  maximum: Hyper[float, Gt[0.0], Le[1.0]] = 0.2,
) -> IndicatorResult:
  """Calculate Parabolic SAR (Stop and Reverse).

  Parabolic SAR provides potential entry and exit points by trailing
  price with an accelerating stop level.

  Formula:
  SAR(t+1) = SAR(t) + AF * (EP - SAR(t))

  Where:
  - AF: Acceleration Factor (starts at 0.02, increments by 0.02 on new EP)
  - EP: Extreme Point (highest high in uptrend, lowest low in downtrend)

  Interpretation:
  - Price above SAR: Uptrend (SAR is support)
  - Price below SAR: Downtrend (SAR is resistance)
  - SAR flip: Potential trend reversal

  Features:
  - State machine with register optimization
  - Handles trend reversals automatically

  Args:
    high: High prices
    low: Low prices
    acceleration: AF start and increment (default: 0.02)
    maximum: Maximum AF (default: 0.2)

  Returns:
    IndicatorResult with SAR values

  Example:
    >>> result = sar(high, low, acceleration=0.02, maximum=0.2)
  """
  h = to_numpy(high)
  low_np = to_numpy(low)

  sar_values = compute_sar_numba(h, low_np, acceleration, acceleration, maximum)

  return IndicatorResult(data_index=high.index, value=sar_values, name="sar")


@configurable
@validate
def sarext(  # noqa: PLR0913, PLR0917
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  start_value: Hyper[float] = 0.0,
  offset_on_reversal: Hyper[float] = 0.0,
  acceleration_init_long: Hyper[float] = 0.02,
  acceleration_long: Hyper[float] = 0.02,
  acceleration_max_long: Hyper[float] = 0.2,
  acceleration_init_short: Hyper[float] = 0.02,
  acceleration_short: Hyper[float] = 0.02,
  acceleration_max_short: Hyper[float] = 0.2,
) -> IndicatorResult:
  """Calculate Parabolic SAR Extended (SAREXT).

  More configurable version of the standard Parabolic SAR.

  Args:
    high: High prices.
    low: Low prices.
    start_value: Start value (default 0.0).
    offset_on_reversal: Offset on reversal (default 0.0).
    acceleration_init_long: Initial acceleration for long (default 0.02).
    acceleration_long: Acceleration increment for long (default 0.02).
    acceleration_max_long: Maximum acceleration for long (default 0.2).
    acceleration_init_short: Initial acceleration for short (default 0.02).
    acceleration_short: Acceleration increment for short (default 0.02).
    acceleration_max_short: Maximum acceleration for short (default 0.2).

  Returns:
    IndicatorResult(index, sar)
  """
  h = to_numpy(high)
  low_np = to_numpy(low)

  sar_values = compute_sarext_numba(
    h,
    low_np,
    start_value,
    offset_on_reversal,
    acceleration_init_long,
    acceleration_long,
    acceleration_max_long,
    acceleration_init_short,
    acceleration_short,
    acceleration_max_short,
  )

  return IndicatorResult(data_index=high.index, value=sar_values, name="sar")
