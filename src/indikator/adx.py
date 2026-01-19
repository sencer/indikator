"""ADX (Average Directional Index) indicator module.

This module provides ADX calculation, a trend strength indicator that
measures how strong a trend is, regardless of direction.
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

from indikator._adx_numba import compute_adx_numba, compute_adx_numba_pure
from indikator._results import ADXResult, ADXSingleResult


@configurable
@validate
def adx(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> ADXSingleResult:
  """Calculate Average Directional Index (ADX).

  ADX measures trend strength regardless of direction. This function returns
  only the ADX series for maximum performance (matching TA-Lib).

  For Directional Indicators (+DI, -DI), use `adx_with_di()`.

  Interpretation:
  - ADX < 20: Weak trend / ranging market
  - ADX 25-50: Strong trend
  - ADX > 75: Extremely strong trend

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    ADXSingleResult(index, adx)
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

  # Calculate ADX using pure Numba function (no DI array overhead)
  adx_values = compute_adx_numba_pure(high_arr, low_arr, close_arr, period)

  return ADXSingleResult(index=high.index, adx=adx_values)


@configurable
@validate
def adx_with_di(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> ADXResult:
  """Calculate Average Directional Index (ADX) with DI components.

  Extended calculation that returns +DI and -DI alongside ADX.

  Components:
  - ADX: Average Directional Index (trend strength)
  - +DI: Plus Directional Indicator (bullish pressure)
  - -DI: Minus Directional Indicator (bearish pressure)

  Directional Indicators:
  - +DI > -DI: Bullish
  - -DI > +DI: Bearish
  - +DI crossing above -DI: Buy signal
  - -DI crossing above +DI: Sell signal

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    ADXResult object with adx, plus_di, minus_di series.
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

  # Calculate ADX and DIs
  adx_values, plus_di, minus_di = compute_adx_numba(
    high_arr, low_arr, close_arr, period
  )

  return ADXResult(index=high.index, adx=adx_values, plus_di=plus_di, minus_di=minus_di)


@configurable
@validate
def plus_dm(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> pd.Series:
  """Calculate Plus Directional Movement (+DM).

  Returns the smoothed accumulated +DM over the period.

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with +DM values.
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))

  from indikator._adx_numba import compute_dms_numba

  plus_dm_vals, _, _ = compute_dms_numba(high_arr, low_arr, close_arr, period)
  return pd.Series(plus_dm_vals, index=high.index, name="plus_dm")


@configurable
@validate
def minus_dm(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> pd.Series:
  """Calculate Minus Directional Movement (-DM).

  Returns the smoothed accumulated -DM over the period.

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with -DM values.
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))

  from indikator._adx_numba import compute_dms_numba

  _, minus_dm_vals, _ = compute_dms_numba(high_arr, low_arr, close_arr, period)
  return pd.Series(minus_dm_vals, index=high.index, name="minus_dm")


@configurable
@validate
def plus_di(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> pd.Series:
  """Calculate Plus Directional Indicator (+DI).

  +DI = 100 * (+DM / TR) (Smoothed)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with +DI values.
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))

  from indikator._adx_numba import compute_di_numba

  plus_di_vals, _ = compute_di_numba(high_arr, low_arr, close_arr, period)

  return pd.Series(plus_di_vals, index=high.index, name="plus_di")


@configurable
@validate
def minus_di(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> pd.Series:
  """Calculate Minus Directional Indicator (-DI).

  -DI = 100 * (-DM / TR) (Smoothed)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with -DI values.
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))

  from indikator._adx_numba import compute_di_numba

  _, minus_di_vals = compute_di_numba(high_arr, low_arr, close_arr, period)

  return pd.Series(minus_di_vals, index=high.index, name="minus_di")


@configurable
@validate
def dx(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> pd.Series:
  """Calculate Directional Movement Index (DX).

  DX = 100 * |+DI - -DI| / (+DI + -DI)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with DX values.
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))

  from indikator._adx_numba import compute_dx_numba

  dx_vals = compute_dx_numba(high_arr, low_arr, close_arr, period)
  return pd.Series(dx_vals, index=high.index, name="dx")


@configurable
@validate
def adxr(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> pd.Series:
  """Calculate Average Directional Movement Rating (ADXR).

  ADXR = (ADX + ADX[i - period]) / 2

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with ADXR values.
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))

  from indikator._adx_numba import compute_adxr_numba

  adxr_vals = compute_adxr_numba(high_arr, low_arr, close_arr, period)

  return pd.Series(adxr_vals, index=high.index, name="adxr")
