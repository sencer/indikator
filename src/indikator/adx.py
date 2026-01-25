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
from indikator._results import (
  ADXResult,
  ADXRResult,
  ADXSingleResult,
  DXResult,
  MinusDIResult,
  MinusDMResult,
  PlusDIResult,
  PlusDMResult,
)


@configurable
@validate
def adx(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
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

  return ADXSingleResult(data_index=high.index, adx=adx_values)


@configurable
@validate
def adx_with_di(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
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

  return ADXResult(
    data_index=high.index, adx=adx_values, plus_di=plus_di, minus_di=minus_di
  )


@configurable
@validate
def plus_dm(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> PlusDMResult:
  """Calculate Plus Directional Movement (+DM).

  Returns the smoothed accumulated +DM over the period.

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    PlusDMResult
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  from indikator._adx_numba import compute_dms_numba  # noqa: PLC0415

  plus_dm_vals, _, _ = compute_dms_numba(high_arr, low_arr, close_arr, period)
  return PlusDMResult(data_index=high.index, plus_dm=plus_dm_vals)


@configurable
@validate
def minus_dm(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> MinusDMResult:
  """Calculate Minus Directional Movement (-DM).

  Returns the smoothed accumulated -DM over the period.

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    MinusDMResult
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  from indikator._adx_numba import compute_dms_numba  # noqa: PLC0415

  _, minus_dm_vals, _ = compute_dms_numba(high_arr, low_arr, close_arr, period)
  return MinusDMResult(data_index=high.index, minus_dm=minus_dm_vals)


@configurable
@validate
def plus_di(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> PlusDIResult:
  """Calculate Plus Directional Indicator (+DI).

  +DI = 100 * (+DM / TR) (Smoothed)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    PlusDIResult
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  from indikator._adx_numba import compute_di_numba  # noqa: PLC0415

  plus_di_vals, _ = compute_di_numba(high_arr, low_arr, close_arr, period)

  return PlusDIResult(data_index=high.index, plus_di=plus_di_vals)


@configurable
@validate
def minus_di(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> MinusDIResult:
  """Calculate Minus Directional Indicator (-DI).

  -DI = 100 * (-DM / TR) (Smoothed)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    MinusDIResult
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  from indikator._adx_numba import compute_di_numba  # noqa: PLC0415

  _, minus_di_vals = compute_di_numba(high_arr, low_arr, close_arr, period)

  return MinusDIResult(data_index=high.index, minus_di=minus_di_vals)


@configurable
@validate
def dx(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> DXResult:
  """Calculate Directional Movement Index (DX).

  DX = 100 * |+DI - -DI| / (+DI + -DI)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    DXResult
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  from indikator._adx_numba import compute_dx_numba  # noqa: PLC0415

  dx_vals = compute_dx_numba(high_arr, low_arr, close_arr, period)
  return DXResult(data_index=high.index, dx=dx_vals)


@configurable
@validate
def adxr(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> ADXRResult:
  """Calculate Average Directional Movement Rating (ADXR).

  ADXR = (ADX + ADX[i - period]) / 2

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    ADXRResult
  """
  high_arr = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  close_arr = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  from indikator._adx_numba import compute_adxr_numba  # noqa: PLC0415

  adxr_vals = compute_adxr_numba(high_arr, low_arr, close_arr, period)

  return ADXRResult(data_index=high.index, adxr=adxr_vals)
