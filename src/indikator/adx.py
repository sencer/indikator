"""ADX (Average Directional Index) indicator module.

This module provides ADX calculation, a trend strength indicator that
measures how strong a trend is, regardless of direction.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import (
  ADXResult,
  IndicatorResult,
)
from indikator.numba.adx import compute_adx_numba, compute_adx_numba_pure
from indikator.utils import to_numpy


@configurable
@validate
def adx(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
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
    IndicatorResult(index, adx)
  """
  # Convert to numpy for Numba
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  # Calculate ADX using pure Numba function (no DI array overhead)
  adx_values = compute_adx_numba_pure(high_arr, low_arr, close_arr, period)

  return IndicatorResult(data_index=high.index, value=adx_values, name="adx")


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
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

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
) -> IndicatorResult:
  """Calculate Plus Directional Movement (+DM).

  Returns the smoothed accumulated +DM over the period.

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    IndicatorResult
  """
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  from indikator.numba.adx import compute_dms_numba  # noqa: PLC0415

  plus_dm_vals, _, _ = compute_dms_numba(high_arr, low_arr, close_arr, period)
  return IndicatorResult(data_index=high.index, value=plus_dm_vals, name="plus_dm")


@configurable
@validate
def minus_dm(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> IndicatorResult:
  """Calculate Minus Directional Movement (-DM).

  Returns the smoothed accumulated -DM over the period.

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    IndicatorResult
  """
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  from indikator.numba.adx import compute_dms_numba  # noqa: PLC0415

  _, minus_dm_vals, _ = compute_dms_numba(high_arr, low_arr, close_arr, period)
  return IndicatorResult(data_index=high.index, value=minus_dm_vals, name="minus_dm")


@configurable
@validate
def plus_di(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> IndicatorResult:
  """Calculate Plus Directional Indicator (+DI).

  +DI = 100 * (+DM / TR) (Smoothed)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    IndicatorResult
  """
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  from indikator.numba.adx import compute_di_numba  # noqa: PLC0415

  plus_di_vals, _ = compute_di_numba(high_arr, low_arr, close_arr, period)

  return IndicatorResult(data_index=high.index, value=plus_di_vals, name="plus_di")


@configurable
@validate
def minus_di(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> IndicatorResult:
  """Calculate Minus Directional Indicator (-DI).

  -DI = 100 * (-DM / TR) (Smoothed)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    IndicatorResult
  """
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  from indikator.numba.adx import compute_di_numba  # noqa: PLC0415

  _, minus_di_vals = compute_di_numba(high_arr, low_arr, close_arr, period)

  return IndicatorResult(data_index=high.index, value=minus_di_vals, name="minus_di")


@configurable
@validate
def dx(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> IndicatorResult:
  """Calculate Directional Movement Index (DX).

  DX = 100 * |+DI - -DI| / (+DI + -DI)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    IndicatorResult
  """
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  from indikator.numba.adx import compute_dx_numba  # noqa: PLC0415

  dx_vals = compute_dx_numba(high_arr, low_arr, close_arr, period)
  return IndicatorResult(data_index=high.index, value=dx_vals, name="dx")


@configurable
@validate
def adxr(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
  """Calculate Average Directional Movement Rating (ADXR).

  ADXR = (ADX + ADX[i - period]) / 2

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    IndicatorResult
  """
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  from indikator.numba.adx import compute_adxr_numba  # noqa: PLC0415

  adxr_vals = compute_adxr_numba(high_arr, low_arr, close_arr, period)

  return IndicatorResult(data_index=high.index, value=adxr_vals, name="adxr")
