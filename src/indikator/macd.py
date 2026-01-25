"""MACD (Moving Average Convergence Divergence) indicator module."""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
  from collections.abc import Callable

  from numpy.typing import NDArray

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._macd_numba import compute_macd_numba
from indikator._results import MACDResult


def _get_ma_func(
  matype: int,
) -> "Callable[[pd.Series, int], NDArray[np.float64]]":
  from indikator.dema import dema  # noqa: PLC0415
  from indikator.ema import ema  # noqa: PLC0415
  from indikator.kama import kama  # noqa: PLC0415
  from indikator.mesa import mama  # noqa: PLC0415
  from indikator.sma import sma  # noqa: PLC0415
  from indikator.t3 import t3  # noqa: PLC0415
  from indikator.tema import tema  # noqa: PLC0415
  from indikator.trima import trima  # noqa: PLC0415
  from indikator.wma import wma  # noqa: PLC0415

  mapping = {
    0: lambda d, p: sma(d, p).sma,
    1: lambda d, p: ema(d, p).ema,
    2: lambda d, p: wma(d, p).wma,
    3: lambda d, p: dema(d, p).dema,
    4: lambda d, p: tema(d, p).tema,
    5: lambda d, p: trima(d, p).trima,
    6: lambda d, p: kama(d, p).kama,
    7: lambda d, _: mama(d).mama,  # Period ignored for MAMA
    8: lambda d, p: t3(d, p).t3,
  }
  return mapping.get(matype, mapping[1])  # Default EMA


@configurable
@validate
def macd(
  data: Validated[pd.Series, Finite, NotEmpty],
  fast_period: Hyper[int, Ge[2]] = 12,
  slow_period: Hyper[int, Ge[2]] = 26,
  signal_period: Hyper[int, Ge[2]] = 9,
) -> MACDResult:
  """Calculate Moving Average Convergence Divergence (MACD).

  MACD is a trend-following momentum indicator that shows the relationship
  between two moving averages of a security's price.

  Formula:
  MACD Line = EMA(fast_period) - EMA(slow_period)
  Signal Line = EMA(MACD Line, signal_period)
  Histogram = MACD Line - Signal Line

  Interpretation:
  - MACD crossing above Signal: Bullish
  - MACD crossing below Signal: Bearish
  - MACD > 0: Fast MA > Slow MA (Uptrend)
  - MACD < 0: Fast MA < Slow MA (Downtrend)
  - Histogram widening: Trend strengthening
  - Histogram narrowing: Trend weakening

  Args:
    data: Input Series.
    fast_period: Fast EMA period (default: 12)
    slow_period: Slow EMA period (default: 26)
    signal_period: Signal line EMA period (default: 9)

  Returns:
    DataFrame with 'macd', 'macd_signal', 'macd_histogram' columns

  Raises:
    ValueError: If fast_period >= slow_period

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> result = macd(prices)
    >>> # Returns DataFrame with MACD components
  """
  # Validate parameters
  if fast_period >= slow_period:
    raise ValueError(
      f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
    )

  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate MACD using Numba-optimized function
  macd_line, signal_line, histogram = compute_macd_numba(
    values, fast_period, slow_period, signal_period
  )

  return MACDResult(
    index=data.index, macd=macd_line, signal=signal_line, histogram=histogram
  )


@configurable
@validate
def macdext(  # noqa: PLR0913, PLR0917
  data: Validated[pd.Series, Finite, NotEmpty],
  fast_period: Hyper[int, Ge[2]] = 12,
  fast_matype: Hyper[int, Ge[0]] = 0,
  slow_period: Hyper[int, Ge[2]] = 26,
  slow_matype: Hyper[int, Ge[0]] = 0,
  signal_period: Hyper[int, Ge[2]] = 9,
  signal_matype: Hyper[int, Ge[0]] = 0,
) -> MACDResult:
  """Calculate MACD with full control over MA types.

  Args:
    data: Input Series.
    fast_period: Fast period (default: 12)
    fast_matype: Fast MA type (default: 0 - SMA)
    slow_period: Slow period (default: 26)
    slow_matype: Slow MA type (default: 0 - SMA)
    signal_period: Signal period (default: 9)
    signal_matype: Signal MA type (default: 0 - SMA)

  Returns:
    MACDResult(index, macd, signal, histogram)
  """
  fast_func = _get_ma_func(fast_matype)
  slow_func = _get_ma_func(slow_matype)
  signal_func = _get_ma_func(signal_matype)

  f_ma = fast_func(data, fast_period)
  s_ma = slow_func(data, slow_period)

  macd_line = f_ma - s_ma

  # Wrap macd_line in Series for signal calculation
  macd_series = pd.Series(macd_line, index=data.index)
  signal_line = signal_func(macd_series, signal_period)

  histogram = macd_line - signal_line

  return MACDResult(
    index=data.index, macd=macd_line, signal=signal_line, histogram=histogram
  )


@configurable
@validate
def macdfix(
  data: Validated[pd.Series, Finite, NotEmpty],
  signal_period: Hyper[int, Ge[2]] = 9,
) -> MACDResult:
  """Calculate MACD with fixed periods (12, 26).

  Matches TA-Lib MACDFIX.
  """
  return macd(data, fast_period=12, slow_period=26, signal_period=signal_period)
