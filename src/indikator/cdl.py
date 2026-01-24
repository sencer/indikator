"""Candlestick Pattern Recognition module.

This module provides detection for common candlestick patterns.
Returns integer series:
- 100: Bullish pattern detected
- -100: Bearish pattern detected
- 0: No pattern
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._cdl_numba import (
  detect_doji_numba,
  detect_engulfing_numba,
  detect_evening_star_numba,
  detect_hammer_numba,
  detect_hanging_man_numba,
  detect_harami_numba,
  detect_inverted_hammer_numba,
  detect_marubozu_numba,
  detect_morning_star_numba,
  detect_shooting_star_numba,
  detect_three_black_crows_numba,
  detect_three_inside_numba,
  detect_three_line_strike_numba,
  detect_three_outside_numba,
  detect_three_white_soldiers_numba,
  detect_dark_cloud_cover_numba,
  detect_kicking_numba,
  detect_matching_low_numba,
  detect_piercing_numba,
)


@configurable
@validate
def cdl_doji(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Doji pattern.

  Returns 100 if detected, 0 otherwise.
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_doji_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_doji")


@configurable
@validate
def cdl_hammer(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Hammer pattern.

  Returns 100 (Bullish) if detected, 0 otherwise.
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_hammer_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_hammer")


@configurable
@validate
def cdl_engulfing(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Engulfing pattern.

  Returns:
  - 100: Bullish Engulfing
  - -100: Bearish Engulfing
  - 0: None
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_engulfing_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_engulfing")


@configurable
@validate
def cdl_harami(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Harami pattern.

  Returns:
  - 100: Bullish Harami
  - -100: Bearish Harami
  - 0: None
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_harami_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_harami")


@configurable
@validate
def cdl_shooting_star(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Shooting Star pattern.

  Returns:
  - -100: Bearish Shooting Star
  - 0: None
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_shooting_star_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_shooting_star")


@configurable
@validate
def cdl_inverted_hammer(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Inverted Hammer pattern.

  Returns:
  - 100: Bullish Inverted Hammer
  - 0: None
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_inverted_hammer_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_inverted_hammer")


@configurable
@validate
def cdl_hanging_man(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Hanging Man pattern.

  Returns:
  - -100: Bearish Hanging Man
  - 0: None
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_hanging_man_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_hanging_man")


@configurable
@validate
def cdl_marubozu(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Marubozu pattern.

  Returns:
  - 100: Bullish (White) Marubozu
  - -100: Bearish (Black) Marubozu
  - 0: None
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_marubozu_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_marubozu")


@configurable
@validate
def cdl_morning_star(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Morning Star pattern.

  Returns:
  - 100: Bullish Morning Star
  - 0: None
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_morning_star_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_morning_star")


@configurable
@validate
def cdl_evening_star(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Evening Star pattern.

  Returns:
  - -100: Bearish Evening Star
  - 0: None
  """
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore

  result = detect_evening_star_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_evening_star")


@configurable
@validate
def cdl_3black_crows(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Black Crows pattern.

  Returns:
  - -100: Bearish Three Black Crows
  - 0: None
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_black_crows_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_3black_crows")


@configurable
@validate
def cdl_3white_soldiers(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Three White Soldiers pattern.

  Returns:
  - 100: Bullish Three White Soldiers
  - 0: None
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_white_soldiers_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_3white_soldiers")


@configurable
@validate
def cdl_3inside(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Inside Up/Down pattern.

  Returns:
  - 100: Three Inside Up (Bullish)
  - -100: Three Inside Down (Bearish)
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_inside_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_3inside")


@configurable
@validate
def cdl_3outside(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Outside Up/Down pattern.

  Returns:
  - 100: Three Outside Up (Bullish)
  - -100: Three Outside Down (Bearish)
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_outside_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_3outside")


@configurable
@validate
def cdl_3line_strike(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Line Strike pattern.

  Returns:
  - 100: Bullish Strike
  - -100: Bearish Strike
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_line_strike_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_3line_strike")


@configurable
@validate
def cdl_piercing(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Piercing Pattern.

  Returns:
  - 100: Bullish Piercing
  - 0: None
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_piercing_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_piercing")


@configurable
@validate
def cdl_dark_cloud_cover(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Dark Cloud Cover Pattern.

  Returns:
  - -100: Bearish Dark Cloud Cover
  - 0: None
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_dark_cloud_cover_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_dark_cloud_cover")


@configurable
@validate
def cdl_kicking(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Kicking Pattern.

  Returns:
  - 100: Bullish Kicking
  - -100: Bearish Kicking
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_kicking_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_kicking")


@configurable
@validate
def cdl_matching_low(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Matching Low Pattern.

  Returns:
  - 100: Bullish Matching Low
  - 0: None
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_matching_low_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_matching_low")


def _alloc_ohlc(open_, high, low, close):
  # Helper to reduce boilerplate
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False)) # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False)) # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False)) # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False)) # pyright: ignore
  return o, h, l, c
