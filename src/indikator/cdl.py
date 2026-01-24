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
