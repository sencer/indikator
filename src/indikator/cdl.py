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
  detect_kicking_numba,
  detect_matching_low_numba,
  detect_piercing_numba,
  detect_high_wave_numba,
  detect_long_legged_doji_numba,
  detect_rickshaw_man_numba,
  detect_spinning_top_numba,
  detect_gap_side_by_side_white_numba,
  detect_separating_lines_numba,
  detect_tasuki_gap_numba,
  detect_tristar_numba,
  detect_two_crows_numba,
  detect_upside_gap_two_crows_numba,
  detect_abandoned_baby_numba,
  detect_advance_block_numba,
  detect_belt_hold_numba,
  detect_breakaway_numba,
  detect_closing_marubozu_numba,
  detect_dragonfly_doji_numba,
  detect_gravestone_doji_numba,
  detect_hikkake_numba,
  detect_homing_pigeon_numba,
  detect_identical_three_crows_numba,
  detect_in_neck_numba,
  detect_ladder_bottom_numba,
  detect_long_line_numba,
  detect_mat_hold_numba,
  detect_on_neck_numba,
  detect_rise_fall_three_methods_numba,
  detect_short_line_numba,
  detect_stalled_pattern_numba,
  detect_stick_sandwich_numba,
  detect_takuri_numba,
  detect_thrusting_numba,
  detect_unique_three_river_numba,
  detect_counterattack_numba,
  detect_doji_star_numba,
  detect_conceal_baby_swallow_numba,
  detect_harami_cross_numba,
  detect_hikkake_modified_numba,
  detect_morning_doji_star_numba,
  detect_evening_doji_star_numba,
  detect_kicking_by_length_numba,
  detect_three_stars_in_south_numba,
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


@configurable
@validate
def cdl_spinning_top(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Spinning Top pattern.

  Returns:
  - 100: Bullish/Neutral Spinning Top
  - -100: Bearish/Neutral Spinning Top
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_spinning_top_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_spinning_top")


@configurable
@validate
def cdl_rickshaw_man(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Rickshaw Man pattern.

  Returns:
  - 100: Detected
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_rickshaw_man_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_rickshaw_man")


@configurable
@validate
def cdl_high_wave(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect High Wave pattern.

  Returns:
  - 100: Bullish High Wave
  - -100: Bearish High Wave
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_high_wave_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_high_wave")


@configurable
@validate
def cdl_long_legged_doji(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Long Legged Doji pattern.

  Returns:
  - 100: Detected
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_long_legged_doji_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_long_legged_doji")


@configurable
@validate
def cdl_tristar(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Tristar pattern.

  Returns:
  - 100: Bullish Tristar
  - -100: Bearish Tristar
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_tristar_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_tristar")


@configurable
@validate
def cdl_tasuki_gap(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Tasuki Gap pattern.

  Returns:
  - 100: Upside Tasuki Gap
  - -100: Downside Tasuki Gap
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_tasuki_gap_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_tasuki_gap")


@configurable
@validate
def cdl_separating_lines(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Separating Lines pattern.

  Returns:
  - 100: Bullish
  - -100: Bearish
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_separating_lines_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_separating_lines")


@configurable
@validate
def cdl_gap_side_by_side_white(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Gap Side-by-Side White Lines.

  Returns:
  - 100: Bullish
  - -100: Bearish
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_gap_side_by_side_white_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_gap_side_by_side_white")


@configurable
@validate
def cdl_2crows(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Two Crows pattern.

  Returns:
  - -100: Bearish Two Crows
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_two_crows_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_2crows")


@configurable
@validate
def cdl_upside_gap_two_crows(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Upside Gap Two Crows pattern.

  Returns:
  - -100: Bearish Upside Gap Two Crows
  """
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_upside_gap_two_crows_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_upside_gap_two_crows")


@configurable
@validate
def cdl_abandoned_baby(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Abandoned Baby."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_abandoned_baby_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_abandoned_baby")


@configurable
@validate
def cdl_advance_block(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Advance Block."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_advance_block_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_advance_block")


@configurable
@validate
def cdl_belt_hold(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Belt Hold."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_belt_hold_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_belt_hold")


@configurable
@validate
def cdl_breakaway(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Breakaway."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_breakaway_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_breakaway")


@configurable
@validate
def cdl_closing_marubozu(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Closing Marubozu."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_closing_marubozu_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_closing_marubozu")


@configurable
@validate
def cdl_dragonfly_doji(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Dragonfly Doji."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_dragonfly_doji_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_dragonfly_doji")


@configurable
@validate
def cdl_gravestone_doji(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Gravestone Doji."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_gravestone_doji_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_gravestone_doji")


@configurable
@validate
def cdl_hikkake(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Hikkake."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_hikkake_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_hikkake")


@configurable
@validate
def cdl_homing_pigeon(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Homing Pigeon."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_homing_pigeon_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_homing_pigeon")


@configurable
@validate
def cdl_identical_3crows(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Identical Three Crows."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_identical_three_crows_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_identical_3crows")


@configurable
@validate
def cdl_in_neck(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect In-Neck."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_in_neck_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_in_neck")


@configurable
@validate
def cdl_ladder_bottom(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Ladder Bottom."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_ladder_bottom_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_ladder_bottom")


@configurable
@validate
def cdl_long_line(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Long Line."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_long_line_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_long_line")


@configurable
@validate
def cdl_mat_hold(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Mat Hold."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_mat_hold_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_mat_hold")


@configurable
@validate
def cdl_on_neck(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect On-Neck."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_on_neck_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_on_neck")


@configurable
@validate
def cdl_rise_fall_3methods(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Rise/Fall Three Methods."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_rise_fall_three_methods_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_rise_fall_3methods")


@configurable
@validate
def cdl_short_line(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Short Line."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_short_line_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_short_line")


@configurable
@validate
def cdl_stalled_pattern(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Stalled Pattern."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_stalled_pattern_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_stalled_pattern")


@configurable
@validate
def cdl_stick_sandwich(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Stick Sandwich."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_stick_sandwich_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_stick_sandwich")


@configurable
@validate
def cdl_takuri(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Takuri."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_takuri_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_takuri")


@configurable
@validate
def cdl_thrusting(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Thrusting."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_thrusting_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_thrusting")


@configurable
@validate
def cdl_unique_3river(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Unique 3 River."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_unique_three_river_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_unique_3river")


@configurable
@validate
def cdl_counterattack(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Counterattack."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_counterattack_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_counterattack")


@configurable
@validate
def cdl_doji_star(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Doji Star."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_doji_star_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_doji_star")


@configurable
@validate
def cdl_conceal_baby_swallow(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Concealing Baby Swallow."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_conceal_baby_swallow_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_conceal_baby_swallow")


@configurable
@validate
def cdl_harami_cross(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Harami Cross."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_harami_cross_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_harami_cross")


@configurable
@validate
def cdl_hikkake_mod(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Modified Hikkake."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_hikkake_modified_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_hikkake_mod")


@configurable
@validate
def cdl_morning_doji_star(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Morning Doji Star."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_morning_doji_star_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_morning_doji_star")


@configurable
@validate
def cdl_evening_doji_star(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Evening Doji Star."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_evening_doji_star_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_evening_doji_star")


@configurable
@validate
def cdl_kicking_by_length(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Kicking By Length."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_kicking_by_length_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_kicking_by_length")


@configurable
@validate
def cdl_3stars_in_south(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Stars In The South."""
  o, h, l, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_stars_in_south_numba(o, h, l, c)
  return pd.Series(result, index=open_.index, name="cdl_3stars_in_south")


def _alloc_ohlc(open_, high, low, close):
  # Helper to reduce boilerplate
  o = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False)) # pyright: ignore
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False)) # pyright: ignore
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False)) # pyright: ignore
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False)) # pyright: ignore
  return o, h, l, c
