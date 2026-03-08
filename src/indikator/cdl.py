"""Candlestick Pattern Recognition module.

This module provides detection for common candlestick patterns.
Returns integer series:
- 100: Bullish pattern detected
- -100: Bearish pattern detected
- 0: No pattern
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from indikator.numba.cdl import (
  detect_3black_crows_numba,
  detect_3white_soldiers_numba,
  detect_abandoned_baby_numba,
  detect_advance_block_numba,
  detect_belt_hold_numba,
  detect_breakaway_numba,
  detect_closing_marubozu_numba,
  detect_conceal_baby_swallow_numba,
  detect_counterattack_numba,
  detect_dark_cloud_cover_numba,
  detect_doji_numba,
  detect_doji_star_numba,
  detect_dragonfly_doji_numba,
  detect_engulfing_numba,
  detect_evening_doji_star_numba,
  detect_evening_star_numba,
  detect_gap_side_by_side_white_numba,
  detect_gravestone_doji_numba,
  detect_hammer_numba,
  detect_hanging_man_numba,
  detect_harami_cross_parallel,
  detect_harami_parallel,
  detect_high_wave_parallel,
  detect_hikkake_modified_numba,
  detect_hikkake_numba,
  detect_homing_pigeon_numba,
  detect_identical_three_crows_numba,
  detect_in_neck_parallel,
  detect_inverted_hammer_numba,
  detect_kicking_numba,
  detect_ladder_bottom_parallel,
  detect_long_legged_doji_numba,
  detect_long_line_numba,
  detect_marubozu_numba,
  detect_mat_hold_parallel,
  detect_matching_low_parallel,
  detect_morning_doji_star_numba,
  detect_morning_star_numba,
  detect_on_neck_parallel,
  detect_piercing_numba,
  detect_rickshaw_man_numba,
  detect_rise_fall_three_methods_numba,
  detect_separating_lines_numba,
  detect_shooting_star_numba,
  detect_short_line_numba,
  detect_spinning_top_numba,
  detect_stalled_pattern_numba,
  detect_stick_sandwich_parallel,
  detect_takuri_numba,
  detect_tasuki_gap_numba,
  detect_three_inside_numba,
  detect_three_line_strike_numba,
  detect_three_outside_numba,
  detect_three_stars_in_south_numba,
  detect_thrusting_parallel,
  detect_tristar_numba,
  detect_two_crows_numba,
  detect_unique_three_river_numba,
  detect_upside_gap_two_crows_numba,
  detect_xsidegap3methods_numba,
)
from indikator.numba.sma import compute_sma_numba
from indikator.utils import to_numpy


@configurable
@validate
def cdl_doji(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Doji pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_doji_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_doji")


@configurable
@validate
def cdl_hammer(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Hammer pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_hammer_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_hammer")


@configurable
@validate
def cdl_engulfing(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Engulfing pattern.

  Returns:
  - 100: Bullish Engulfing
  - -100: Bearish Engulfing
  - 0: None
  """
  o = to_numpy(open_)
  h = to_numpy(high)
  low_arr = to_numpy(low)
  c = to_numpy(close)

  result = detect_engulfing_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_engulfing")


@configurable
@validate
def cdl_harami(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Harami pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  # Parallel implementation (Strategy #2) typically 0.7x TA-Lib on 1M rows
  result = detect_harami_parallel(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_harami")


@configurable
@validate
def cdl_shooting_star(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Shooting Star pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_shooting_star_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_shooting_star")


@configurable
@validate
def cdl_inverted_hammer(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Inverted Hammer pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_inverted_hammer_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_inverted_hammer")


@configurable
@validate
def cdl_hanging_man(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Hanging Man pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_hanging_man_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_hanging_man")


@configurable
@validate
def cdl_marubozu(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Marubozu pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_marubozu_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_marubozu")


@configurable
@validate
def cdl_morning_star(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  penetration: Hyper[float, Ge[0.0]] = 0.3,
) -> pd.Series:
  """Detect Morning Star pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_morning_star_numba(o, h, low_arr, c, penetration=penetration)
  return pd.Series(result, index=open_.index, name="cdl_morning_star")


@configurable
@validate
def cdl_evening_star(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  penetration: Hyper[float, Ge[0.0]] = 0.3,
) -> pd.Series:
  """Detect Evening Star pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_evening_star_numba(o, h, low_arr, c, penetration=penetration)
  return pd.Series(result, index=open_.index, name="cdl_evening_star")


@configurable
@validate
def cdl_3white_soldiers(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Three White Soldiers pattern.

  Returns:
  - 100: Bullish Three White Soldiers
  - 0: None
  """
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_3white_soldiers_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_3white_soldiers")


@configurable
@validate
def cdl_3inside(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Inside Up/Down pattern.

  Returns:
  - 100: Three Inside Up (Bullish)
  - -100: Three Inside Down (Bearish)
  """
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_inside_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_3inside")


@configurable
@validate
def cdl_3outside(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Outside Up/Down pattern.

  Returns:
  - 100: Three Outside Up (Bullish)
  - -100: Three Outside Down (Bearish)
  """
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_outside_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_3outside")


@configurable
@validate
def cdl_3line_strike(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Line Strike pattern.

  Returns:
  - 100: Bullish Strike
  - -100: Bearish Strike
  """
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_line_strike_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_3line_strike")


@configurable
@validate
def cdl_tristar(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Tristar."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_tristar_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_tristar")


@configurable
@validate
def cdl_piercing(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Piercing Pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_piercing_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_piercing")


@configurable
@validate
def cdl_dark_cloud_cover(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  penetration: Hyper[float, Ge[0.0]] = 0.5,
) -> pd.Series:
  """Detect Dark Cloud Cover."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_dark_cloud_cover_numba(o, h, low_arr, c, penetration=penetration)
  return pd.Series(result, index=open_.index, name="cdl_dark_cloud_cover")


@configurable
@validate
def cdl_kicking(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Kicking Pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_kicking_numba(o, h, low_arr, c, by_length=False)
  return pd.Series(result, index=open_.index, name="cdl_kicking")


@configurable
@validate
def cdl_kicking_by_length(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Kicking - bull/bear determined by the longer marubozu."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_kicking_numba(o, h, low_arr, c, by_length=True)
  return pd.Series(result, index=open_.index, name="cdl_kicking_by_length")


def _rolling_sma_prev(
  series: pd.Series | NDArray[np.float64], period: int, scale: float = 1.0
) -> NDArray[np.float64]:
  """Calculate rolling SMA of previous 'period' elements."""
  arr = to_numpy(series) if isinstance(series, pd.Series) else series

  return compute_sma_numba(arr, period, scale=scale, shifted=True)


def _rolling_sma(
  series: pd.Series | NDArray[np.float64], period: int, scale: float = 1.0
) -> NDArray[np.float64]:
  """Calculate rolling SMA of 'period' elements (inclusive of current)."""
  arr = to_numpy(series) if isinstance(series, pd.Series) else series

  return compute_sma_numba(arr, period, scale=scale, shifted=False)


@configurable
@validate
def cdl_short_line(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Short Line."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_short_line_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_short_line")


@configurable
@validate
def cdl_spinning_top(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Spinning Top pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_spinning_top_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_spinning_top")


@configurable
@validate
def cdl_matching_low(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Matching Low pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  # Parallel implementation (Strategy #2)
  result = detect_matching_low_parallel(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_matching_low")


@configurable
@validate
def cdl_high_wave(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect High Wave pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)

  result = detect_high_wave_parallel(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_high_wave")


@configurable
@validate
def cdl_long_legged_doji(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Long Legged Doji."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_long_legged_doji_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_long_legged_doji")


@configurable
@validate
def cdl_rickshaw_man(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Rickshaw Man."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_rickshaw_man_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_rickshaw_man")


@configurable
@validate
def cdl_tasuki_gap(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Tasuki Gap."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  rng = h - low_arr
  avg_near = _rolling_sma_prev(rng, 5, scale=0.2)

  result = detect_tasuki_gap_numba(o, h, low_arr, c, avg_near=avg_near)
  return pd.Series(result, index=open_.index, name="cdl_tasuki_gap")


@configurable
@validate
def cdl_separating_lines(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Separating Lines pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)

  result = detect_separating_lines_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_separating_lines")


@configurable
@validate
def cdl_gap_side_by_side_white(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Gap Side-by-Side White Lines.

  Returns:
  - 100: Bullish
  - -100: Bearish
  """
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)

  # TA-Lib 'Near' defaults: Period=5, Factor=0.2, Type=HighLow
  # TA-Lib 'Equal' defaults: Period=5, Factor=0.05, Type=HighLow
  rng = h - low_arr
  avg_rng_5 = _rolling_sma_prev(rng, 5)

  avg_near = avg_rng_5 * 0.2
  avg_equal = avg_rng_5 * 0.05

  result = detect_gap_side_by_side_white_numba(
    o, h, low_arr, c, avg_near=avg_near, avg_equal=avg_equal
  )
  return pd.Series(result, index=open_.index, name="cdl_gap_side_by_side_white")


@configurable
@validate
def cdl_2crows(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Two Crows pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_two_crows_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_2crows")


@configurable
@validate
def cdl_upside_gap_two_crows(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Upside Gap Two Crows pattern."""
  o, h, l_arr, c = _alloc_ohlc(open_, high, low, close)

  result = detect_upside_gap_two_crows_numba(o, h, l_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_upside_gap_two_crows")


@configurable
@validate
def cdl_abandoned_baby(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  penetration: Hyper[float, Ge[0.0]] = 0.3,
) -> pd.Series:
  """Detect Abandoned Baby."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_abandoned_baby_numba(o, h, low_arr, c, penetration=penetration)
  return pd.Series(result, index=open_.index, name="cdl_abandoned_baby")


@configurable
@validate
def cdl_advance_block(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Advance Block."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_advance_block_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_advance_block")


@configurable
@validate
def cdl_belt_hold(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Belt Hold."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_belt_hold_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_belt_hold")


@configurable
@validate
def cdl_3black_crows(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Black Crows."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_3black_crows_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_3black_crows")


@configurable
@validate
def cdl_breakaway(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Breakaway."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)

  result = detect_breakaway_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_breakaway")
  return pd.Series(result, index=open_.index, name="cdl_breakaway")


@configurable
@validate
def cdl_closing_marubozu(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Closing Marubozu."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_closing_marubozu_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_closing_marubozu")


@configurable
@validate
def cdl_dragonfly_doji(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Dragonfly Doji."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_dragonfly_doji_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_dragonfly_doji")


@configurable
@validate
def cdl_gravestone_doji(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Gravestone Doji."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_gravestone_doji_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_gravestone_doji")


@configurable
@validate
def cdl_hikkake(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Hikkake."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_hikkake_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_hikkake")


@configurable
@validate
def cdl_homing_pigeon(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Homing Pigeon."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_homing_pigeon_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_homing_pigeon")


@configurable
@validate
def cdl_identical_3crows(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Identical Three Crows."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_identical_three_crows_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_identical_3crows")


@configurable
@validate
def cdl_in_neck(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect In-Neck."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  # Parallel implementation (Strategy #2)
  result = detect_in_neck_parallel(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_in_neck")


@configurable
@validate
def cdl_ladder_bottom(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Ladder Bottom."""
  o, h, l_arr, c = _alloc_ohlc(open_, high, low, close)
  # Parallel implementation (Strategy #2)
  result = detect_ladder_bottom_parallel(o, h, l_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_ladder_bottom")


@configurable
@validate
def cdl_long_line(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Long Line."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_long_line_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_long_line")


@configurable
@validate
def cdl_mat_hold(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  penetration: Hyper[float, Ge[0.0]] = 0.5,
) -> pd.Series:
  """Detect Mat Hold pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  # Parallel implementation (Strategy #2)
  result = detect_mat_hold_parallel(o, h, low_arr, c, penetration)
  return pd.Series(result, index=open_.index, name="cdl_mat_hold")


@configurable
@validate
def cdl_on_neck(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect On-Neck."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  # Parallel implementation (Strategy #2)
  result = detect_on_neck_parallel(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_on_neck")


@configurable
@validate
def cdl_rise_fall_3methods(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Rise/Fall Three Methods pattern.

  Returns:
  - 100: Rising Three Methods (Bullish)
  - -100: Falling Three Methods (Bearish)
  """
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_rise_fall_three_methods_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_rise_fall_3methods")


@configurable
@validate
def cdl_stalled_pattern(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Stalled Pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_stalled_pattern_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_stalled_pattern")


@configurable
@validate
def cdl_stick_sandwich(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Stick Sandwich."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  # Parallel implementation (Strategy #2)
  result = detect_stick_sandwich_parallel(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_stick_sandwich")


@configurable
@validate
def cdl_takuri(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Takuri pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_takuri_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_takuri")


@configurable
@validate
def cdl_thrusting(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Thrusting."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  # Parallel implementation (Strategy #2)
  result = detect_thrusting_parallel(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_thrusting")


@configurable
@validate
def cdl_unique_3river(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Unique 3 River pattern.

  Returns:
  - 100: Bullish
  """
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_unique_three_river_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_unique_3river")


@configurable
@validate
def cdl_counterattack(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Counterattack."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_counterattack_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_counterattack")


@configurable
@validate
def cdl_doji_star(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Doji Star pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_doji_star_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_doji_star")


@configurable
@validate
def cdl_conceal_baby_swallow(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Concealing Baby Swallow pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)

  result = detect_conceal_baby_swallow_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_conceal_baby_swallow")


@configurable
@validate
def cdl_harami_cross(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Harami Cross."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_harami_cross_parallel(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_harami_cross")


@configurable
@validate
def cdl_hikkake_mod(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Modified Hikkake."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)

  # TA-Lib 'Near' defaults: Period=5, Factor=0.2, Type=HighLow
  rng = h - low_arr
  avg_near = _rolling_sma_prev(rng, 5, scale=0.2)

  result = detect_hikkake_modified_numba(o, h, low_arr, c, avg_near=avg_near)
  return pd.Series(result, index=open_.index, name="cdl_hikkake_mod")


@configurable
@validate
def cdl_morning_doji_star(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  penetration: Hyper[float, Ge[0.0]] = 0.3,
) -> pd.Series:
  """Detect Morning Doji Star pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_morning_doji_star_numba(o, h, low_arr, c, penetration=penetration)
  return pd.Series(result, index=open_.index, name="cdl_morning_doji_star")


@configurable
@validate
def cdl_evening_doji_star(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  penetration: Hyper[float, Ge[0.0]] = 0.3,
) -> pd.Series:
  """Detect Evening Doji Star pattern."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_evening_doji_star_numba(o, h, low_arr, c, penetration=penetration)
  return pd.Series(result, index=open_.index, name="cdl_evening_doji_star")


@configurable
@configurable
@validate
def cdl_3stars_in_south(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Three Stars In The South."""
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_three_stars_in_south_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_3stars_in_south")


@configurable
@validate
def cdl_xsidegap3methods(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Detect Upside/Downside Gap Three Methods.

  Returns:
  - 100: Bullish
  - -100: Bearish
  """
  o, h, low_arr, c = _alloc_ohlc(open_, high, low, close)
  result = detect_xsidegap3methods_numba(o, h, low_arr, c)
  return pd.Series(result, index=open_.index, name="cdl_xsidegap3methods")


def _alloc_ohlc(
  open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> tuple[
  NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
  # Helper to reduce boilerplate
  o = to_numpy(open_)
  h = to_numpy(high)
  low_arr = to_numpy(low)
  c = to_numpy(close)
  return o, h, low_arr, c
