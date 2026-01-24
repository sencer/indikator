"""Pivot Points indicator module.

This module provides pivot point calculation for support and resistance levels.
"""

from typing import Literal

from datawarden import (
  Datetime,
  Finite,
  Index,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Hyper, configurable
import numpy as np
import pandas as pd

from indikator._results import PivotPointsResult


@configurable
@validate
def pivots(
  high: Validated[pd.Series, Finite, Index(Datetime), NotEmpty],
  low: Validated[pd.Series, Finite, Index(Datetime), NotEmpty],
  close: Validated[pd.Series, Finite, Index(Datetime), NotEmpty],
  method: Literal["standard", "fibonacci", "woodie", "camarilla"] = "standard",
  anchor: Hyper[str] = "D",
) -> PivotPointsResult:
  """Calculate Pivot Points.

  Pivot points are significant support and resistance levels derived from
  prior period's price action.

  Methods:
  - Standard: Floor pivots (Classic)
  - Fibonacci: Standard Pivot + Fibonacci extensions
  - Woodie: Weighted close, forward-looking
  - Camarilla: Mean reversion/Breakout levels

  Levels typically include:
  - P: Pivot Point (Central)
  - R1, R2, R3, R4: Resistance levels
  - S1, S2, S3, S4: Support levels

  Args:
    high: High prices with DatetimeIndex
    low: Low prices with DatetimeIndex
    close: Close prices with DatetimeIndex
    method: Calculation method (default: 'standard')
    anchor: Period to aggregate prior data (default: 'D' for Daily)

  Returns:
    PivotPointsResult(index, levels: dict)
  """
  # Determine reset indices (start of new sessions)
  if not isinstance(high.index, pd.DatetimeIndex):
    raise ValueError("Index must be DatetimeIndex")

  # Anchor grouping
  # We need to compute HIGH, LOW, CLOSE of the PREVIOUS session and project to CURRENT session.
  # The robust way:
  # 1. Resample to anchor frequency (e.g. 'D') to get H/L/C of each period.
  # 2. Shift by 1 to get "previous period" values.
  # 3. Reindex back to intraday frequency (ffill).
  # 4. Compute pivots from these previous-period values.

  # Grouping/Resampling
  # Explicitly construct DataFrame to ensure column names
  df = pd.DataFrame({"high": high, "low": low, "close": close})

  resampled = df.resample(anchor).agg({"high": "max", "low": "min", "close": "last"})

  # Shift to get prior period stats
  prior = resampled.shift(1)

  # Reindex to original index (ffill)
  # This broadcasts the prior day's H/L/C to all bars of current day
  aligned = prior.reindex(high.index, method="ffill")

  # Convert to arrays
  prev_high = aligned["high"].to_numpy(dtype=np.float64, copy=False)
  prev_low = aligned["low"].to_numpy(dtype=np.float64, copy=False)
  prev_close = aligned["close"].to_numpy(dtype=np.float64, copy=False)

  # Handle NaN at start (first day has no prior)
  # Arrays already have NaNs from shift+reindex.

  levels = {}

  if method == "standard":
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)

    levels = {"pivot": pp, "r1": r1, "s1": s1, "r2": r2, "s2": s2, "r3": r3, "s3": s3}

  elif method == "fibonacci":
    pp = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low
    r1 = pp + 0.382 * rng
    s1 = pp - 0.382 * rng
    r2 = pp + 0.618 * rng
    s2 = pp - 0.618 * rng
    r3 = pp + 1.000 * rng
    s3 = pp - 1.000 * rng

    levels = {"pivot": pp, "r1": r1, "s1": s1, "r2": r2, "s2": s2, "r3": r3, "s3": s3}

  elif method == "woodie":
    # Woodie uses CURRENT open in formula usually, but some variations use close.
    # Standard Woodie: P = (H + L + 2*C) / 4.
    pp = (prev_high + prev_low + 2 * prev_close) / 4.0
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)

    levels = {"pivot": pp, "r1": r1, "s1": s1, "r2": r2, "s2": s2}

  elif method == "camarilla":
    rng = prev_high - prev_low
    r1 = prev_close + rng * 1.1 / 12.0
    s1 = prev_close - rng * 1.1 / 12.0
    r2 = prev_close + rng * 1.1 / 6.0
    s2 = prev_close - rng * 1.1 / 6.0
    r3 = prev_close + rng * 1.1 / 4.0
    s3 = prev_close - rng * 1.1 / 4.0
    r4 = prev_close + rng * 1.1 / 2.0
    s4 = prev_close - rng * 1.1 / 2.0
    pp = (prev_high + prev_low + prev_close) / 3.0

    levels = {
      "pivot": pp,
      "r1": r1,
      "s1": s1,
      "r2": r2,
      "s2": s2,
      "r3": r3,
      "s3": s3,
      "r4": r4,
      "s4": s4,
    }
  else:
    raise ValueError(f"Invalid method: {method}")

  return PivotPointsResult(index=high.index, levels=levels)
