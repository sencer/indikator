"""Numba-optimized zigzag legs calculation.

This module contains JIT-compiled functions for zigzag leg counting.
Separated for better code organization and testability.
"""
# pyright: reportAttributeAccessIssue=false, reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True)  # pyright: ignore[reportUntypedFunctionDecorator]  # pragma: no cover
def compute_zigzag_legs_numba(
  closes: NDArray[np.float64],
  threshold: float,
  min_distance_pct: float,
  confirmation_bars: int,
  epsilon: float = 1e-9,
) -> NDArray[np.float64]:
  # Complexity suppressions: This function is performance-critical and optimized
  # for Numba JIT compilation. Refactoring for lower complexity would hurt performance.
  # ruff: noqa: C901, PLR0912, PLR0915, PLR0914, PLR1702
  """Numba JIT-compiled zigzag leg counter with confirmation and noise filtering.

  Tracks market structure (Higher Highs/Lower Lows) to count legs within
  the current trend. Uses signed counts to indicate trend direction:
  - Positive counts (+1, +2, +3...) for bullish trends (Higher Highs)
  - Negative counts (-1, -2, -3...) for bearish trends (Lower Lows)
  - Zero (0) for no established trend

  Args:
    closes: Array of closing prices
    threshold: Minimum percentage change to trigger a reversal
    min_distance_pct: Minimum percentage move to update pivot
    confirmation_bars: Number of bars to confirm a reversal (0 = immediate)
    epsilon: Small value to prevent division by zero

  Returns:
    Array of signed leg counts for each bar
  """
  n = len(closes)
  legs = np.zeros(n, dtype=np.float64)

  if n == 0:
    return legs

  pivot = closes[0]
  trend = 0  # Current leg direction: 1 (Up), -1 (Down), 0 (None)
  count = 0
  confirmation_counter = 0
  pending_reversal = False
  pending_pivot = 0.0

  # Market Structure Tracking
  structure_trend = 0  # 1 (Bullish), -1 (Bearish), 0 (None)
  last_high = float("-inf")
  last_low = float("inf")
  current_leg_is_impulse = False

  for i in range(n):
    price = closes[i]

    # Division by zero protection
    change = 0.0 if abs(pivot) < epsilon else (price - pivot) / pivot

    if trend == 0:
      # Establish initial trend
      if abs(change) > threshold:
        trend = 1 if change > 0 else -1
        pivot = price

        # Initialize structure
        if trend == 1:
          structure_trend = 1
          last_high = price
          last_low = -np.inf
          count = 1
          current_leg_is_impulse = True
        else:
          structure_trend = -1
          last_low = price
          last_high = np.inf
          count = 1
          current_leg_is_impulse = True

    elif trend == 1:  # Up Leg
      if price > pivot:
        # Only update pivot if move is significant enough
        distance = (price - pivot) / (pivot + epsilon)
        if distance > min_distance_pct:
          pivot = price
          # Check for structure break (Live Counting)
          if structure_trend == 1:
            if pivot > last_high and not current_leg_is_impulse:
              count += 1
              current_leg_is_impulse = True
          elif structure_trend == -1 and pivot > last_high:
            # We are in a Bearish trend, but this Up leg just broke the last high!
            # Trend Change: Bearish -> Bullish
            structure_trend = 1
            count = 1
            last_high = pivot
            last_low = -np.inf
            current_leg_is_impulse = True

          # Cancel any pending reversal if we make new highs
          pending_reversal = False
          confirmation_counter = 0

      elif change < -threshold:  # Potential reversal down
        if not pending_reversal:
          # Start confirmation period
          pending_reversal = True
          pending_pivot = price
          confirmation_counter = 1
        else:
          # Continue confirmation
          confirmation_counter += 1
          # Update pending pivot to lowest price during confirmation
          pending_pivot = min(pending_pivot, price)

        # Confirm reversal if we've waited long enough
        if confirmation_counter >= confirmation_bars:
          trend = -1
          high_of_leg = pivot

          # Update structure based on the High we just finished
          if structure_trend == 1:  # Bullish
            last_high = high_of_leg
            current_leg_is_impulse = False  # Next leg (Down) starts as correction

          elif structure_trend == -1:  # Bearish
            if high_of_leg > last_high:
              # Higher High - Trend Change to Bullish
              structure_trend = 1
              count = 1
              last_high = high_of_leg
              last_low = -np.inf  # Reset low for new bullish trend
              current_leg_is_impulse = True  # First leg of new trend is impulse
            else:
              # Lower High - Correction in Bearish trend
              last_high = high_of_leg
              current_leg_is_impulse = False

          # Now we are in Down Leg.
          pivot = pending_pivot  # This is the current Low

          pending_reversal = False
          confirmation_counter = 0

      # Price moved back - cancel pending reversal
      elif pending_reversal:
        pending_reversal = False
        confirmation_counter = 0

    elif trend == -1:  # Down Leg
      if price < pivot:
        # Only update pivot if move is significant enough
        distance = abs((price - pivot) / (pivot + epsilon))
        if distance > min_distance_pct:
          pivot = price
          # Check for structure break (Live Counting)
          if structure_trend == -1:
            if pivot < last_low and not current_leg_is_impulse:
              count += 1
              current_leg_is_impulse = True
          elif structure_trend == 1 and pivot < last_low:
            # We are in a Bullish trend, but this Down leg just broke the last low!
            # Trend Change: Bullish -> Bearish
            structure_trend = -1
            count = 1
            last_low = pivot
            last_high = np.inf
            current_leg_is_impulse = True

          # Cancel any pending reversal if we make new lows
          pending_reversal = False
          confirmation_counter = 0

      elif change > threshold:  # Potential reversal up
        if not pending_reversal:
          # Start confirmation period
          pending_reversal = True
          pending_pivot = price
          confirmation_counter = 1
        else:
          # Continue confirmation
          confirmation_counter += 1
          # Update pending pivot to highest price during confirmation
          pending_pivot = max(pending_pivot, price)

        # Confirm reversal if we've waited long enough
        if confirmation_counter >= confirmation_bars:
          trend = 1
          low_of_leg = pivot

          # Update structure based on the Low we just finished
          if structure_trend == 1:  # Bullish
            if low_of_leg < last_low:
              # Lower Low - Trend Change to Bearish
              structure_trend = -1
              count = 1
              last_low = low_of_leg
              last_high = np.inf  # Reset high for new bearish trend
              current_leg_is_impulse = True
            else:
              # Higher Low - Correction in Bullish trend
              last_low = low_of_leg
              current_leg_is_impulse = False

          elif structure_trend == -1:  # Bearish
            last_low = low_of_leg
            current_leg_is_impulse = False

          # Now we are in Up Leg
          pivot = pending_pivot  # This is the current High

          pending_reversal = False
          confirmation_counter = 0
      # Price moved back - cancel pending reversal
      elif pending_reversal:
        pending_reversal = False
        confirmation_counter = 0

    # Store signed count: positive for bullish, negative for bearish
    legs[i] = count * structure_trend

  return legs
