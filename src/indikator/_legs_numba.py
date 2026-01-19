"""Numba-optimized ZigZag calculation.

This module contains JIT-compiled functions for identifying ZigZag legs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_zigzag_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  deviation: float,
  percentage_mode: bool,
) -> tuple[
  NDArray[np.int8],
  NDArray[np.int64],
  NDArray[np.int64],
  NDArray[np.float64],
  NDArray[np.float64],
]:
  """Identify ZigZag legs.

  Args:
    high: High prices
    low: Low prices
    close: Close prices (used for calculation depending on preference, usually High/Low used)
    deviation: Minimum deviation to form a leg
    percentage_mode: True for % deviation, False for absolute

  Returns:
    Tuple of arrays: (directions, start_indices, end_indices, start_prices, end_prices)
  """
  n = len(high)
  if n == 0:
    empty_i8 = np.array([0], dtype=np.int8)[:-1]
    empty_i64 = np.array([0], dtype=np.int64)[:-1]
    empty_f64 = np.array([0.0], dtype=np.float64)[:-1]
    return empty_i8, empty_i64, empty_i64, empty_f64, empty_f64

  # We can't know the number of legs in advance, so we allocate max possible
  max_legs = n
  directions = np.zeros(max_legs, dtype=np.int8)
  start_indices = np.zeros(max_legs, dtype=np.int64)
  end_indices = np.zeros(max_legs, dtype=np.int64)
  start_prices = np.zeros(max_legs, dtype=np.float64)
  end_prices = np.zeros(max_legs, dtype=np.float64)

  leg_count = 0

  # Initialize
  # Find first trend
  # We need to find the first swing that exceeds deviation

  start_idx = 0
  last_pivot_price = 0.0
  last_pivot_idx = 0
  trend = 0  # 1=Up, -1=Down, 0=Unknown

  # Find initial trend
  # We look ahead until a move > deviation occurs
  first_high = high[0]
  first_low = low[0]

  # Simple initialization: Assume trend matches first move > deviation
  for i in range(1, n):
    h = high[i]
    l = low[i]

    # Check move from start (using close[0] or high/low[0]?)
    # Convention: Pivot at 0 is usually set to High[0] or Low[0] retrospectively.
    # Let's look for High - Low range traversal.

    diff_up = h - first_low
    diff_down = first_high - l

    dev_val_up = diff_up / first_low if percentage_mode else diff_up
    dev_val_down = diff_down / first_high if percentage_mode else diff_down

    if dev_val_up > deviation and dev_val_down > deviation:
      # Outside bar, huge volatility?
      # Whichever is larger?
      if dev_val_up > dev_val_down:
        trend = 1
        last_pivot_price = first_low
        last_pivot_idx = 0
      else:
        trend = -1
        last_pivot_price = first_high
        last_pivot_idx = 0
      break
    if dev_val_up > deviation:
      trend = 1
      last_pivot_price = first_low
      last_pivot_idx = 0
      break
    if dev_val_down > deviation:
      trend = -1
      last_pivot_price = first_high
      last_pivot_idx = 0
      break

  if trend == 0:
    # No moves > deviation found in entire history
    return (
      directions[:0],
      start_indices[:0],
      end_indices[:0],
      start_prices[:0],
      end_prices[:0],
    )

  # Current pivot (extreme point of current leg)
  curr_pivot_price = last_pivot_price
  curr_pivot_idx = last_pivot_idx

  # If trend=1 (Up), we just came from a Low (last_pivot). We are looking for a High.
  # If trend=-1 (Down), we just came from a High (last_pivot). We are looking for a Low.

  # Actually, if Trend=1, we are IN an Up leg.
  # So last_pivot was the Low.
  # We search for the Highest High.
  # Until we drop by deviation from that Highest High.

  # If we initialized trend=1, it means we found a move UP.
  # So the START of this leg was the Low at index 0.
  # We are currently establishing the High of this Up leg.

  # Initialize curr_pivot (the extreme of the current leg so far)
  # If Up leg, curr pivot is High[i]
  # If Down leg, curr pivot is Low[i]

  # Restart loop from i where we broke
  # But actually, we process sequentially.

  # Need to handle the first bar correctly.
  if trend == 1:
    curr_pivot_price = high[last_pivot_idx]  # Tentative high
    curr_pivot_idx = last_pivot_idx
  else:
    curr_pivot_price = low[last_pivot_idx]  # Tentative low
    curr_pivot_idx = last_pivot_idx

  # Iterate
  for i in range(last_pivot_idx + 1, n):
    h = high[i]
    l = low[i]

    if trend == 1:  # Up leg
      if h > curr_pivot_price:
        # New high in current up leg, extend leg
        curr_pivot_price = h
        curr_pivot_idx = i
      else:
        # Check for reversal
        # Reversal distance from Highest High
        dist = curr_pivot_price - l
        change = dist / curr_pivot_price if percentage_mode else dist

        if change >= deviation:
          # Confirmed reversal
          # The Up leg ended at curr_pivot_idx

          # Record Up Leg
          directions[leg_count] = 1
          start_indices[leg_count] = last_pivot_idx
          end_indices[leg_count] = curr_pivot_idx
          start_prices[leg_count] = last_pivot_price
          end_prices[leg_count] = curr_pivot_price
          leg_count += 1

          # Switch trend to Down
          trend = -1
          last_pivot_price = curr_pivot_price
          last_pivot_idx = curr_pivot_idx

          # Current bar becomes the new tentative low
          curr_pivot_price = l
          curr_pivot_idx = i

    elif l < curr_pivot_price:
      # New low in current down leg
      curr_pivot_price = l
      curr_pivot_idx = i
    else:
      # Check for reversal (Up)
      dist = h - curr_pivot_price
      change = (
        dist / curr_pivot_price if percentage_mode else dist
      )  # Usually % from Low

      if change >= deviation:
        # Confirmed reversal
        # Down leg ended

        # Record Down Leg
        directions[leg_count] = -1
        start_indices[leg_count] = last_pivot_idx
        end_indices[leg_count] = curr_pivot_idx
        start_prices[leg_count] = last_pivot_price
        end_prices[leg_count] = curr_pivot_price
        leg_count += 1

        # Switch trend to Up
        trend = 1
        last_pivot_price = curr_pivot_price
        last_pivot_idx = curr_pivot_idx

        # Current bar new high
        curr_pivot_price = h
        curr_pivot_idx = i

  # Final leg (in progress)
  # Usually ZigZag ends with a line to the latest extreme?
  # Or last confirmed leg?
  # Standard usually draws to last extreme.

  # Add the last leg
  directions[leg_count] = trend
  start_indices[leg_count] = last_pivot_idx
  end_indices[leg_count] = curr_pivot_idx
  start_prices[leg_count] = last_pivot_price
  end_prices[leg_count] = curr_pivot_price
  leg_count += 1

  return (
    directions[:leg_count],
    start_indices[:leg_count],
    end_indices[:leg_count],
    start_prices[:leg_count],
    end_prices[:leg_count],
  )
