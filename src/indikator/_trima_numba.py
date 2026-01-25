"""Numba-optimized Triangular Moving Average (TRIMA) calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)
def compute_trima_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate Triangular Moving Average (TRIMA) using a fused kernel.
  # ruff: noqa: PLR0912, PLR0913, PLR0914, PLR0915, C901, PLR2004, ERA001, SIM, N806, B007, ARG, E741, TC, ANN

    Fuses two simple moving averages into a single pass using a ring buffer
    for intermediate SMA values. O(N) time, O(N) memory (output only).
  """
  n = len(prices)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n == 0 or period <= 0:
    return out

  # Determine periods
  if period % 2 == 1:
    p1 = (period + 1) // 2
    p2 = p1
  else:
    p1 = period // 2
    p2 = p1 + 1

  # Total warmup needed
  # SMA1 needs p1 inputs (valid at index p1-1)
  # SMA2 needs p2 SMA1 inputs.
  # First valid SMA1 is at p1-1.
  # SMA2 takes SMA1[p1-1], SMA1[p1], ... SMA1[p1-1 + p2-1].
  # So first valid TRIMA is at p1 - 1 + p2 - 1 = p1 + p2 - 2?
  # Let's verify with period=4 (p1=2, p2=3). valid=2+3-2=3.
  # TA-Lib TRIMA(4) valid at index 3. Correct.
  # period=3 (p1=2, p2=2). valid=2+2-2=2. Correct.

  if n < p1 + p2 - 1:
    return out

  # Ring Buffer for SMA1 history
  # We need the last p2 values of SMA1 to compute the rolling window for SMA2.
  # sma1_buf stores [sma1(t-p2+1), ..., sma1(t)]
  sma1_buf = np.empty(p2, dtype=np.float64)
  # Since we fill it sequentially, we can just use a pointer.
  buf_idx = 0

  # State variables
  sum1 = 0.0
  sum2 = 0.0

  # Count of VALID outputs from SMA1
  count1_valid = 0

  # Loop over prices
  for i in range(n):
    val = prices[i]

    # --- SMA 1 Update ---
    # Add new value
    sum1 += val
    # Remove old value if beyond window p1
    if i >= p1:
      sum1 -= prices[i - p1]

    # Check if SMA1 is valid
    current_sma1 = np.nan
    if i >= p1 - 1:
      current_sma1 = sum1 / p1

      # --- SMA 2 Update ---
      # We only update SMA2 if we have a valid SMA1 input
      # Wait, SMA2 logic is: sum over current window of SMA1 values.
      # We feed 'current_sma1' into SMA2 pipeline.

      sum2 += current_sma1

      # Remove old SMA1 value from sum2 window?
      # The value to remove is the one that entered SMA2 window p2 steps ago.
      # We store history in sma1_buf.

      # We need to know if we have filled p2 items in SMA2 window.
      count1_valid += 1

      if count1_valid > p2:
        # Remove the oldest value from the buffer
        # The buffer at buf_idx holds the oldest value (about to be overwritten)
        old_sma1 = sma1_buf[buf_idx]
        sum2 -= old_sma1

      # Write current to buffer
      sma1_buf[buf_idx] = current_sma1
      buf_idx += 1
      if buf_idx == p2:
        buf_idx = 0

      # Check if SMA2 is valid
      # SMA2 is valid once we have processed p2 valid inputs from SMA1
      if count1_valid >= p2:
        out[i] = sum2 / p2

  return out
