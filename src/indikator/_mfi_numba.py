"""Numba-optimized MFI (Money Flow Index) calculation.

This module contains JIT-compiled functions for MFI calculation.
Separated for better code organization and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True)  # pragma: no cover
def compute_mfi_numba(
  typical_prices: NDArray[np.float64],
  volumes: NDArray[np.float64],
  window: int,
  epsilon: float = 1e-9,
) -> NDArray[np.float64]:
  """Numba JIT-compiled MFI calculation with O(n) complexity.

  MFI is a volume-weighted version of RSI that measures buying/selling pressure.

  Formula:
  1. Money Flow = Typical Price * Volume
  2. Positive Money Flow = sum of money flow when typical price increases
  3. Negative Money Flow = sum of money flow when typical price decreases
  4. Money Ratio = Positive MF / Negative MF
  5. MFI = 100 - (100 / (1 + Money Ratio))

  This implementation uses a sliding window approach for O(n) complexity
  instead of the naive O(n*window) approach.

  Args:
    typical_prices: Array of typical prices ((H+L+C)/3)
    volumes: Array of volumes
    window: Lookback period (typically 14)
    epsilon: Small value to prevent division by zero

  Returns:
    Array of MFI values (0-100 range, NaN for initial bars)
  """
  n = len(typical_prices)
  mfi = np.full(n, np.nan)

  if n < window + 1:  # Need window+1 bars for first calculation
    return mfi

  # Calculate money flow
  money_flow = typical_prices * volumes

  # Precompute directional money flow arrays
  # +1 for positive direction, -1 for negative, 0 for flat
  pos_mf_contrib = np.zeros(n, dtype=np.float64)
  neg_mf_contrib = np.zeros(n, dtype=np.float64)

  for i in range(1, n):
    if typical_prices[i] > typical_prices[i - 1]:
      pos_mf_contrib[i] = money_flow[i]
    elif typical_prices[i] < typical_prices[i - 1]:
      neg_mf_contrib[i] = money_flow[i]

  # Initialize window sums for the first valid position
  pos_mf_sum = 0.0
  neg_mf_sum = 0.0

  # Initialize: sum the first window of contributions (indices 1 to window)
  for i in range(1, window + 1):
    pos_mf_sum += pos_mf_contrib[i]
    neg_mf_sum += neg_mf_contrib[i]

  # Calculate MFI for position 'window' (first valid)
  if neg_mf_sum < epsilon:
    mfi[window] = 100.0
  else:
    money_ratio = pos_mf_sum / neg_mf_sum
    mfi[window] = 100.0 - (100.0 / (1.0 + money_ratio))

  # Slide the window: O(n) instead of O(n*window)
  for i in range(window + 1, n):
    # Add new element entering the window
    pos_mf_sum += pos_mf_contrib[i]
    neg_mf_sum += neg_mf_contrib[i]

    # Remove old element leaving the window
    # The element leaving is at index (i - window)
    pos_mf_sum -= pos_mf_contrib[i - window]
    neg_mf_sum -= neg_mf_contrib[i - window]

    # Calculate MFI
    if neg_mf_sum < epsilon:
      mfi[i] = 100.0  # No selling pressure, MFI = 100
    else:
      money_ratio = pos_mf_sum / neg_mf_sum
      mfi[i] = 100.0 - (100.0 / (1.0 + money_ratio))

  return mfi
