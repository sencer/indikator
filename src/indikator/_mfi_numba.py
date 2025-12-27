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


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
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

  # Initialize window sums for the first valid position
  pos_mf_sum = 0.0
  neg_mf_sum = 0.0

  # Initialize: sum the first window of contributions (indices 1 to window)
  for i in range(1, window + 1):
    mf = typical_prices[i] * volumes[i]
    if typical_prices[i] > typical_prices[i - 1]:
      pos_mf_sum += mf
    elif typical_prices[i] < typical_prices[i - 1]:
      neg_mf_sum += mf

  # Calculate MFI for position 'window' (first valid)
  mfi[window] = (
    100.0 if neg_mf_sum < epsilon else 100.0 - (100.0 / (1.0 + pos_mf_sum / neg_mf_sum))
  )

  # Slide the window: O(n) instead of O(n*window)
  for i in range(window + 1, n):
    # Add new element entering the window at index i
    mf_in = typical_prices[i] * volumes[i]
    if typical_prices[i] > typical_prices[i - 1]:
      pos_mf_sum += mf_in
    elif typical_prices[i] < typical_prices[i - 1]:
      neg_mf_sum += mf_in

    # Remove old element leaving the window at index (i - window)
    # The contribution at index j depends on j and j-1
    j = i - window
    mf_out = typical_prices[j] * volumes[j]
    if typical_prices[j] > typical_prices[j - 1]:
      pos_mf_sum -= mf_out
    elif typical_prices[j] < typical_prices[j - 1]:
      neg_mf_sum -= mf_out

    # Calculate MFI
    mfi[i] = (
      100.0
      if neg_mf_sum < epsilon
      else 100.0 - (100.0 / (1.0 + pos_mf_sum / neg_mf_sum))
    )

  return mfi
