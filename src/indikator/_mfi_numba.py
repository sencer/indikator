"""Numba-optimized MFI (Money Flow Index) calculation.

This module contains JIT-compiled functions for MFI calculation.
Separated for better code organization and testability.
"""
# pyright: reportAny=false

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True)  # pyright: ignore[reportUntypedFunctionDecorator]  # pragma: no cover
def compute_mfi_numba(
  typical_prices: NDArray[np.float64],
  volumes: NDArray[np.float64],
  window: int,
  epsilon: float = 1e-9,
) -> NDArray[np.float64]:
  """Numba JIT-compiled MFI calculation.

  MFI is a volume-weighted version of RSI that measures buying/selling pressure.

  Formula:
  1. Money Flow = Typical Price * Volume
  2. Positive Money Flow = sum of money flow when typical price increases
  3. Negative Money Flow = sum of money flow when typical price decreases
  4. Money Ratio = Positive MF / Negative MF
  5. MFI = 100 - (100 / (1 + Money Ratio))

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

  # Calculate rolling positive and negative money flows
  for i in range(window, n):
    pos_mf = 0.0
    neg_mf = 0.0

    # Look back over window period
    for j in range(i - window + 1, i + 1):
      if j == 0:
        # First bar has no previous comparison, skip
        continue

      # Compare current typical price to previous
      if typical_prices[j] > typical_prices[j - 1]:
        pos_mf += money_flow[j]
      elif typical_prices[j] < typical_prices[j - 1]:
        neg_mf += money_flow[j]
      # If equal, neither positive nor negative

    # Calculate MFI
    if neg_mf < epsilon:
      mfi[i] = 100.0  # No selling pressure, MFI = 100
    else:
      money_ratio = pos_mf / neg_mf
      mfi[i] = 100.0 - (100.0 / (1.0 + money_ratio))

  return mfi
