"""Numba-optimized AD (Accumulation/Distribution Line) calculation.

Cumulative volume-weighted indicator using CLV.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ad_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  volume: NDArray[np.float64],
) -> NDArray[np.float64]:  # pragma: no cover
  """Numba JIT-compiled A/D Line calculation.

  A/D = cumsum(CLV * Volume)
  """
  n = len(close)
  ad = np.empty(n, dtype=np.float64)

  cumulative = 0.0

  for i in range(n):
    hl = high[i] - low[i]
    clv = (2.0 * close[i] - high[i] - low[i]) / hl if hl > EPSILON else 0.0

    cumulative += clv * volume[i]
    ad[i] = cumulative

  return ad


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_adosc_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  volume: NDArray[np.float64],
  fast_period: int,
  slow_period: int,
) -> NDArray[np.float64]:  # pragma: no cover
  """Numba JIT-compiled A/D Oscillator (Chaikin Oscillator).

  ADOSC = EMA(AD, fast) - EMA(AD, slow)

  Optimized single-pass with split warmup/main loops.

  Args:
    high: High prices
    low: Low prices
    close: Close prices
    volume: Volume
    fast_period: Fast EMA period (typically 3)
    slow_period: Slow EMA period (typically 10)

  Returns:
    Array of ADOSC values
  """
  n = len(close)

  if n < slow_period:
    return np.full(n, np.nan)

  adosc = np.empty(n, dtype=np.float64)

  # EMA multipliers
  k_fast = 2.0 / (fast_period + 1)
  k_slow = 2.0 / (slow_period + 1)
  k1_fast = 1.0 - k_fast
  k1_slow = 1.0 - k_slow

  # First AD value and seed EMAs
  hl_range = high[0] - low[0]
  clv = (2.0 * close[0] - high[0] - low[0]) / hl_range if hl_range > EPSILON else 0.0
  ad_cumulative = clv * volume[0]
  ema_fast = ad_cumulative
  ema_slow = ad_cumulative
  adosc[0] = np.nan

  # Warmup phase: just update EMAs, output NaN
  for i in range(1, slow_period - 1):
    hl_range = high[i] - low[i]
    clv = (2.0 * close[i] - high[i] - low[i]) / hl_range if hl_range > EPSILON else 0.0
    ad_cumulative += clv * volume[i]
    ema_fast = ad_cumulative * k_fast + ema_fast * k1_fast
    ema_slow = ad_cumulative * k_slow + ema_slow * k1_slow
    adosc[i] = np.nan

  # Main loop: compute output (no branch)
  for i in range(slow_period - 1, n):
    hl_range = high[i] - low[i]
    clv = (2.0 * close[i] - high[i] - low[i]) / hl_range if hl_range > EPSILON else 0.0
    ad_cumulative += clv * volume[i]
    ema_fast = ad_cumulative * k_fast + ema_fast * k1_fast
    ema_slow = ad_cumulative * k_slow + ema_slow * k1_slow
    adosc[i] = ema_fast - ema_slow

  return adosc
