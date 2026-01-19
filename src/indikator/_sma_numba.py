"""Numba-optimized SMA (Simple Moving Average) calculation.

This module contains JIT-compiled functions for SMA calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(
  nopython=True, cache=True, nogil=True, fastmath=True, parallel=True
)  # pragma: no cover
def compute_sma_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled SMA calculation.

  SMA = (P1 + P2 + ... + Pn) / n

  Uses parallel chunked rolling sum for optimal performance on multicore.

  Args:
    prices: Array of prices (typically closing prices)
    period: Lookback period

  Returns:
    Array of SMA values (NaN for initial bars where period not satisfied)
  """
  n = len(prices)
  sma = np.full(n, np.nan)

  if n < period:
    return sma

  # Parallel execution using chunks
  # Numba handles prange by splitting the range into chunks for threads.
  # We need to compute the initial sum for each chunk independently.

  # For loop from period-1 (first valid index returns value) to n
  # But SMA at index i needs sum(p[i-period+1 : i+1])

  # To make it parallel, loop variable `i` is the Output Index.
  # Inside the loop, we could recompute sum for every point: O(N*period). Slow!
  # But we can do chunked rolling.

  # Numba parallel loop:
  # The compiler splits `prange` into chunks [start, end).
  # We can't easily instruct Numba to run a specific alg per chunk in `prange` body
  # unless logic is inside loop.

  # Actually, a simple trick for parallel rolling sum:
  # Split the array manually? No, Numba is cleaner.

  # BUT `prange` doesn't expose 'start of chunk'.
  # So we depend on Numba's scheduling.

  # Alternative: Naive parallel sum O(N*Period)?
  # If Period is small (e.g. 14, 30), O(N*Period) might be faster than O(N) sequential
  # if we have 16 cores (16x throughput).
  # 30 * (1/16) ~= 2. So naive loop is only 2x slower than optimal parallel?
  # But optimal parallel is O(N).

  # Wait, benchmark showed SMA ~0.7x TA-Lib speed.
  # Trying to beat C sequential with Python/JIT parallel.
  #
  # Let's try Parallel Naive Sum for small period?
  # If period is large, it degrades.

  # Better approach:
  # Just use pure NumPy cumulative sum parallelization logic?
  # `(cumsum[period:] - cumsum[:-period]) / period`.
  # Constructing cumsum is O(N) sequential (mostly).
  # NumPy cumsum is VERY fast.
  # Let's try the cumsum approach in Numba land or just return to pure NumPy?
  # If I rewrite `compute_sma_numba` to use `np.cumsum`, Numba might not optimize it well?
  # Or use `objmode`?
  # Actually, pure Python `compute_sma_numba` without `@jit` might be fastest if using `np.cumsum`.
  # But `compute_sma_numba` is decorated.

  # I will implement the Parallel Chunked approach manually.
  # We define a fixed chunk size (e.g. 4096 or n/num_threads).
  # But getting num_threads inside JIT is hard.

  # Let's try the simplest parallel form:
  # Re-summing at every step is safe but slow.

  # Optimization: Use `np.convolve` in Python mode?
  # TA-Lib is unbeatable for simple sliding window sum unless we use SIMD or threads.

  # Let's assume standard `prange` with re-summing IS the optimization for small `period`.
  # TRIX and others benefited from fusion.
  # SMA is solitary.

  # Actually, I'll stick to sequential rolling sum for SMA if period is large.
  # BUT for typical use (period < 100), maybe naive parallel is ok?
  # 1M rows. 10ms for optimal.
  # Parallel naive: 1M * 14 ops / 8 cores ~= 1.75M ops.
  # Sequential rolling: 1M * 2 ops = 2M ops.
  # So Naive Parallel IS FASTER than Sequential Rolling for period < 16 (assuming 8 cores).
  # For period=30, naive parallel is slower (30/8 = 3.75 vs 2).

  # Smart Chunking:
  # Divide n into K chunks.

  num_chunks = 16  # Heuristic for modern CPUs
  chunk_size = (n - period) // num_chunks

  # We can use `prange(num_chunks)` loop!

  # First `sma` array initialization
  inv_period = 1.0 / period

  for c in prange(num_chunks + 1):
    start = (period - 1) + c * chunk_size
    end = (period - 1) + (c + 1) * chunk_size
    if c == num_chunks:
      end = n

    if start >= n:
      continue

    # Process chunk [start, end)
    # Calculate initial sum for this chunk
    # We need sum of prices[start-period : start]

    # Determine rolling sum at `start-1`
    # Actually `sma[start]` needs `sum` ending at `start`.
    # So sum(prices[start-period+1 : start+1])? NO.
    # SMA[i] = mean(prices[i-period+1 : i+1])

    chunk_sum = 0.0
    # Initial sum for the FIRST element of the chunk (index `start`)
    # range(start - period + 1, start + 1)
    for k in range(start - period + 1, start + 1):
      chunk_sum += prices[k]

    sma[start] = chunk_sum * inv_period

    # Rolling loop for the rest of the chunk
    current_sum = chunk_sum
    for i in range(start + 1, end):
      current_sum = current_sum + prices[i] - prices[i - period]
      sma[i] = current_sum * inv_period

  return sma
