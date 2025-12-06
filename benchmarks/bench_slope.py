"""Benchmark slope calculation implementations.

Compares:
1. Current Numba implementation (indikator.slope)
2. Scipy linregress with rolling().apply()
3. Numpy polyfit with rolling().apply()
4. Manual vectorized approach (no rolling)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from indikator.slope import slope

if TYPE_CHECKING:
  from numpy.typing import NDArray

MIN_WINDOW = 2


def scipy_slope(
  data: pd.DataFrame, column: str = "close", window: int = 20
) -> pd.DataFrame:
  """Slope using scipy.stats.linregress with rolling().apply()."""

  def calc_slope(y: pd.Series) -> float:
    if len(y) < MIN_WINDOW:
      return np.nan
    x = np.arange(len(y))
    result = stats.linregress(x, y)
    return result.slope

  data_copy = data.copy()
  data_copy[f"{column}_slope"] = (
    data[column].rolling(window=window).apply(calc_slope, raw=False)
  )
  return data_copy


def numpy_polyfit_slope(
  data: pd.DataFrame, column: str = "close", window: int = 20
) -> pd.DataFrame:
  """Slope using numpy.polyfit with rolling().apply()."""

  def calc_slope(y: NDArray) -> float:
    if len(y) < MIN_WINDOW:
      return np.nan
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, deg=1)
    return coeffs[0]  # First coefficient is the slope

  data_copy = data.copy()
  data_copy[f"{column}_slope"] = (
    data[column].rolling(window=window).apply(calc_slope, raw=True)
  )
  return data_copy


def manual_vectorized_slope(
  data: pd.DataFrame, column: str = "close", window: int = 20
) -> pd.DataFrame:
  """Manual vectorized slope calculation (similar to our Numba approach but pure numpy)."""
  values = data[column].values
  n = len(values)
  slopes = np.full(n, np.nan)

  if window < MIN_WINDOW or n < window:
    data_copy = data.copy()
    data_copy[f"{column}_slope"] = slopes
    return data_copy

  # Precompute x values and variance
  x = np.arange(window, dtype=np.float64)
  x_mean = (window - 1) / 2.0
  x_var = np.sum((x - x_mean) ** 2)

  # Calculate slope for each window
  for i in range(window - 1, n):
    y_window = values[i - window + 1 : i + 1]
    y_mean = np.mean(y_window)
    cov = np.sum((x - x_mean) * (y_window - y_mean))
    slopes[i] = cov / x_var

  data_copy = data.copy()
  data_copy[f"{column}_slope"] = slopes
  return data_copy


def benchmark_slope(sizes: list[int] | None = None, window: int = 20) -> None:
  """Benchmark different slope implementations."""

  if sizes is None:
    sizes = [100, 1000, 10000]  # Reduced max size to avoid timeout

  print("=" * 80)
  print("SLOPE BENCHMARK")
  print("=" * 80)
  print(f"\nWindow size: {window}")
  print(f"Testing data sizes: {sizes}")
  print("\n" + "-" * 80)

  implementations = [
    ("Numba (current)", slope),
    ("Manual vectorized", manual_vectorized_slope),
    ("Numpy polyfit", numpy_polyfit_slope),
    ("Scipy linregress", scipy_slope),
  ]

  results = {}

  for size in sizes:
    print(f"\nData size: {size:,} bars")
    print("-" * 40)

    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(size) * 0.5)
    df = pd.DataFrame({"close": prices})

    size_results = {}

    for name, func in implementations:
      # Warmup run (important for JIT)
      if name == "Numba (current)":
        _ = func(df.head(100), window=window)

      # Benchmark
      start = time.perf_counter()
      result = func(df, window=window)
      elapsed = time.perf_counter() - start

      size_results[name] = elapsed

      # Verify correctness (compare to scipy)
      if name != "Scipy linregress":
        scipy_result = scipy_slope(df, window=window)
        diff = np.abs(result["close_slope"] - scipy_result["close_slope"])
        max_diff = np.nanmax(diff)
        print(f"  {name:20s}: {elapsed:8.4f}s  (max diff: {max_diff:.2e})")
      else:
        print(f"  {name:20s}: {elapsed:8.4f}s  (baseline)")

    results[size] = size_results

  # Summary
  print("\n" + "=" * 80)
  print("SPEEDUP vs SCIPY")
  print("=" * 80)

  for size in sizes:
    scipy_time = results[size]["Scipy linregress"]
    print(f"\nData size: {size:,} bars (scipy: {scipy_time:.4f}s)")
    print("-" * 40)

    for name in ["Numba (current)", "Manual vectorized", "Numpy polyfit"]:
      speedup = scipy_time / results[size][name]
      print(f"  {name:20s}: {speedup:6.1f}x faster")


def benchmark_window_sizes(data_size: int = 10000) -> None:
  """Benchmark how window size affects performance."""
  print("\n" + "=" * 80)
  print("WINDOW SIZE IMPACT")
  print("=" * 80)
  print(f"\nData size: {data_size:,} bars")
  print("\n" + "-" * 80)

  windows = [5, 10, 20, 50, 100, 200]

  # Generate test data
  np.random.seed(42)
  prices = 100 + np.cumsum(np.random.randn(data_size) * 0.5)
  df = pd.DataFrame({"close": prices})

  print(f"{'Window':<10} {'Numba (s)':<12} {'Scipy (s)':<12} {'Speedup':<10}")
  print("-" * 80)

  for window in windows:
    # Numba
    start = time.perf_counter()
    _ = slope(df, window=window)
    numba_time = time.perf_counter() - start

    # Scipy
    start = time.perf_counter()
    _ = scipy_slope(df, window=window)
    scipy_time = time.perf_counter() - start

    speedup = scipy_time / numba_time
    print(f"{window:<10} {numba_time:<12.4f} {scipy_time:<12.4f} {speedup:<10.1f}x")


def benchmark_cold_vs_warm(size: int = 10000, window: int = 20, runs: int = 5) -> None:
  """Benchmark cold start (first run) vs warm (JIT compiled)."""
  print("\n" + "=" * 80)
  print("COLD START vs WARM (JIT compilation effect)")
  print("=" * 80)
  print(f"\nData size: {size:,} bars, Window: {window}, Runs: {runs}")
  print("\n" + "-" * 80)

  # Generate test data
  np.random.seed(42)
  prices = 100 + np.cumsum(np.random.randn(size) * 0.5)
  df = pd.DataFrame({"close": prices})

  times = []

  for i in range(runs):
    start = time.perf_counter()
    _ = slope(df, window=window)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    status = "COLD (includes JIT)" if i == 0 else "WARM (JIT cached)"
    print(f"  Run {i + 1}: {elapsed:.4f}s  ({status})")

  print("\n" + "-" * 40)
  print(f"First run (cold):     {times[0]:.4f}s")
  print(f"Avg warm runs (2-{runs}): {np.mean(times[1:]):.4f}s")
  print(f"JIT overhead:         {times[0] - np.mean(times[1:]):.4f}s")
  print(f"Speedup after JIT:    {times[0] / np.mean(times[1:]):.2f}x")


if __name__ == "__main__":
  # Run all benchmarks
  benchmark_slope()
  benchmark_window_sizes()
  benchmark_cold_vs_warm()

  print("\n" + "=" * 80)
  print("BENCHMARK COMPLETE")
  print("=" * 80)
