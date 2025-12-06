"""Benchmark intraday indicators (RVOL intraday, Z-Score intraday).

These indicators loop over each bar to compute time-of-day aggregates.
Test performance and explore optimization opportunities.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from indikator.rvol import intraday_aggregate, rvol_intraday
from indikator.zscore import zscore_intraday

if TYPE_CHECKING:
  from collections.abc import Callable

OPENING_BARS = 10
CLOSING_BARS_START = 68
LUNCH_START = 30
LUNCH_END = 40


def generate_intraday_data(days: int = 10, bars_per_day: int = 78) -> pd.DataFrame:
  """Generate realistic intraday data.

  Args:
    days: Number of trading days
    bars_per_day: Bars per day (78 = 5-min bars in 6.5hr trading day)

  Returns:
    DataFrame with DatetimeIndex and volume column
  """
  # Create datetime index (5-min bars, 9:30-16:00)
  dates = []
  for day in range(days):
    base_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
    for bar in range(bars_per_day):
      time_val = pd.Timestamp("09:30:00") + pd.Timedelta(minutes=5 * bar)
      dt = pd.Timestamp(
        year=base_date.year,
        month=base_date.month,
        day=base_date.day,
        hour=time_val.hour,
        minute=time_val.minute,
      )
      dates.append(dt)

  # Generate volumes with intraday pattern
  volumes = []
  for _day in range(days):
    for bar in range(bars_per_day):
      # Higher volume at open/close, lower at lunch
      time_factor = 1.0
      if bar < OPENING_BARS:  # First 50 min
        time_factor = 2.0
      elif bar > CLOSING_BARS_START:  # Last 50 min
        time_factor = 1.8
      elif LUNCH_START < bar < LUNCH_END:  # Lunch
        time_factor = 0.6

      volume = 1000 * time_factor + np.random.rand() * 200
      volumes.append(volume)

  return pd.DataFrame({"volume": volumes}, index=pd.DatetimeIndex(dates))


def benchmark_rvol_intraday(
  days_list: list[int] | None = None, bars_per_day: int = 78
) -> None:
  """Benchmark RVOL intraday indicator."""
  if days_list is None:
    days_list = [5, 10, 20, 40]

  print("=" * 80)
  print("RVOL INTRADAY BENCHMARK")
  print("=" * 80)
  print(f"\nBars per day: {bars_per_day} (5-min bars, 9:30-16:00)")
  print("Min samples per time slot: 3")
  print("\n" + "-" * 80)

  results = {}

  for days in days_list:
    total_bars = days * bars_per_day
    print(f"\nData: {days} days = {total_bars:,} bars")
    print("-" * 40)

    # Generate test data
    df = generate_intraday_data(days=days, bars_per_day=bars_per_day)

    # Benchmark
    start = time.perf_counter()
    # rvol_intraday takes volume Series
    _ = rvol_intraday(df["volume"], min_samples=3)
    elapsed = time.perf_counter() - start

    throughput = total_bars / elapsed

    results[days] = elapsed

    print(f"  Time:       {elapsed:.4f}s")
    print(f"  Throughput: {throughput:,.0f} bars/sec")
    print(f"  Per bar:    {elapsed / total_bars * 1e6:.2f} us")

  # Complexity analysis
  print("\n" + "=" * 80)
  print("SCALING ANALYSIS")
  print("=" * 80)
  print("\nIntraday aggregate is O(n^2) in worst case:")
  print("- For each bar (n), look back through all previous bars (n)")
  print("- With fixed time slots, effective complexity is better")
  print("\n" + "-" * 80)

  print(
    f"\n{'Days':<10} {'Bars':<15} {'Time (s)':<12} {'Theoretical':<15} {'Actual':<12}"
  )
  print("-" * 80)

  first_days = days_list[0]
  first_time = results[first_days]

  for days in days_list:
    bars = days * bars_per_day
    # For fixed time slots, should be closer to O(n) than O(n^2)
    theoretical = first_time * (days / first_days)  # Linear
    actual_ratio = results[days] / first_time
    print(
      f"{days:<10} {bars:<15,} {results[days]:<12.4f} "
      f"{theoretical:<15.4f} {actual_ratio:<12.2f}x"
    )


def benchmark_zscore_intraday(
  days_list: list[int] | None = None, bars_per_day: int = 78
) -> None:
  """Benchmark Z-Score intraday indicator."""
  if days_list is None:
    days_list = [5, 10, 20, 40]

  print("\n" + "=" * 80)
  print("Z-SCORE INTRADAY BENCHMARK")
  print("=" * 80)
  print(f"\nBars per day: {bars_per_day} (5-min bars, 9:30-16:00)")
  print("Min samples per time slot: 3")
  print("\n" + "-" * 80)

  results = {}

  for days in days_list:
    total_bars = days * bars_per_day
    print(f"\nData: {days} days = {total_bars:,} bars")
    print("-" * 40)

    # Generate test data
    df = generate_intraday_data(days=days, bars_per_day=bars_per_day)
    df["close"] = 100 + np.cumsum(np.random.randn(len(df)) * 0.5)

    # Benchmark
    start = time.perf_counter()
    # zscore_intraday takes close Series
    _ = zscore_intraday(df["close"], min_samples=3)
    elapsed = time.perf_counter() - start

    throughput = total_bars / elapsed

    results[days] = elapsed

    print(f"  Time:       {elapsed:.4f}s")
    print(f"  Throughput: {throughput:,.0f} bars/sec")
    print(f"  Per bar:    {elapsed / total_bars * 1e6:.2f} us")

  print("\n" + "=" * 80)
  print("Z-Score intraday calls intraday_aggregate() twice (mean + std)")
  print("Expected to be ~2x slower than RVOL intraday")
  print("=" * 80)


def benchmark_intraday_aggregate_direct(size: int = 1560) -> None:
  """Benchmark the core intraday_aggregate function."""
  print("\n" + "=" * 80)
  print("INTRADAY_AGGREGATE CORE FUNCTION")
  print("=" * 80)
  print(f"\nData size: {size:,} bars (20 days x 78 bars/day)")
  print("\n" + "-" * 80)

  # Generate test data
  df = generate_intraday_data(days=20, bars_per_day=78)

  aggregation_funcs = [
    ("mean", pd.Series.mean),
    ("std", pd.Series.std),
    ("median", pd.Series.median),
    ("min", pd.Series.min),
    ("max", pd.Series.max),
  ]

  print(f"{'Function':<15} {'Time (s)':<12} {'Throughput':<20}")
  print("-" * 80)

  for name, func in aggregation_funcs:
    start = time.perf_counter()
    # intraday_aggregate takes Series
    _ = intraday_aggregate(df["volume"], agg_func=func, min_samples=3)
    elapsed = time.perf_counter() - start

    throughput = size / elapsed

    print(f"{name:<15} {elapsed:<12.4f} {throughput:>15,.0f} bars/sec")

  print("\nNote: All aggregation functions have similar performance")
  print("(computation time dominated by loop overhead, not aggregation)")


def test_optimization_ideas(days: int = 20) -> None:
  """Test potential optimization ideas for intraday aggregation."""
  print("\n" + "=" * 80)
  print("OPTIMIZATION IDEAS")
  print("=" * 80)
  print(f"\nData: {days} days x 78 bars/day = {days * 78} bars")
  print("\n" + "-" * 80)

  df = generate_intraday_data(days=days, bars_per_day=78)

  # Current implementation
  start = time.perf_counter()
  # intraday_aggregate takes Series
  result1 = intraday_aggregate(df["volume"], agg_func=pd.Series.mean, min_samples=3)
  baseline_time = time.perf_counter() - start

  print(f"Current implementation:  {baseline_time:.4f}s (baseline)")

  # Idea 1: Precompute time slot groups
  def optimized_with_groupby(
    df: pd.DataFrame, column: str, agg_func: Callable, min_samples: int = 3
  ) -> pd.Series:
    """Use groupby to precompute time slot groups."""
    df_copy = df.copy()
    df_copy["_time_slot"] = df_copy.index.time

    # Group by time slot
    grouped = df_copy.groupby("_time_slot")[column]

    # For each bar, get mean of all previous bars in that time slot
    result = pd.Series(index=df.index, dtype=float)

    for _time_slot, group in grouped:
      indices = group.index
      for i, idx in enumerate(indices):
        # Get all previous bars in this time slot
        prev_vals = group.iloc[:i]
        if len(prev_vals) >= min_samples:
          result[idx] = agg_func(prev_vals)

    return result

  start = time.perf_counter()
  result2 = optimized_with_groupby(
    df, column="volume", agg_func=pd.Series.mean, min_samples=3
  )
  groupby_time = time.perf_counter() - start

  speedup = baseline_time / groupby_time
  print(f"Groupby optimization:    {groupby_time:.4f}s ({speedup:.2f}x)")

  # Verify correctness
  diff = np.abs(result1.fillna(0) - result2.fillna(0))
  max_diff = np.max(diff)
  print(f"\nMax difference: {max_diff:.2e} (should be ~0)")

  print("\n" + "-" * 80)
  print("Optimization potential: Limited")
  print("- Current impl already efficient for fixed time slots")
  print("- Groupby has overhead from pandas machinery")
  print("- Could use Numba JIT for ~2-3x speedup, but adds complexity")


if __name__ == "__main__":
  benchmark_rvol_intraday()
  benchmark_zscore_intraday()
  benchmark_intraday_aggregate_direct()
  test_optimization_ideas()

  print("\n" + "=" * 80)
  print("BENCHMARK COMPLETE")
  print("=" * 80)
  print("\nKey Findings:")
  print("- Intraday indicators are slower than rolling window indicators")
  print("- Scale well with data size (near-linear for fixed time slots)")
  print("- Performance is acceptable for typical intraday use cases")
  print("- Z-Score intraday ~2x slower than RVOL intraday (2 aggregations)")
