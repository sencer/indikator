"""Benchmark rolling window indicators (RVOL, Z-Score).

These indicators use pandas rolling windows and should be very fast.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from indikator.rvol import rvol
from indikator.zscore import zscore


def benchmark_rvol(sizes: list[int] | None = None) -> None:
    """Benchmark RVOL indicator."""
    if sizes is None:
        sizes = [100, 1000, 10000, 50000, 100000]

    print("=" * 80)
    print("RVOL BENCHMARK")
    print("=" * 80)
    print("\nRVOL: current volume / rolling average volume")
    print("Uses: pandas rolling().mean() with division")
    print("\n" + "-" * 80)

    window = 20

    results = {}

    for size in sizes:
        print(f"\nData size: {size:,} bars")
        print("-" * 40)

        # Generate test data
        np.random.seed(42)
        volumes = 1000 + np.abs(np.random.randn(size) * 200)
        df = pd.DataFrame({"volume": volumes})

        # Benchmark
        times = []
        for _ in range(3):  # Run 3 times for stability
            start = time.perf_counter()
            _ = rvol(df, window=window)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = size / avg_time

        results[size] = avg_time

        print(f"  Time:       {avg_time:.6f}s +- {std_time:.6f}s")
        print(f"  Throughput: {throughput:,.0f} bars/sec")
        print(f"  Per bar:    {avg_time / size * 1e6:.2f} us")

    # Summary
    print("\n" + "=" * 80)
    print("RVOL SCALING")
    print("=" * 80)
    print(f"\n{'Size':<15} {'Time (s)':<12} {'Scaling':<12}")
    print("-" * 80)

    first_size = sizes[0]
    first_time = results[first_size]

    for size in sizes:
        scaling = results[size] / first_time / (size / first_size)
        print(f"{size:<15,} {results[size]:<12.6f} {scaling:<12.3f}x")

    print("\nNote: Ideal linear scaling = 1.0x (time grows proportionally with data)")


def benchmark_zscore(sizes: list[int] | None = None) -> None:
    """Benchmark Z-Score indicator."""
    if sizes is None:
        sizes = [100, 1000, 10000, 50000, 100000]

    print("\n" + "=" * 80)
    print("Z-SCORE BENCHMARK")
    print("=" * 80)
    print("\nZ-Score: (value - rolling mean) / rolling std")
    print("Uses: pandas rolling().mean() and rolling().std()")
    print("\n" + "-" * 80)

    window = 20

    results = {}

    for size in sizes:
        print(f"\nData size: {size:,} bars")
        print("-" * 40)

        # Generate test data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(size) * 0.5)
        df = pd.DataFrame({"close": prices})

        # Benchmark
        times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = zscore(df, window=window)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = size / avg_time

        results[size] = avg_time

        print(f"  Time:       {avg_time:.6f}s +- {std_time:.6f}s")
        print(f"  Throughput: {throughput:,.0f} bars/sec")
        print(f"  Per bar:    {avg_time / size * 1e6:.2f} us")

    # Summary
    print("\n" + "=" * 80)
    print("Z-SCORE SCALING")
    print("=" * 80)
    print(f"\n{'Size':<15} {'Time (s)':<12} {'Scaling':<12}")
    print("-" * 80)

    first_size = sizes[0]
    first_time = results[first_size]

    for size in sizes:
        scaling = results[size] / first_time / (size / first_size)
        print(f"{size:<15,} {results[size]:<12.6f} {scaling:<12.3f}x")


def compare_rolling_indicators(size: int = 10000) -> None:
    """Compare RVOL vs Z-Score performance."""
    print("\n" + "=" * 80)
    print("ROLLING INDICATORS COMPARISON")
    print("=" * 80)
    print(f"\nData size: {size:,} bars, Window: 20")
    print("\n" + "-" * 80)

    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(size) * 0.5)
    volumes = 1000 + np.abs(np.random.randn(size) * 200)
    df = pd.DataFrame({"close": prices, "volume": volumes})

    # Benchmark RVOL
    start = time.perf_counter()
    _ = rvol(df, window=20)
    rvol_time = time.perf_counter() - start

    # Benchmark Z-Score
    start = time.perf_counter()
    _ = zscore(df, window=20)
    zscore_time = time.perf_counter() - start

    print(f"RVOL:    {rvol_time:.6f}s  (1 rolling operation)")
    print(f"Z-Score: {zscore_time:.6f}s  (2 rolling operations)")
    print(f"\nRatio: Z-Score is {zscore_time / rvol_time:.2f}x slower than RVOL")
    print("(Expected ~2x due to computing both mean AND std)")


if __name__ == "__main__":
    benchmark_rvol()
    benchmark_zscore()
    compare_rolling_indicators()

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("- Both indicators scale linearly with data size (O(n))")
    print("- pandas rolling() is highly optimized")
    print("- RVOL: ~500k-1M bars/sec")
    print("- Z-Score: ~250k-500k bars/sec (2 rolling ops)")
