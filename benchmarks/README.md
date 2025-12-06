# Indikator Performance Benchmarks

Comprehensive performance benchmarks for the indikator library.

## Quick Start

```bash
# Run all benchmarks
uv run python benchmarks/run_all.py

# Or run individual benchmarks
uv run python benchmarks/bench_slope.py
uv run python benchmarks/bench_rolling_indicators.py
uv run python benchmarks/bench_intraday_indicators.py
uv run python benchmarks/bench_optimizations.py
```

## Benchmark Scripts

### `bench_slope.py`

Validates the "50-100x faster than scipy" claim for the slope indicator.

**Compares:**
- Current Numba implementation
- Scipy linregress with rolling().apply()
- Numpy polyfit with rolling().apply()
- Manual vectorized approach

**Key findings:**
- **1,830x faster** at 1,000 bars
- **7,699x faster** at 10,000 bars
- **8,545x faster** at 50,000 bars
- Original claim of "50-100x" was very conservative

### `bench_rolling_indicators.py`

Benchmarks rolling window indicators (RVOL, Z-Score).

**Tests:**
- RVOL performance across data sizes
- Z-Score performance across data sizes
- Scaling characteristics
- Throughput measurements

**Key findings:**
- RVOL: **9-12 million bars/sec**
- Z-Score: **9-10 million bars/sec**
- Both scale sub-linearly (better than O(n))
- Pandas rolling() is extremely well optimized

### `bench_intraday_indicators.py`

Benchmarks time-of-day indicators (RVOL intraday, Z-Score intraday).

**Tests:**
- RVOL intraday scaling
- Z-Score intraday scaling
- Core intraday_aggregate function
- Different aggregation functions

**Key findings:**
- Current: **1,500-2,500 bars/sec**
- Near-linear scaling for fixed time slots
- Z-Score intraday ~2x slower than RVOL (2 aggregations)
- Performance acceptable for typical use cases

### `bench_optimizations.py`

Explores optimization opportunities for intraday indicators.

**Tests:**
- Groupby-based optimization
- Correctness verification
- Complexity analysis
- Scaling comparison

**Key findings:**
- **7-12x speedup** possible with groupby optimization
- Reduces complexity from O(n²) to O(n)
- Mathematically equivalent (verified)
- Optimized: **11,000-13,000 bars/sec**

### `run_all.py`

Runs all benchmarks and generates comprehensive summary.

## Performance Summary

| Indicator | Throughput | vs Baseline | Notes |
|-----------|-----------|-------------|-------|
| Slope (Numba) | ~14k bars in 0.7ms | **8,545x faster** than scipy | Numba JIT optimized |
| RVOL | **9-12M bars/sec** | N/A | Pandas rolling optimized |
| Z-Score | **9-10M bars/sec** | N/A | Pandas rolling optimized |
| RVOL intraday | 1,500-2,500 bars/sec | N/A | Loop-based |
| RVOL intraday (opt) | 11,000-13,000 bars/sec | **7-12x faster** | Groupby optimization |
| Z-Score intraday | 750-1,250 bars/sec | ~2x slower | 2 aggregations |

## Optimization Recommendations

### Already Optimal
- ✓ Slope indicator (Numba JIT)
- ✓ RVOL (pandas rolling)
- ✓ Z-Score (pandas rolling)

### Potential Optimizations
- **Intraday indicators**: 7-12x speedup available
  - Trade-off: code complexity vs performance
  - Recommended for large datasets (100+ days)
  - Implementation ready in `bench_optimizations.py`

## Dependencies

Benchmarks require:
- `scipy` (for comparison baselines)
- All standard indikator dependencies

Install with:
```bash
uv pip install scipy
```

## Interpreting Results

### Throughput
- **Million bars/sec**: Extremely fast, negligible overhead
- **Thousands bars/sec**: Fast, suitable for real-time
- **Hundreds bars/sec**: Acceptable for batch processing

### Scaling
- **Sub-linear (<1.0x)**: Better than expected (caching effects)
- **Linear (1.0x)**: Ideal scaling
- **Super-linear (>1.0x)**: Performance degrades with size

### Speedup
- **<2x**: Marginal improvement
- **2-10x**: Significant improvement
- **>10x**: Dramatic improvement (worth documenting)

## Adding New Benchmarks

Template:
```python
def benchmark_my_indicator(sizes: list[int] = [100, 1000, 10000]) -> None:
    """Benchmark my_indicator."""
    for size in sizes:
        # Generate test data
        df = generate_test_data(size)

        # Benchmark
        start = time.perf_counter()
        result = my_indicator(df)
        elapsed = time.perf_counter() - start

        # Report
        throughput = size / elapsed
        print(f"Size: {size:,}  Time: {elapsed:.4f}s  Throughput: {throughput:,.0f} bars/sec")
```

## CI Integration

To run in CI:
```yaml
- name: Run benchmarks
  run: uv run python benchmarks/run_all.py
  timeout-minutes: 5
```

## Historical Results

Results may vary based on:
- CPU model and speed
- Available memory
- System load
- Python version
- NumPy/Pandas versions

Benchmark on your target hardware for accurate estimates.
