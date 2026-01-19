"""Robust benchmark suite comparing Indikator vs TA-lib.

Methodology for stable measurements:
1. Interleaved runs - alternate between implementations to cancel cache bias
2. Randomized order - shuffle run order to avoid systematic bias
3. Multiple rounds - repeat many times and use robust statistics
4. Warmup isolation - separate warmup phase before timing
5. GC disabled during timing - avoid GC pauses
"""

from collections.abc import Callable
import gc
import random
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False
  print("WARNING: TA-Lib not found. Skipping comparison benchmarks.")

from datawarden import config

print("Importing Indikator modules (validation disabled)...")
with config.Overrides(skip_validation=True):
  from indikator import (
    adx,
    aroon,
    atr,
    bollinger_bands,
    cci,
    churn_factor,
    cmo,
    ema,
    legs,
    macd,
    mfi,
    obv,
    roc,
    rsi,
    rvol,
    sector_correlation,
    slope,
    sma,
    stoch,
    trix,
    vwap,
    willr,
    zscore,
  )


def generate_data(size: int) -> dict[str, Any]:
  """Generate synthetic market data."""
  np.random.seed(42)
  returns = np.random.randn(size) * 0.001
  price = 100 * np.exp(np.cumsum(returns))
  high = price * (1 + np.abs(np.random.randn(size) * 0.002))
  low = price * (1 - np.abs(np.random.randn(size) * 0.002))
  close = price
  high = np.maximum(high, close)
  low = np.minimum(low, close)
  volume = np.abs(np.random.randn(size) * 1000 + 10000)
  dates = pd.date_range("2020-01-01", periods=size, freq="1min")

  return {
    "high": pd.Series(high, index=dates),
    "low": pd.Series(low, index=dates),
    "close": pd.Series(close, index=dates),
    "volume": pd.Series(volume, index=dates),
    "open": pd.Series(price, index=dates),
    "sector": pd.Series(close, index=dates) * (1 + np.random.randn(size) * 0.01),
    "np_high": high.astype(np.float64),
    "np_low": low.astype(np.float64),
    "np_close": close.astype(np.float64),
    "np_volume": volume.astype(np.float64),
  }


def interleaved_benchmark(
  fn_a: Callable[[], Any],
  fn_b: Callable[[], Any] | None,
  n_rounds: int = 50,
  warmup: int = 10,
) -> tuple[float, float | None]:
  """Benchmark two functions with interleaved runs for fair comparison.

  Runs A and B alternately in randomized order to cancel CPU/cache bias.
  Returns P25 times in milliseconds.
  """
  # Warmup phase (isolated)
  for _ in range(warmup):
    fn_a()
    if fn_b:
      fn_b()

  # Create interleaved schedule
  # Each round: one A, one B (if exists), in random order
  times_a: list[float] = []
  times_b: list[float] = []

  for _ in range(n_rounds):
    # Randomize which runs first this round
    order = [("a", fn_a)]
    if fn_b:
      order.append(("b", fn_b))
    random.shuffle(order)

    for label, fn in order:
      gc.disable()
      start = time.perf_counter()
      fn()
      elapsed = time.perf_counter() - start
      gc.enable()

      if label == "a":
        times_a.append(elapsed)
      else:
        times_b.append(elapsed)

  # Use P25 for robustness (less affected by outliers than median)
  t_a = sorted(times_a)[len(times_a) // 4] * 1000
  t_b = sorted(times_b)[len(times_b) // 4] * 1000 if times_b else None

  return t_a, t_b


def run_benchmarks() -> None:
  """Run benchmark suite at multiple data sizes."""
  sizes = [10_000, 100_000, 1_000_000]
  size_names = ["10K", "100K", "1M"]

  benchmarks: list[
    tuple[
      str,
      Callable[..., Any],
      Callable[[dict], tuple],
      Any,
      Callable[[dict], tuple] | None,
    ]
  ] = [
    ("SMA", sma, lambda d: (d["close"], 30), talib.SMA, lambda d: (d["np_close"], 30)),
    ("EMA", ema, lambda d: (d["close"], 30), talib.EMA, lambda d: (d["np_close"], 30)),
    ("RSI", rsi, lambda d: (d["close"], 14), talib.RSI, lambda d: (d["np_close"], 14)),
    ("ROC", roc, lambda d: (d["close"], 10), talib.ROC, lambda d: (d["np_close"], 10)),
    ("CMO", cmo, lambda d: (d["close"], 14), talib.CMO, lambda d: (d["np_close"], 14)),
    (
      "Slope",
      slope,
      lambda d: (d["close"], 14),
      talib.LINEARREG_SLOPE,
      lambda d: (d["np_close"], 14),
    ),
    (
      "TRIX",
      trix,
      lambda d: (d["close"], 30),
      talib.TRIX,
      lambda d: (d["np_close"], 30),
    ),
    (
      "Bollinger",
      bollinger_bands,
      lambda d: (d["close"], 20, 2.0),
      talib.BBANDS,
      lambda d: (d["np_close"], 20, 2.0, 2.0),
    ),
    (
      "MACD",
      macd,
      lambda d: (d["close"], 12, 26, 9),
      talib.MACD,
      lambda d: (d["np_close"], 12, 26, 9),
    ),
    (
      "OBV",
      obv,
      lambda d: (d["close"], d["volume"]),
      talib.OBV,
      lambda d: (d["np_close"], d["np_volume"]),
    ),
    (
      "ATR",
      atr,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.ATR,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
    ),
    (
      "ADX",
      adx,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.ADX,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
    ),
    (
      "CCI",
      cci,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.CCI,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
    ),
    (
      "WillR",
      willr,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.WILLR,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
    ),
    (
      "Stoch",
      stoch,
      lambda d: (d["high"], d["low"], d["close"], 14, 3, 3),
      talib.STOCH,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14, 3, 0, 3, 0),
    ),
    (
      "AROON",
      aroon,
      lambda d: (d["high"], d["low"], 25),
      talib.AROON,
      lambda d: (d["np_high"], d["np_low"], 25),
    ),
    (
      "MFI",
      mfi,
      lambda d: (d["high"], d["low"], d["close"], d["volume"], 14),
      talib.MFI,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], d["np_volume"], 14),
    ),
    # Indikator-only
    ("Churn", churn_factor, lambda d: (d["high"], d["low"], d["volume"]), None, None),
    ("Legs", legs, lambda d: (d["high"], d["low"], d["close"], 0.05), None, None),
    ("RVOL", rvol, lambda d: (d["volume"], 20), None, None),
    ("Z-Score", zscore, lambda d: (d["close"], 20), None, None),
    (
      "VWAP",
      vwap,
      lambda d: (d["high"], d["low"], d["close"], d["volume"]),
      None,
      None,
    ),
    ("SectorCorr", sector_correlation, lambda d: (d["close"], d["sector"]), None, None),
  ]

  # Header
  print("\n" + "=" * 95)
  header = f"{'Indicator':<12}"
  for name in size_names:
    header += f" | {name:>8} {'Ratio':>7}"
  print(header)
  print("-" * 95)

  for name, ind_func, ind_args_fn, ta_func, ta_args_fn in benchmarks:
    row = f"{name:<12}"

    for size_idx, size in enumerate(sizes):
      data = generate_data(size)
      ind_args = ind_args_fn(data)

      # Number of rounds scales inversely with data size
      n_rounds = 100 if size <= 100_000 else 30

      # Create wrapped callables
      def make_ind_fn(args=ind_args):
        def fn():
          with config.Overrides(skip_validation=True):
            return ind_func(*args)

        return fn

      ind_fn = make_ind_fn()

      ta_fn = None
      if HAS_TALIB and ta_func is not None and ta_args_fn is not None:
        ta_args = ta_args_fn(data)

        def make_ta_fn(func=ta_func, args=ta_args):
          return lambda: func(*args)

        ta_fn = make_ta_fn()

      try:
        t_ind, t_ta = interleaved_benchmark(ind_fn, ta_fn, n_rounds=n_rounds)

        if t_ta is not None:
          ratio = t_ind / t_ta
          row += f" | {t_ind:>8.2f} {ratio:>6.2f}x"
        else:
          row += f" | {t_ind:>8.2f} {'-':>7}"
      except Exception as e:
        row += f" | {'FAIL':>8} {'-':>7}"
        print(f"  Error in {name}: {e}")

    print(row)

  print("-" * 95)
  print("Methodology: Interleaved runs with randomized order, P25 timing")
  print("Ratio = Indikator/TA-lib (>1 = slower, <1 = faster)")


if __name__ == "__main__":
  run_benchmarks()
