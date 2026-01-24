"""Robust benchmark suite comparing Indikator vs TA-lib.

Methodology for stable measurements:
1. Interleaved runs - alternate between implementations to cancel cache bias
2. Randomized order - shuffle run order to avoid systematic bias
3. Multiple rounds - repeat many times and use robust statistics
4. Warmup isolation - separate warmup phase before timing
5. GC disabled during timing - avoid GC pauses
"""

import argparse
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
    ad,
    adosc,
    adx,
    adxr,
    apo,
    aroon,
    aroonosc,
    atr,
    avgprice,
    beta,
    beta_statistical,
    bollinger_bands,
    bop,
    cci,
    cdl_2crows,
    cdl_3black_crows,
    cdl_3inside,
    cdl_3line_strike,
    cdl_3outside,
    cdl_3stars_in_south,
    cdl_3white_soldiers,
    cdl_abandoned_baby,
    cdl_advance_block,
    cdl_belt_hold,
    cdl_breakaway,
    cdl_closing_marubozu,
    cdl_conceal_baby_swallow,
    cdl_counterattack,
    cdl_dark_cloud_cover,
    cdl_doji,
    cdl_doji_star,
    cdl_dragonfly_doji,
    cdl_engulfing,
    cdl_evening_doji_star,
    cdl_evening_star,
    cdl_gap_side_by_side_white,
    cdl_gravestone_doji,
    cdl_hammer,
    cdl_hanging_man,
    cdl_harami,
    cdl_harami_cross,
    cdl_high_wave,
    cdl_hikkake,
    cdl_hikkake_mod,
    cdl_homing_pigeon,
    cdl_identical_3crows,
    cdl_in_neck,
    cdl_inverted_hammer,
    cdl_kicking,
    cdl_kicking_by_length,
    cdl_ladder_bottom,
    cdl_long_legged_doji,
    cdl_long_line,
    cdl_marubozu,
    cdl_mat_hold,
    cdl_matching_low,
    cdl_morning_doji_star,
    cdl_morning_star,
    cdl_on_neck,
    cdl_piercing,
    cdl_rickshaw_man,
    cdl_rise_fall_3methods,
    cdl_separating_lines,
    cdl_xsidegap3methods,
    cdl_shooting_star,
    cdl_short_line,
    cdl_spinning_top,
    cdl_stalled_pattern,
    cdl_stick_sandwich,
    cdl_takuri,
    cdl_tasuki_gap,
    cdl_thrusting,
    cdl_tristar,
    cdl_unique_3river,
    cdl_upside_gap_two_crows,
    churn_factor,
    cmo,
    correl,
    dema,
    dx,
    ema,
    ht_dcperiod,
    ht_dcphase,
    ht_phasor,
    ht_sine,
    ht_trendline,
    ht_trendmode,
    kama,
    ma,
    mama,
    legs,
    linearreg,
    linearreg_angle,
    linearreg_intercept,
    macd,
    macdext,
    macdfix,
    mavp,
    medprice,
    mfi,
    midpoint,
    midprice,
    minus_di,
    minus_dm,
    mom,
    natr,
    obv,
    plus_di,
    plus_dm,
    ppo,
    roc,
    rocp,
    rocr,
    rocr100,
    rsi,
    rvol,
    sar,
    sarext,
    sector_correlation,
    slope,
    sma,
    stddev,
    stoch,
    stochf,
    stochrsi,
    t3,
    tema,
    trange,
    trima,
    trix,
    tsf,
    typprice,
    ultosc,
    var,
    vwap,
    wclprice,
    willr,
    wma,
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
    "np_high": high.astype(np.float64),
    "np_low": low.astype(np.float64),
    "np_close": close.astype(np.float64),
    "np_open": price.astype(np.float64),
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
  parser = argparse.ArgumentParser(description="Run Indikator benchmarks")
  parser.add_argument(
    "--indicators",
    nargs="+",
    help="List of indicators to benchmark (case-insensitive)",
  )
  args = parser.parse_args()

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
    # fmt: off
    (
      "HT_DCPERIOD",
      ht_dcperiod,
      lambda d: (d["close"],),
      talib.HT_DCPERIOD,
      lambda d: (d["np_close"],),
    ),
    (
      "HT_DCPHASE",
      ht_dcphase,
      lambda d: (d["close"],),
      talib.HT_DCPHASE,
      lambda d: (d["np_close"],),
    ),
    (
      "HT_PHASOR",
      ht_phasor,
      lambda d: (d["close"],),
      talib.HT_PHASOR,
      lambda d: (d["np_close"],),
    ),
    (
      "HT_SINE",
      ht_sine,
      lambda d: (d["close"],),
      talib.HT_SINE,
      lambda d: (d["np_close"],),
    ),
    (
      "HT_TRENDMODE",
      ht_trendmode,
      lambda d: (d["close"],),
      talib.HT_TRENDMODE,
      lambda d: (d["np_close"],),
    ),
    (
      "MAVP",
      mavp,
      lambda d: (d["close"], pd.Series(np.full(len(d["close"]), 14, dtype=float))),
      talib.MAVP,
      lambda d: (d["np_close"], np.full(len(d["np_close"]), 14, dtype=float)),
    ),
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
    # --- Tier 1 Indicators ---
    (
      "DEMA",
      dema,
      lambda d: (d["close"], 30),
      talib.DEMA,
      lambda d: (d["np_close"], 30),
    ),
    (
      "TEMA",
      tema,
      lambda d: (d["close"], 30),
      talib.TEMA,
      lambda d: (d["np_close"], 30),
    ),
    ("WMA", wma, lambda d: (d["close"], 30), talib.WMA, lambda d: (d["np_close"], 30)),
    (
      "KAMA",
      kama,
      lambda d: (d["close"], 10),
      talib.KAMA,
      lambda d: (d["np_close"], 10),
    ),
    (
      "SAR",
      sar,
      lambda d: (d["high"], d["low"], 0.02, 0.2),
      talib.SAR,
      lambda d: (d["np_high"], d["np_low"], 0.02, 0.2),
    ),
    ("MOM", mom, lambda d: (d["close"], 10), talib.MOM, lambda d: (d["np_close"], 10)),
    (
      "NATR",
      natr,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.NATR,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
    ),
    (
      "ULTOSC",
      ultosc,
      lambda d: (d["high"], d["low"], d["close"], 7, 14, 28),
      talib.ULTOSC,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 7, 14, 28),
    ),
    (
      "AD",
      ad,
      lambda d: (d["high"], d["low"], d["close"], d["volume"]),
      talib.AD,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], d["np_volume"]),
    ),
    (
      "ADOSC",
      adosc,
      lambda d: (d["high"], d["low"], d["close"], d["volume"], 3, 10),
      talib.ADOSC,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], d["np_volume"], 3, 10),
    ),
    (
      "StochRSI",
      stochrsi,
      lambda d: (d["close"], 14, 14, 14, 3),
      talib.STOCHRSI,
      lambda d: (d["np_close"], 14, 14, 3, 0),
    ),
    # --- Tier 2 Indicators ---
    (
      "TRANGE",
      trange,
      lambda d: (d["high"], d["low"], d["close"]),
      talib.TRANGE,
      lambda d: (d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "PLUS_DM",
      plus_dm,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.PLUS_DM,
      lambda d: (d["np_high"], d["np_low"], 14),
    ),
    (
      "MINUS_DM",
      minus_dm,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.MINUS_DM,
      lambda d: (d["np_high"], d["np_low"], 14),
    ),
    (
      "PLUS_DI",
      plus_di,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.PLUS_DI,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
    ),
    (
      "MINUS_DI",
      minus_di,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.MINUS_DI,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
    ),
    (
      "DX",
      dx,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.DX,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
    ),
    (
      "ADXR",
      adxr,
      lambda d: (d["high"], d["low"], d["close"], 14),
      talib.ADXR,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
    ),
    (
      "STOCHF",
      stochf,
      lambda d: (d["high"], d["low"], d["close"], 5, 3),
      talib.STOCHF,
      lambda d: (d["np_high"], d["np_low"], d["np_close"], 5, 3, 0),
    ),
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
    # --- New / Standardized ---
    (
      "BOP",
      bop,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.BOP,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "ROCP",
      rocp,
      lambda d: (d["close"], 10),
      talib.ROCP,
      lambda d: (d["np_close"], 10),
    ),
    (
      "ROCR",
      rocr,
      lambda d: (d["close"], 10),
      talib.ROCR,
      lambda d: (d["np_close"], 10),
    ),
    (
      "ROCR100",
      rocr100,
      lambda d: (d["close"], 10),
      talib.ROCR100,
      lambda d: (d["np_close"], 10),
    ),
    (
      "TRIMA",
      trima,
      lambda d: (d["close"], 30),
      talib.TRIMA,
      lambda d: (d["np_close"], 30),
    ),
    (
      "T3",
      t3,
      lambda d: (d["close"], 5, 0.7),
      talib.T3,
      lambda d: (d["np_close"], 5, 0.7),
    ),
    (
      "APO",
      apo,
      lambda d: (d["close"], 12, 26, 0),
      talib.APO,
      lambda d: (d["np_close"], 12, 26, 0),
    ),
    (
      "PPO",
      ppo,
      lambda d: (d["close"], 12, 26, 0),
      talib.PPO,
      lambda d: (d["np_close"], 12, 26, 0),
    ),
    # --- Price transforms ---
    (
      "TYPPRICE",
      typprice,
      lambda d: (d["high"], d["low"], d["close"]),
      talib.TYPPRICE,
      lambda d: (d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "MEDPRICE",
      medprice,
      lambda d: (d["high"], d["low"]),
      talib.MEDPRICE,
      lambda d: (d["np_high"], d["np_low"]),
    ),
    (
      "WCLPRICE",
      wclprice,
      lambda d: (d["high"], d["low"], d["close"]),
      talib.WCLPRICE,
      lambda d: (d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "AVGPRICE",
      avgprice,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.AVGPRICE,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "MIDPRICE",
      midprice,
      lambda d: (d["high"], d["low"], 14),
      talib.MIDPRICE,
      lambda d: (d["np_high"], d["np_low"], 14),
    ),
    (
      "MIDPOINT",
      midpoint,
      lambda d: (d["close"], 14),
      talib.MIDPOINT,
      lambda d: (d["np_close"], 14),
    ),
    # --- Statistical ---
    (
      "STDDEV",
      stddev,
      lambda d: (d["close"], 5, 1.0),
      talib.STDDEV,
      lambda d: (d["np_close"], 5, 1.0),
    ),
    (
      "VAR",
      var,
      lambda d: (d["close"], 5, 1.0),
      talib.VAR,
      lambda d: (d["np_close"], 5, 1.0),
    ),
    # --- Oscillators ---
    (
      "AROONOSC",
      aroonosc,
      lambda d: (d["high"], d["low"], 25),
      talib.AROONOSC,
      lambda d: (d["np_high"], d["np_low"], 25),
    ),
    (
      "LINEARREG",
      linearreg,
      lambda d: (d["close"], 14),
      talib.LINEARREG,
      lambda d: (d["np_close"], 14),
    ),
    (
      "LINREGINT",
      linearreg_intercept,
      lambda d: (d["close"], 14),
      talib.LINEARREG_INTERCEPT,
      lambda d: (d["np_close"], 14),
    ),
    (
      "LINREGANG",
      linearreg_angle,
      lambda d: (d["close"], 14),
      talib.LINEARREG_ANGLE,
      lambda d: (d["np_close"], 14),
    ),
    (
      "TSF",
      tsf,
      lambda d: (d["close"], 14),
      talib.TSF,
      lambda d: (d["np_close"], 14),
    ),
    (
      "BETA_STAT",
      beta_statistical,
      lambda d: (d["close"], d["open"], 5),
      None,
      None,
    ),
    (
      "BETA",
      beta,
      lambda d: (d["close"], d["open"], 5),
      talib.BETA,
      lambda d: (d["np_close"], d["np_open"], 5),
    ),
    (
      "CORREL",
      correl,
      lambda d: (d["close"], d["open"], 30),
      talib.CORREL,
      lambda d: (d["np_close"], d["np_open"], 30),
    ),
    (
      "HT_Trendline",
      ht_trendline,
      lambda d: (d["close"],),
      talib.HT_TRENDLINE,
      lambda d: (d["np_close"],),
    ),
    (
      "MAMA",
      mama,
      lambda d: (d["close"], 0.5, 0.05),
      talib.MAMA,
      lambda d: (d["np_close"], 0.5, 0.05),
    ),
    (
      "SAREXT",
      sarext,
      lambda d: (d["high"], d["low"]),
      talib.SAREXT,
      lambda d: (d["np_high"], d["np_low"]),
    ),
    (
      "MACDEXT",
      macdext,
      lambda d: (d["close"], 12, 0, 26, 0, 9, 0),
      talib.MACDEXT,
      lambda d: (d["np_close"], 12, 0, 26, 0, 9, 0),
    ),
    (
      "MACDFIX",
      macdfix,
      lambda d: (d["close"], 9),
      talib.MACDFIX,
      lambda d: (d["np_close"], 9),
    ),
    (
      "Gap3Methods",
      cdl_xsidegap3methods,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLXSIDEGAP3METHODS,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "DOJI",
      cdl_doji,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLDOJI,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "HAMMER",
      cdl_hammer,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLHAMMER,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "ENGULFING",
      cdl_engulfing,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLENGULFING,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "HARAMI",
      cdl_harami,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLHARAMI,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "ShootingStar",
      cdl_shooting_star,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLSHOOTINGSTAR,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "InvHammer",
      cdl_inverted_hammer,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLINVERTEDHAMMER,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "HangingMan",
      cdl_hanging_man,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLHANGINGMAN,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Marubozu",
      cdl_marubozu,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLMARUBOZU,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "MorningStar",
      cdl_morning_star,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLMORNINGSTAR,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "EveningStar",
      cdl_evening_star,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLEVENINGSTAR,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "3BlackCrows",
      cdl_3black_crows,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDL3BLACKCROWS,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "3WhiteSoldiers",
      cdl_3white_soldiers,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDL3WHITESOLDIERS,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "3Inside",
      cdl_3inside,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDL3INSIDE,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "3Outside",
      cdl_3outside,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDL3OUTSIDE,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "3LineStrike",
      cdl_3line_strike,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDL3LINESTRIKE,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Piercing",
      cdl_piercing,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLPIERCING,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "DarkCloud",
      cdl_dark_cloud_cover,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLDARKCLOUDCOVER,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Kicking",
      cdl_kicking,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLKICKING,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "MatchingLow",
      cdl_matching_low,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLMATCHINGLOW,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "SpinningTop",
      cdl_spinning_top,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLSPINNINGTOP,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "RickshawMan",
      cdl_rickshaw_man,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLRICKSHAWMAN,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "HighWave",
      cdl_high_wave,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLHIGHWAVE,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "LongLegDoji",
      cdl_long_legged_doji,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLLONGLEGGEDDOJI,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Tristar",
      cdl_tristar,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLTRISTAR,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "TasukiGap",
      cdl_tasuki_gap,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLTASUKIGAP,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "SepLines",
      cdl_separating_lines,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLSEPARATINGLINES,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "GapSideSide",
      cdl_gap_side_by_side_white,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLGAPSIDESIDEWHITE,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "2Crows",
      cdl_2crows,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDL2CROWS,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "UpGap2Crows",
      cdl_upside_gap_two_crows,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLUPSIDEGAP2CROWS,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "AbandBaby",
      cdl_abandoned_baby,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLABANDONEDBABY,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "AdvBlock",
      cdl_advance_block,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLADVANCEBLOCK,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "BeltHold",
      cdl_belt_hold,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLBELTHOLD,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Breakaway",
      cdl_breakaway,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLBREAKAWAY,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "ClosingMaru",
      cdl_closing_marubozu,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLCLOSINGMARUBOZU,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Dragonfly",
      cdl_dragonfly_doji,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLDRAGONFLYDOJI,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Gravestone",
      cdl_gravestone_doji,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLGRAVESTONEDOJI,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Hikkake",
      cdl_hikkake,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLHIKKAKE,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "HomingPig",
      cdl_homing_pigeon,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLHOMINGPIGEON,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Ident3Crows",
      cdl_identical_3crows,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLIDENTICAL3CROWS,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "InNeck",
      cdl_in_neck,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLINNECK,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "LadderBot",
      cdl_ladder_bottom,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLLADDERBOTTOM,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "LongLine",
      cdl_long_line,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLLONGLINE,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "MatHold",
      cdl_mat_hold,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLMATHOLD,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "OnNeck",
      cdl_on_neck,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLONNECK,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "3Methods",
      cdl_rise_fall_3methods,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLRISEFALL3METHODS,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "ShortLine",
      cdl_short_line,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLSHORTLINE,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Stalled",
      cdl_stalled_pattern,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLSTALLEDPATTERN,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "StickSand",
      cdl_stick_sandwich,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLSTICKSANDWICH,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Takuri",
      cdl_takuri,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLTAKURI,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Thrusting",
      cdl_thrusting,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLTHRUSTING,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "3River",
      cdl_unique_3river,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLUNIQUE3RIVER,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "Counter",
      cdl_counterattack,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLCOUNTERATTACK,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "DojiStar",
      cdl_doji_star,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLDOJISTAR,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "ConcealBaby",
      cdl_conceal_baby_swallow,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLCONCEALBABYSWALL,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "HaramiCross",
      cdl_harami_cross,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLHARAMICROSS,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "HikkakeMod",
      cdl_hikkake_mod,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLHIKKAKEMOD,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "MornDojiStar",
      cdl_morning_doji_star,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLMORNINGDOJISTAR,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "EvenDojiStar",
      cdl_evening_doji_star,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLEVENINGDOJISTAR,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "KickLength",
      cdl_kicking_by_length,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDLKICKINGBYLENGTH,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    (
      "3StarsSouth",
      cdl_3stars_in_south,
      lambda d: (d["open"], d["high"], d["low"], d["close"]),
      talib.CDL3STARSINSOUTH,
      lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
    ),
    # fmt: on
  ]

  # Filter benchmarks if requested
  if args.indicators:
    target_names = {name.lower() for name in args.indicators}
    benchmarks = [b for b in benchmarks if b[0].lower() in target_names]

    if not benchmarks:
      print("No benchmarks matched the requested indicators.")
      return

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
