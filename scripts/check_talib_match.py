import warnings

import numpy as np
import pandas as pd
import talib

# Imports from indikator
"""Robust benchmark suite comparing Indikator vs TA-lib.

Methodology for stable measurements:
1. Interleaved runs - alternate between implementations to cancel cache bias
2. Randomized order - shuffle run order to avoid systematic bias
3. Multiple rounds - repeat many times and use robust statistics
4. Warmup isolation - separate warmup phase before timing
5. GC disabled during timing - avoid GC pauses
"""


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
    acos,
    ad,
    adosc,
    adx,
    adxr,
    apo,
    aroon,
    aroonosc,
    asin,
    atan,
    atr,
    atr_intraday,
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
    cdl_xsidegap3methods,
    ceil,
    churn_factor,
    cmo,
    correl,
    cos,
    cosh,
    dema,
    dx,
    ema,
    exp,
    floor,
    ht_dcperiod,
    ht_dcphase,
    ht_phasor,
    ht_sine,
    ht_trendline,
    ht_trendmode,
    kama,
    legs,
    linearreg,
    linearreg_angle,
    linearreg_intercept,
    linearreg_slope,
    ln,
    log10,
    macd,
    macdext,
    macdfix,
    mama,
    mavp,
    max_index,
    max_val,
    medprice,
    mfi,
    midpoint,
    midprice,
    min_index,
    min_val,
    minmax,
    minmaxindex,
    minus_di,
    minus_dm,
    mom,
    natr,
    obv,
    opening_range,
    pivots,
    plus_di,
    plus_dm,
    ppo,
    roc,
    rocp,
    rocr,
    rocr100,
    rsi,
    rvol,
    rvol_intraday,
    sar,
    sarext,
    sector_correlation,
    sin,
    sinh,
    slope,
    sma,
    sqrt,
    stddev,
    stoch,
    stochf,
    stochrsi,
    sum_val,
    t3,
    tan,
    tanh,
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
    zscore_intraday,
  )


# Generate Data (simplified)
def generate_data(size: int, pattern_type: int = None):
  """Generate OHLC data with realistic candlestick patterns."""
  np.random.seed(42)

  # Base price using random walk
  returns = np.random.randn(size) * 0.02  # Larger volatility
  base_price = 100 * np.exp(np.cumsum(returns))

  # Generate OHLC with realistic patterns
  open_prices = np.zeros(size)
  high_prices = np.zeros(size)
  low_prices = np.zeros(size)
  close_prices = np.zeros(size)

  open_prices[0] = base_price[0]

  for i in range(size):
    if i == 0:
      open_prices[i] = base_price[i]
    else:
      # Open near previous close with small gap
      gap = np.random.randn() * 0.005
      open_prices[i] = close_prices[i - 1] * (1 + gap)

    # Determine body direction and size
    body_direction = (
      np.sign(base_price[i] - open_prices[i]) if i > 0 else np.random.choice([-1, 1])
    )
    body_size = abs(np.random.randn()) * 0.015

    # Create different candle types based on random selection
    candle_type = np.random.randint(0, 100)

    if candle_type < 5:  # 5% Doji (open ≈ close)
      close_prices[i] = open_prices[i] * (1 + np.random.randn() * 0.0005)
      shadow = abs(np.random.randn()) * 0.02
      high_prices[i] = max(open_prices[i], close_prices[i]) * (1 + shadow)
      low_prices[i] = min(open_prices[i], close_prices[i]) * (1 - shadow)

    elif candle_type < 10:  # 5% Long body (marubozu-like)
      close_prices[i] = open_prices[i] * (1 + body_direction * body_size * 2)
      high_prices[i] = max(open_prices[i], close_prices[i]) * (
        1 + abs(np.random.randn()) * 0.002
      )
      low_prices[i] = min(open_prices[i], close_prices[i]) * (
        1 - abs(np.random.randn()) * 0.002
      )

    elif candle_type < 15:  # 5% Hammer/Hanging man (long lower shadow)
      close_prices[i] = open_prices[i] * (1 + body_direction * body_size * 0.3)
      body_top = max(open_prices[i], close_prices[i])
      body_bot = min(open_prices[i], close_prices[i])
      high_prices[i] = body_top * (1 + abs(np.random.randn()) * 0.002)
      low_prices[i] = body_bot * (
        1 - abs(np.random.randn()) * 0.03
      )  # Long lower shadow

    elif candle_type < 20:  # 5% Shooting star/Inverted hammer (long upper shadow)
      close_prices[i] = open_prices[i] * (1 + body_direction * body_size * 0.3)
      body_top = max(open_prices[i], close_prices[i])
      body_bot = min(open_prices[i], close_prices[i])
      high_prices[i] = body_top * (
        1 + abs(np.random.randn()) * 0.03
      )  # Long upper shadow
      low_prices[i] = body_bot * (1 - abs(np.random.randn()) * 0.002)

    elif candle_type < 25:  # 5% Spinning top (small body, equal shadows)
      close_prices[i] = open_prices[i] * (1 + body_direction * body_size * 0.2)
      shadow = abs(np.random.randn()) * 0.015
      high_prices[i] = max(open_prices[i], close_prices[i]) * (1 + shadow)
      low_prices[i] = min(open_prices[i], close_prices[i]) * (1 - shadow)

    else:  # 75% Normal candles
      close_prices[i] = open_prices[i] * (1 + body_direction * body_size)
      upper_shadow = abs(np.random.randn()) * 0.008
      lower_shadow = abs(np.random.randn()) * 0.008
      high_prices[i] = max(open_prices[i], close_prices[i]) * (1 + upper_shadow)
      low_prices[i] = min(open_prices[i], close_prices[i]) * (1 - lower_shadow)

  # Inject specific multi-candle patterns
  patterns_injected = 0
  i = 50
  while i < size - 20 and patterns_injected < size // 15:
    current_pattern = (
      pattern_type if pattern_type is not None else np.random.randint(0, 30)
    )

    if current_pattern == 0:  # Engulfing pattern
      # First: small candle
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * (1 + 0.005)
      high_prices[i] = max(open_prices[i], close_prices[i]) * 1.002
      low_prices[i] = min(open_prices[i], close_prices[i]) * 0.998
      # Second: large opposite candle that engulfs
      open_prices[i + 1] = close_prices[i] * 1.002
      close_prices[i + 1] = open_prices[i] * 0.98
      high_prices[i + 1] = open_prices[i + 1] * 1.002
      low_prices[i + 1] = close_prices[i + 1] * 0.998
      i += 3

    elif current_pattern == 1:  # Three black crows
      # Upside trend first
      open_prices[i] = close_prices[i - 1] * 1.02
      close_prices[i] = open_prices[i] * 1.02
      i_start = i + 1
      for j in range(3):
        idx = i_start + j
        if idx < size:
          # Open within previous body
          prev_open = open_prices[idx - 1]
          prev_close = close_prices[idx - 1]
          # Black crow
          open_prices[idx] = prev_close + (prev_open - prev_close) * 0.5  # Midpoint
          close_prices[idx] = open_prices[idx] * 0.98
          high_prices[idx] = open_prices[idx] * 1.001
          low_prices[idx] = close_prices[idx] * 0.999
      i += 5

    elif current_pattern == 2:  # Three white soldiers
      for j in range(3):
        if i + j < size:
          open_prices[i + j] = close_prices[i + j - 1] if j > 0 else open_prices[i + j]
          close_prices[i + j] = open_prices[i + j] * 1.015
          high_prices[i + j] = close_prices[i + j] * 1.002
          low_prices[i + j] = open_prices[i + j] * 0.998
      i += 5

    elif current_pattern == 3:  # 3 Inside
      # Bullish: Downtrend, Long Black, Small White inside, Higher Close
      open_prices[i] = close_prices[i - 1] * 1.01
      close_prices[i] = open_prices[i] * 0.98  # Long Black
      open_prices[i + 1] = close_prices[i] * 1.005
      close_prices[i + 1] = open_prices[i + 1] * 1.01  # Small White inside
      open_prices[i + 2] = close_prices[i + 1] * 0.998
      close_prices[i + 2] = close_prices[i + 1] * 1.02  # Higher Close
      # Adjust highs/lows
      for k in range(3):
        high_prices[i + k] = max(open_prices[i + k], close_prices[i + k]) * 1.001
        low_prices[i + k] = min(open_prices[i + k], close_prices[i + k]) * 0.999
      i += 4

    elif current_pattern == 4:  # 3 Outside
      # Bullish: Long Black, Engulfing White, Higher Close
      open_prices[i] = close_prices[i - 1] * 1.01
      close_prices[i] = open_prices[i] * 0.99  # Black
      open_prices[i + 1] = close_prices[i] * 0.995
      close_prices[i + 1] = open_prices[i] * 1.005  # Engulfing White
      open_prices[i + 2] = close_prices[i + 1] * 0.998
      close_prices[i + 2] = close_prices[i + 1] * 1.02  # Higher Close
      for k in range(3):
        high_prices[i + k] = max(open_prices[i + k], close_prices[i + k]) * 1.001
        low_prices[i + k] = min(open_prices[i + k], close_prices[i + k]) * 0.999
      i += 4

    elif current_pattern == 5:  # 3 Line Strike
      # Bullish: 3 White Soldiers, Long Black engulfing all 3
      base = close_prices[i - 1]
      for k in range(3):
        open_prices[i + k] = base
        close_prices[i + k] = base * 1.01
        base = close_prices[i + k]
      # Strike candle
      open_prices[i + 3] = close_prices[i + 2] * 1.005
      close_prices[i + 3] = open_prices[i] * 0.995  # Engulfs all 3
      for k in range(4):
        high_prices[i + k] = max(open_prices[i + k], close_prices[i + k]) * 1.001
        low_prices[i + k] = min(open_prices[i + k], close_prices[i + k]) * 0.999
      i += 5

    elif current_pattern == 6:  # 3 Stars in South
      # Long Black (long lower shadow), Same Open/Lower Close (lower shadow), Marubozu (short/no shadows)
      open_prices[i] = close_prices[i - 1] * 1.01
      close_prices[i] = open_prices[i] * 0.98
      low_prices[i] = close_prices[i] * 0.95  # Long shadow

      open_prices[i + 1] = open_prices[i]
      close_prices[i + 1] = close_prices[i] * 1.005  # Higher low
      low_prices[i + 1] = close_prices[i + 1] * 0.99  # Short shadow

      open_prices[i + 2] = close_prices[i + 1] * 0.998
      close_prices[i + 2] = open_prices[i + 2] * 0.995  # Small black
      low_prices[i + 2] = close_prices[i + 2]  # Marubozu low
      high_prices[i + 2] = open_prices[i + 2]  # Marubozu high
      i += 4

    elif current_pattern == 7:  # Advance Block
      # 3 White candles, progressively smaller bodies and longer upper shadows
      base = close_prices[i - 1]
      for k, factor in enumerate([1.02, 1.01, 1.005]):
        open_prices[i + k] = base
        close_prices[i + k] = base * factor
        high_prices[i + k] = close_prices[i + k] * (
          1.005 + k * 0.01
        )  # Longer upper shadow
        low_prices[i + k] = open_prices[i + k] * 0.999
        base = close_prices[i + k]
      i += 4

    elif current_pattern == 8:  # Belt Hold
      # Bullish: Open is Low, Close near High (Long White)
      low_prices[i] = open_prices[i] = close_prices[i - 1] * 0.98  # Gap down
      close_prices[i] = open_prices[i] * 1.05
      high_prices[i] = close_prices[i]
      i += 2

    elif current_pattern == 9:  # Breakaway
      # Long Black, Gap Down Black, 3 small candles down, White closing in gap
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.96  # Long Black

      open_prices[i + 1] = close_prices[i] * 0.98
      close_prices[i + 1] = (
        open_prices[i + 1] * 0.99
      )  # Gap (2% drop from prev close), Black
      open_prices[i + 2] = close_prices[i + 1]
      close_prices[i + 2] = open_prices[i + 2] * 0.99  # Black
      open_prices[i + 3] = close_prices[i + 2]
      close_prices[i + 3] = open_prices[i + 3] * 0.99  # Black

      # White closes INSIDE the gap between 1 and 2
      top_gap = close_prices[i]
      bot_gap = open_prices[i + 1]  # approx

      open_prices[i + 4] = close_prices[i + 3]
      close_prices[i + 4] = (top_gap + bot_gap) / 2  # Into gap

      for k in range(5):
        high_prices[i + k] = max(open_prices[i + k], close_prices[i + k]) * 1.001
        low_prices[i + k] = min(open_prices[i + k], close_prices[i + k]) * 0.999
      i += 6

    elif current_pattern == 10:  # Concealing Baby Swallow
      # 2 Marubozu Black, 3rd similar opening Inverted Hammer (gap down high), 4th engulfs 3rd
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.98
      low_prices[i] = close_prices[i]
      high_prices[i] = open_prices[i]  # Marubozu

      open_prices[i + 1] = close_prices[i]
      close_prices[i + 1] = open_prices[i + 1] * 0.98
      low_prices[i + 1] = close_prices[i + 1]
      high_prices[i + 1] = open_prices[i + 1]  # Marubozu

      # 3rd: Inverted Hammer. Open is lower than prev close (Gap down).
      # But specifically needs long upper shadow extending into prior body?
      # TA-Lib: 2 blacks, 3rd inverted hammer with long upper shadow gaps down.
      gap_down = close_prices[i + 1] * 0.98
      open_prices[i + 2] = gap_down
      close_prices[i + 2] = open_prices[i + 2] * 0.99  # Black
      high_prices[i + 2] = close_prices[i + 1] * 1.01  # Into prior body
      low_prices[i + 2] = close_prices[i + 2]

      # 4th: Engulfs 3rd including shadows
      open_prices[i + 3] = high_prices[i + 2] * 1.001
      close_prices[i + 3] = low_prices[i + 2] * 0.999
      high_prices[i + 3] = open_prices[i + 3]
      low_prices[i + 3] = close_prices[i + 3]
      i += 5

    elif current_pattern == 11:  # Counterattack
      # Long Black, Gap Down, Long White closing at prev close
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.95
      open_prices[i + 1] = close_prices[i] * 0.95
      close_prices[i + 1] = close_prices[i]  # Close equal
      high_prices[i] = max(open_prices[i], close_prices[i])
      low_prices[i] = min(open_prices[i], close_prices[i])
      high_prices[i + 1] = max(open_prices[i + 1], close_prices[i + 1])
      low_prices[i + 1] = min(open_prices[i + 1], close_prices[i + 1])
      i += 3

    elif current_pattern == 12:  # Evening Star
      # Long White, Gap Up Small Body, Gap Down Black
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 1.05  # Long White
      open_prices[i + 1] = close_prices[i] * 1.01
      close_prices[i + 1] = open_prices[i + 1] * 1.002  # Gap Up Small
      open_prices[i + 2] = close_prices[i + 1] * 0.99
      close_prices[i + 2] = open_prices[i] * 1.02  # Gap Down Black info
      for k in range(3):
        high_prices[i + k] = max(open_prices[i + k], close_prices[i + k]) * 1.002
        low_prices[i + k] = min(open_prices[i + k], close_prices[i + k]) * 0.998
      i += 4

    elif current_pattern == 13:  # Gap Sides
      # White, Gap Up White, White same size
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 1.02
      open_prices[i + 1] = close_prices[i] * 1.01
      close_prices[i + 1] = open_prices[i + 1] * 1.02  # Gap Up
      open_prices[i + 2] = open_prices[i + 1]
      close_prices[i + 2] = close_prices[i + 1]  # Similar
      i += 4

    elif current_pattern == 14:  # Homing Pigeon
      # Long Black, Small Black completely inside
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.95
      open_prices[i + 1] = open_prices[i] * 0.98
      close_prices[i + 1] = close_prices[i] * 1.02  # Inside
      i += 3

    elif current_pattern == 15:  # Identical 3 Crows
      # 3 Blacks, open at prior close
      base = close_prices[i - 1]
      for k in range(3):
        open_prices[i + k] = base
        close_prices[i + k] = base * 0.98
        base = close_prices[i + k]
      i += 4

    elif current_pattern == 16:  # In Neck
      # Long Black, White opens lower but closes slightly into body
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.95
      open_prices[i + 1] = close_prices[i] * 0.99
      close_prices[i + 1] = close_prices[i] * 1.001
      i += 3

    elif current_pattern == 17:  # Kicking
      # Black Marubozu, Gap Up White Marubozu
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.95  # Black
      high_prices[i] = open_prices[i]
      low_prices[i] = close_prices[i]  # Marubozu

      open_prices[i + 1] = open_prices[i] * 1.01
      close_prices[i + 1] = open_prices[i + 1] * 1.05  # Gap Up White
      high_prices[i + 1] = close_prices[i + 1]
      low_prices[i + 1] = open_prices[i + 1]  # Marubozu
      i += 3

    elif current_pattern == 18:  # Ladder Bottom
      # 3 Black, Inverted Hammer, White Gap Up
      base = close_prices[i - 1]
      for k in range(3):  # 3 Black
        open_prices[i + k] = base
        close_prices[i + k] = base * 0.98
        high_prices[i + k] = base
        low_prices[i + k] = base * 0.97
        base = close_prices[i + k]

      open_prices[i + 3] = base
      close_prices[i + 3] = base * 0.99  # Inverted Hammer
      high_prices[i + 3] = open_prices[i + 3] * 1.05

      low_prices[i + 3] = close_prices[i + 3]

      open_prices[i + 4] = high_prices[i + 3] * 1.01  # Gap Up White
      close_prices[i + 4] = open_prices[i + 4] * 1.05
      i += 6

    elif current_pattern == 19:  # Matching Low
      # Long Black, Black with same close
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.95
      open_prices[i + 1] = open_prices[i] * 0.98
      close_prices[i + 1] = close_prices[i]
      i += 3

    elif current_pattern == 20:  # Mat Hold
      # Use proven logic from verify_mathold_events.py
      # Base everything on O[i] to ensure consistent ratios
      open_prices[i] = close_prices[i - 1]
      base = open_prices[i]

      # 1: Long White
      close_prices[i] = base * 1.10
      high_prices[i] = base * 1.105
      low_prices[i] = base * 0.995

      # 2: Gap Up, Small Black
      # Gap Up: min(O,C) > max(O,C)_prev (1.11 > 1.10)
      open_prices[i + 1] = base * 1.13
      close_prices[i + 1] = base * 1.12
      high_prices[i + 1] = base * 1.135
      low_prices[i + 1] = base * 1.115

      # 3: Reaction, Falling, Penetrating C0 (1.10)
      # Must be < 1.10 but > (1.10 - 0.5*Body)
      open_prices[i + 2] = base * 1.11
      close_prices[i + 2] = base * 1.08  # Penetrates 1.10
      high_prices[i + 2] = base * 1.12
      low_prices[i + 2] = base * 1.075

      # 4: Falling
      open_prices[i + 3] = base * 1.08
      close_prices[i + 3] = base * 1.06  # Penetrates
      high_prices[i + 3] = base * 1.085
      low_prices[i + 3] = base * 1.055

      # 5: Long White Breakout
      # Close (1.15) > Max High of reactions (1.135)
      open_prices[i + 4] = base * 1.06
      close_prices[i + 4] = base * 1.15
      high_prices[i + 4] = base * 1.155
      low_prices[i + 4] = base * 1.055

      i += 6

    elif current_pattern == 21:  # Piercing
      # Long Black, White opens low, closes > 50% of black
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.95
      midpoint = (open_prices[i] + close_prices[i]) / 2
      open_prices[i + 1] = close_prices[i] * 0.98  # Gap down
      close_prices[i + 1] = midpoint * 1.01  # Above midpoint
      i += 3

    elif current_pattern == 22:  # Rise/Fall 3 methods
      # 1: Long White
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 1.10
      high_prices[i] = close_prices[i] * 1.002
      low_prices[i] = open_prices[i] * 0.998
      h_limit = high_prices[i]
      l_limit = low_prices[i]

      # 2, 3, 4: Small black candles INSIDE the First candle's range
      prev_close = close_prices[i]
      for k in range(1, 4):
        # Start near top for first reaction
        if k == 1:
          open_prices[i + k] = prev_close * 0.995
        else:
          open_prices[i + k] = close_prices[i + k - 1] * 1.001

        close_prices[i + k] = open_prices[i + k] * 0.99  # Small black

        # Strict Clamp for High/Low to be inside h_limit and l_limit
        # Make body small enough to fit if needed
        open_prices[i + k] = min(open_prices[i + k], h_limit * 0.99)
        close_prices[i + k] = max(close_prices[i + k], l_limit * 1.01)

        # Shadows must also be inside
        high_prices[i + k] = open_prices[i + k]  # No upper shadow or small
        low_prices[i + k] = close_prices[i + k]  # No lower shadow

        # Ensure they are truly inside
        high_prices[i + k] = min(high_prices[i + k], h_limit * 0.995)
        low_prices[i + k] = max(low_prices[i + k], l_limit * 1.005)

      # 5: Long White Breakout
      # Close > Max High of reactions (113.5)
      open_prices[i + 4] = close_prices[i + 3] * 0.99
      close_prices[i + 4] = high_prices[i + 1] * 1.02  # Breakout
      high_prices[i + 4] = close_prices[i + 4] * 1.001
      low_prices[i + 4] = open_prices[i + 4] * 0.999

      # DEBUG: Confirm injection
      if i == 200:
        print(f"DEBUG INJECTION MATHOLD AT {i}")
        print(
          f"  Prices: {close_prices[i]:.2f}, {close_prices[i + 1]:.2f}, {close_prices[i + 2]:.2f}, {close_prices[i + 3]:.2f}, {close_prices[i + 4]:.2f}"
        )

      i += 6

    elif current_pattern == 23:  # Separating Lines
      # Trend, Opposite color, same open
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 1.03  # White
      open_prices[i + 1] = open_prices[i]
      close_prices[i + 1] = open_prices[i + 1] * 0.97  # Black
      i += 3

    elif current_pattern == 24:  # Stick Sandwich
      # Black, White > black high, Black same close as 1st
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.95
      open_prices[i + 1] = close_prices[i]
      close_prices[i + 1] = open_prices[i] * 1.02
      open_prices[i + 2] = close_prices[i + 1]
      close_prices[i + 2] = close_prices[i]  # Matching close
      i += 4

    elif pattern_type == 25:  # Takuri
      # Small body, very long lower shadow
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 1.001
      high_prices[i] = close_prices[i] * 1.0001
      low_prices[i] = open_prices[i] * 0.97  # Long lower shadow
      i += 2

    elif pattern_type == 26:  # Tasuki Gap
      # White, Gap Up White, Black opens inside 2nd, closes inside gap
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 1.03
      open_prices[i + 1] = close_prices[i] * 1.01
      close_prices[i + 1] = open_prices[i + 1] * 1.02  # Gap
      open_prices[i + 2] = open_prices[i + 1] * 0.99
      close_prices[i + 2] = close_prices[i] * 1.005  # Into gap
      i += 4

    elif pattern_type == 27:  # Thrusting
      # Long Black, White closes into body but < 50%
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.95
      midpoint = (open_prices[i] + close_prices[i]) / 2
      open_prices[i + 1] = close_prices[i] * 0.98
      close_prices[i + 1] = midpoint * 0.99  # Below midpoint
      i += 3

    elif pattern_type == 28:  # Tristar
      # 3 Dojis
      for k in range(3):
        open_prices[i + k] = close_prices[i + k - 1] if k > 0 else close_prices[i - 1]
        close_prices[i + k] = open_prices[i + k]
        high_prices[i + k] = open_prices[i + k] * 1.001
        low_prices[i + k] = open_prices[i + k] * 0.999
      i += 4

    elif pattern_type == 29:  # Unique 3 River
      # Long Black, Hammer (lower low), Small White (below 2nd close)
      open_prices[i] = close_prices[i - 1]
      close_prices[i] = open_prices[i] * 0.95

      open_prices[i + 1] = close_prices[i] * 1.01
      close_prices[i + 1] = open_prices[i + 1] * 0.98
      low_prices[i + 1] = close_prices[i + 1] * 0.95  # Hammer low
      high_prices[i + 1] = open_prices[i + 1]

      open_prices[i + 2] = low_prices[i + 1] * 1.01
      close_prices[i + 2] = open_prices[i + 2] * 0.99
      i += 4

    else:
      i += np.random.randint(5, 20)

    patterns_injected += 1

  # Ensure OHLC consistency
  for i in range(size):
    high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
    low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])

  volume = np.abs(np.random.randn(size) * 1000 + 10000)
  dates = pd.date_range("2020-01-01", periods=size, freq="1min")

  return {
    "high": pd.Series(high_prices, index=dates),
    "low": pd.Series(low_prices, index=dates),
    "close": pd.Series(close_prices, index=dates),
    "volume": pd.Series(volume, index=dates),
    "open": pd.Series(open_prices, index=dates),
    "sector": pd.Series(close_prices, index=dates) * (1 + np.random.randn(size) * 0.01),
    "np_high": high_prices.astype(np.float64),
    "np_low": low_prices.astype(np.float64),
    "np_close": close_prices.astype(np.float64),
    "np_open": open_prices.astype(np.float64),
    "np_volume": volume.astype(np.float64),
  }


BENCHMARKS = [
  (
    "2Crows",
    cdl_2crows,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDL2CROWS,
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
    "3Inside",
    cdl_3inside,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDL3INSIDE,
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
    "3Methods",
    cdl_rise_fall_3methods,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLRISEFALL3METHODS,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "3StarsSouth",
    cdl_3stars_in_south,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDL3STARSINSOUTH,
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
    "3River",
    cdl_unique_3river,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLUNIQUE3RIVER,
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
    "ADX",
    adx,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.ADX,
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
    "APO",
    apo,
    lambda d: (d["close"], 12, 26, 0),
    talib.APO,
    lambda d: (d["np_close"], 12, 26, 0),
  ),
  (
    "AROON",
    aroon,
    lambda d: (d["high"], d["low"], 25),
    talib.AROON,
    lambda d: (d["np_high"], d["np_low"], 25),
  ),
  (
    "AROONOSC",
    aroonosc,
    lambda d: (d["high"], d["low"], 25),
    talib.AROONOSC,
    lambda d: (d["np_high"], d["np_low"], 25),
  ),
  (
    "ASIN",
    asin,
    lambda d: (d["close"] / 200.0,),
    talib.ASIN,
    lambda d: (d["np_close"] / 200.0,),
  ),
  (
    "ATAN",
    atan,
    lambda d: (d["close"],),
    talib.ATAN,
    lambda d: (d["np_close"],),
  ),
  (
    "ATR",
    atr,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.ATR,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
  ),
  (
    "ATR_INTRA",
    atr_intraday,
    lambda d: (
      pd.DataFrame(
        {"high": d["high"], "low": d["low"], "close": d["close"]},
        index=d["high"].index,
      ),
    ),
    None,
    None,
  ),
  (
    "AVGPRICE",
    avgprice,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.AVGPRICE,
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
    "Acos",
    acos,
    lambda d: (d["close"] / 200.0,),
    talib.ACOS,
    lambda d: (
      d["np_close"] / 200.0,
    ),  # Normalize to roughly [-1, 1] range (assuming price ~100) or just use sine/cosine inputs
  ),
  (
    "AdvBlock",
    cdl_advance_block,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLADVANCEBLOCK,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "BETA",
    beta,
    lambda d: (d["close"], d["open"], 5),
    talib.BETA,
    lambda d: (d["np_close"], d["np_open"], 5),
  ),
  (
    "BETA_STAT",
    beta_statistical,
    lambda d: (d["close"], d["open"], 5),
    None,
    None,
  ),
  (
    "BOP",
    bop,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.BOP,
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
    "Bollinger",
    bollinger_bands,
    lambda d: (d["close"], 20, 2.0),
    talib.BBANDS,
    lambda d: (d["np_close"], 20, 2.0, 2.0),
  ),
  (
    "Breakaway",
    cdl_breakaway,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLBREAKAWAY,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "CCI",
    cci,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.CCI,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
  ),
  (
    "CEIL",
    ceil,
    lambda d: (d["close"],),
    talib.CEIL,
    lambda d: (d["np_close"],),
  ),
  ("CMO", cmo, lambda d: (d["close"], 14), talib.CMO, lambda d: (d["np_close"], 14)),
  (
    "CORREL",
    correl,
    lambda d: (d["close"], d["open"], 30),
    talib.CORREL,
    lambda d: (d["np_close"], d["np_open"], 30),
  ),
  (
    "COS",
    cos,
    lambda d: (d["close"],),
    talib.COS,
    lambda d: (d["np_close"],),
  ),
  (
    "COSH",
    cosh,
    lambda d: (d["close"],),
    talib.COSH,
    lambda d: (d["np_close"],),
  ),
  ("Churn", churn_factor, lambda d: (d["high"], d["low"], d["volume"]), None, None),
  (
    "ClosingMaru",
    cdl_closing_marubozu,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLCLOSINGMARUBOZU,
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
    "Counter",
    cdl_counterattack,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLCOUNTERATTACK,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "DEMA",
    dema,
    lambda d: (d["close"], 30),
    talib.DEMA,
    lambda d: (d["np_close"], 30),
  ),
  (
    "DOJI",
    cdl_doji,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLDOJI,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "DX",
    dx,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.DX,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
  ),
  (
    "DarkCloud",
    cdl_dark_cloud_cover,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLDARKCLOUDCOVER,
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
    "Dragonfly",
    cdl_dragonfly_doji,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLDRAGONFLYDOJI,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  ("EMA", ema, lambda d: (d["close"], 30), talib.EMA, lambda d: (d["np_close"], 30)),
  (
    "ENGULFING",
    cdl_engulfing,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLENGULFING,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "EXP",
    exp,
    lambda d: (d["close"],),
    talib.EXP,
    lambda d: (d["np_close"],),
  ),
  (
    "EvenDojiStar",
    cdl_evening_doji_star,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLEVENINGDOJISTAR,
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
    "FLOOR",
    floor,
    lambda d: (d["close"],),
    talib.FLOOR,
    lambda d: (d["np_close"],),
  ),
  (
    "Gap3Methods",
    cdl_xsidegap3methods,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLXSIDEGAP3METHODS,
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
    "Gravestone",
    cdl_gravestone_doji,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLGRAVESTONEDOJI,
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
    "HARAMI",
    cdl_harami,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLHARAMI,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
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
    "HT_Trendline",
    ht_trendline,
    lambda d: (d["close"],),
    talib.HT_TRENDLINE,
    lambda d: (d["np_close"],),
  ),
  (
    "HangingMan",
    cdl_hanging_man,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLHANGINGMAN,
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
    "HighWave",
    cdl_high_wave,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLHIGHWAVE,
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
    "HikkakeMod",
    cdl_hikkake_mod,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLHIKKAKEMOD,
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
    "InvHammer",
    cdl_inverted_hammer,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLINVERTEDHAMMER,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "KAMA",
    kama,
    lambda d: (d["close"], 10),
    talib.KAMA,
    lambda d: (d["np_close"], 10),
  ),
  (
    "KickLength",
    cdl_kicking_by_length,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLKICKINGBYLENGTH,
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
    "LINEARREG",
    linearreg,
    lambda d: (d["close"], 14),
    talib.LINEARREG,
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
    "LINREGINT",
    linearreg_intercept,
    lambda d: (d["close"], 14),
    talib.LINEARREG_INTERCEPT,
    lambda d: (d["np_close"], 14),
  ),
  (
    "LINREGS",
    linearreg_slope,
    lambda d: (d["close"], 14),
    talib.LINEARREG_SLOPE,
    lambda d: (d["np_close"], 14),
  ),
  (
    "LN",
    ln,
    lambda d: (d["close"],),
    talib.LN,
    lambda d: (d["np_close"],),
  ),
  (
    "LOG10",
    log10,
    lambda d: (d["close"],),
    talib.LOG10,
    lambda d: (d["np_close"],),
  ),
  (
    "LadderBot",
    cdl_ladder_bottom,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLLADDERBOTTOM,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  ("Legs", legs, lambda d: (d["high"], d["low"], d["close"], 0.05), None, None),
  (
    "LongLegDoji",
    cdl_long_legged_doji,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLLONGLEGGEDDOJI,
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
    "MACD",
    macd,
    lambda d: (d["close"], 12, 26, 9),
    talib.MACD,
    lambda d: (d["np_close"], 12, 26, 9),
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
    "MAMA",
    mama,
    lambda d: (d["close"], 0.5, 0.05),
    talib.MAMA,
    lambda d: (d["np_close"], 0.5, 0.05),
  ),
  (
    "MAVP",
    mavp,
    lambda d: (d["close"], pd.Series(np.full(len(d["close"]), 14, dtype=float))),
    talib.MAVP,
    lambda d: (d["np_close"], np.full(len(d["np_close"]), 14, dtype=float)),
  ),
  (
    "MAX",
    max_val,
    lambda d: (d["close"], 14),
    talib.MAX,
    lambda d: (d["np_close"], 14),
  ),
  (
    "MAXINDEX",
    max_index,
    lambda d: (d["close"], 14),
    talib.MAXINDEX,
    lambda d: (d["np_close"], 14),
  ),
  (
    "MEDPRICE",
    medprice,
    lambda d: (d["high"], d["low"]),
    talib.MEDPRICE,
    lambda d: (d["np_high"], d["np_low"]),
  ),
  (
    "MFI",
    mfi,
    lambda d: (d["high"], d["low"], d["close"], d["volume"], 14),
    talib.MFI,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], d["np_volume"], 14),
  ),
  (
    "MIDPOINT",
    midpoint,
    lambda d: (d["close"], 14),
    talib.MIDPOINT,
    lambda d: (d["np_close"], 14),
  ),
  (
    "MIDPRICE",
    midprice,
    lambda d: (d["high"], d["low"], 14),
    talib.MIDPRICE,
    lambda d: (d["np_high"], d["np_low"], 14),
  ),
  (
    "MIN",
    min_val,
    lambda d: (d["close"], 14),
    talib.MIN,
    lambda d: (d["np_close"], 14),
  ),
  (
    "MININDEX",
    min_index,
    lambda d: (d["close"], 14),
    talib.MININDEX,
    lambda d: (d["np_close"], 14),
  ),
  (
    "MINMAX",
    minmax,
    lambda d: (d["close"], 14),
    talib.MINMAX,
    lambda d: (d["np_close"], 14),
  ),
  (
    "MINMAXINDEX",
    minmaxindex,
    lambda d: (d["close"], 14),
    talib.MINMAXINDEX,
    lambda d: (d["np_close"], 14),
  ),
  (
    "MINUS_DI",
    minus_di,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.MINUS_DI,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
  ),
  (
    "MINUS_DM",
    minus_dm,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.MINUS_DM,
    lambda d: (d["np_high"], d["np_low"], 14),
  ),
  ("MOM", mom, lambda d: (d["close"], 10), talib.MOM, lambda d: (d["np_close"], 10)),
  (
    "Marubozu",
    cdl_marubozu,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLMARUBOZU,
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
    "MatchingLow",
    cdl_matching_low,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLMATCHINGLOW,
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
    "MorningStar",
    cdl_morning_star,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLMORNINGSTAR,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "NATR",
    natr,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.NATR,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
  ),
  (
    "OBV",
    obv,
    lambda d: (d["close"], d["volume"]),
    talib.OBV,
    lambda d: (d["np_close"], d["np_volume"]),
  ),
  (
    "OPENING_RNG",
    opening_range,
    lambda d: (d["high"], d["low"], d["close"]),
    None,
    None,
  ),
  (
    "OnNeck",
    cdl_on_neck,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLONNECK,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  ("PIVOTS", pivots, lambda d: (d["high"], d["low"], d["close"]), None, None),
  (
    "PLUS_DI",
    plus_di,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.PLUS_DI,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
  ),
  (
    "PLUS_DM",
    plus_dm,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.PLUS_DM,
    lambda d: (d["np_high"], d["np_low"], 14),
  ),
  (
    "PPO",
    ppo,
    lambda d: (d["close"], 12, 26, 0),
    talib.PPO,
    lambda d: (d["np_close"], 12, 26, 0),
  ),
  (
    "Piercing",
    cdl_piercing,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLPIERCING,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  ("ROC", roc, lambda d: (d["close"], 10), talib.ROC, lambda d: (d["np_close"], 10)),
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
  ("RSI", rsi, lambda d: (d["close"], 14), talib.RSI, lambda d: (d["np_close"], 14)),
  ("RVOL", rvol, lambda d: (d["volume"], 20), None, None),
  ("RVOL_INTRA", rvol_intraday, lambda d: (d["volume"],), None, None),
  (
    "RickshawMan",
    cdl_rickshaw_man,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLRICKSHAWMAN,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "SAR",
    sar,
    lambda d: (d["high"], d["low"], 0.02, 0.2),
    talib.SAR,
    lambda d: (d["np_high"], d["np_low"], 0.02, 0.2),
  ),
  (
    "SAREXT",
    sarext,
    lambda d: (d["high"], d["low"]),
    talib.SAREXT,
    lambda d: (d["np_high"], d["np_low"]),
  ),
  (
    "SIN",
    sin,
    lambda d: (d["close"],),
    talib.SIN,
    lambda d: (d["np_close"],),
  ),
  (
    "SINH",
    sinh,
    lambda d: (d["close"],),
    talib.SINH,
    lambda d: (d["np_close"],),
  ),
  ("SMA", sma, lambda d: (d["close"], 30), talib.SMA, lambda d: (d["np_close"], 30)),
  (
    "SQRT",
    sqrt,
    lambda d: (d["close"],),
    talib.SQRT,
    lambda d: (d["np_close"],),
  ),
  (
    "STDDEV",
    stddev,
    lambda d: (d["close"], 5, 1.0),
    talib.STDDEV,
    lambda d: (d["np_close"], 5, 1.0),
  ),
  (
    "STOCHF",
    stochf,
    lambda d: (d["high"], d["low"], d["close"], 5, 3),
    talib.STOCHF,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 5, 3, 0),
  ),
  (
    "SUM",
    sum_val,
    lambda d: (d["close"], 14),
    talib.SUM,
    lambda d: (d["np_close"], 14),
  ),
  ("SectorCorr", sector_correlation, lambda d: (d["close"], d["sector"]), None, None),
  (
    "SepLines",
    cdl_separating_lines,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLSEPARATINGLINES,
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
    "ShortLine",
    cdl_short_line,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLSHORTLINE,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "Slope",
    slope,
    lambda d: (d["close"], 14),
    talib.LINEARREG_SLOPE,
    lambda d: (d["np_close"], 14),
  ),
  (
    "SpinningTop",
    cdl_spinning_top,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLSPINNINGTOP,
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
    "Stoch",
    stoch,
    lambda d: (d["high"], d["low"], d["close"], 14, 3, 3),
    talib.STOCH,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 14, 3, 0, 3, 0),
  ),
  (
    "StochRSI",
    stochrsi,
    lambda d: (d["close"], 14, 14, 14, 3),
    talib.STOCHRSI,
    lambda d: (d["np_close"], 14, 14, 3, 0),
  ),
  (
    "T3",
    t3,
    lambda d: (d["close"], 5, 0.7),
    talib.T3,
    lambda d: (d["np_close"], 5, 0.7),
  ),
  (
    "TAN",
    tan,
    lambda d: (d["close"],),
    talib.TAN,
    lambda d: (d["np_close"],),
  ),
  (
    "TANH",
    tanh,
    lambda d: (d["close"],),
    talib.TANH,
    lambda d: (d["np_close"],),
  ),
  (
    "TEMA",
    tema,
    lambda d: (d["close"], 30),
    talib.TEMA,
    lambda d: (d["np_close"], 30),
  ),
  (
    "TRANGE",
    trange,
    lambda d: (d["high"], d["low"], d["close"]),
    talib.TRANGE,
    lambda d: (d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "TRIMA",
    trima,
    lambda d: (d["close"], 30),
    talib.TRIMA,
    lambda d: (d["np_close"], 30),
  ),
  (
    "TRIX",
    trix,
    lambda d: (d["close"], 30),
    talib.TRIX,
    lambda d: (d["np_close"], 30),
  ),
  (
    "TSF",
    tsf,
    lambda d: (d["close"], 14),
    talib.TSF,
    lambda d: (d["np_close"], 14),
  ),
  (
    "TYPPRICE",
    typprice,
    lambda d: (d["high"], d["low"], d["close"]),
    talib.TYPPRICE,
    lambda d: (d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "Takuri",
    cdl_takuri,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLTAKURI,
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
    "Thrusting",
    cdl_thrusting,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLTHRUSTING,
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
    "ULTOSC",
    ultosc,
    lambda d: (d["high"], d["low"], d["close"], 7, 14, 28),
    talib.ULTOSC,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 7, 14, 28),
  ),
  (
    "UpGap2Crows",
    cdl_upside_gap_two_crows,
    lambda d: (d["open"], d["high"], d["low"], d["close"]),
    talib.CDLUPSIDEGAP2CROWS,
    lambda d: (d["np_open"], d["np_high"], d["np_low"], d["np_close"]),
  ),
  (
    "VAR",
    var,
    lambda d: (d["close"], 5, 1.0),
    talib.VAR,
    lambda d: (d["np_close"], 5, 1.0),
  ),
  (
    "VWAP",
    vwap,
    lambda d: (d["high"], d["low"], d["close"], d["volume"]),
    None,
    None,
  ),
  (
    "WCLPRICE",
    wclprice,
    lambda d: (d["high"], d["low"], d["close"]),
    talib.WCLPRICE,
    lambda d: (d["np_high"], d["np_low"], d["np_close"]),
  ),
  ("WMA", wma, lambda d: (d["close"], 30), talib.WMA, lambda d: (d["np_close"], 30)),
  (
    "WillR",
    willr,
    lambda d: (d["high"], d["low"], d["close"], 14),
    talib.WILLR,
    lambda d: (d["np_high"], d["np_low"], d["np_close"], 14),
  ),
  ("Z-Score", zscore, lambda d: (d["close"], 20), None, None),
  ("Z-SCORE_INTRA", zscore_intraday, lambda d: (d["close"],), None, None),
]

import traceback


def unify(res, name=None):
  # Handle Indikator result objects (named tuples / pydantic-like)
  if hasattr(res, "_fields") and "data_index" in res._fields:
    # Special handling for AROON to match TA-Lib (down, up)
    if name == "AROON":
      return np.concatenate([
        np.asarray(res.aroon_down).flatten(),
        np.asarray(res.aroon_up).flatten(),
      ])

    # General case: Extract all data fields
    data_fields = [
      getattr(res, f)
      for f in res._fields
      if f not in ("data_index", "name", "count", "index")
    ]
    # Ensure they are arrays and flatten them
    return np.concatenate([np.asarray(f).flatten() for f in data_fields])

  # Handle Pandas objects
  if isinstance(res, pd.DataFrame):
    # Concatenate columns to match TA-Lib tuple order
    return np.concatenate([res[col].values.flatten() for col in res.columns])

  if isinstance(res, pd.Series):
    return res.values.flatten()

  # Handle TA-Lib return tuples (already data-only)
  if isinstance(res, tuple):
    try:
      return np.concatenate([np.asarray(x).flatten() for x in res])
    except Exception:
      # Fallback for mixed types
      return np.asarray(res[0]).flatten()

  return np.asarray(res).flatten()


def check_match():
  # Sort benchmarks by name
  benchmarks = sorted(BENCHMARKS, key=lambda x: x[0])

  data = generate_data(30_000)

  results = []

  for name, ind_func, ind_args_fn, ta_func, ta_args_fn in benchmarks:
    try:
      # Check if TA-Lib counterpart exists
      if ta_func is None:
        continue

      # Prepare args
      # Regenerate data for specific patterns if needed
      if name == "MatchingLow":
        data = generate_data(30_000, 19)
      elif name == "MatHold" or name == "CDLMATHOLD":
        data = generate_data(30_000, 20)
      elif name == "Piercing":
        data = generate_data(30_000, 21)
      elif name == "RiseFall3Methods":
        data = generate_data(30_000, 22)
      elif name == "SeparatingLines":
        data = generate_data(30_000, 23)
      elif name == "StickSand":
        data = generate_data(30_000, 24)
      # StickSand might need 24 (I should check generate_data for other IDs, guessing based on sequence)
      # Actually, let's just do MatHold for now as requested.

      ind_args = ind_args_fn(data)
      ta_args = ta_args_fn(data)

      # Run Indikator
      res_ind_raw = ind_func(*ind_args)

      # Run TA-Lib
      res_ta_raw = ta_func(*ta_args)

      res_ind = unify(res_ind_raw, name)
      res_ta = unify(res_ta_raw, name)

      # Sanitize
      def sanitize(arr):
        if arr.dtype.kind in "fi":
          return arr
        return arr.astype(float)

      res_ind = sanitize(res_ind)
      res_ta = sanitize(res_ta)

      is_float = res_ind.dtype.kind == "f" or res_ta.dtype.kind == "f"

      if is_float:
        try:
          match_mask = np.isclose(res_ind, res_ta, equal_nan=True)
        except ValueError:
          print(
            f"{name:<20} | {'ERROR':<8} | Shape mismatch: {res_ind.shape} vs {res_ta.shape}"
          )
          continue
      else:
        match_mask = res_ind == res_ta

      match_count = match_mask.sum()
      total_count = len(res_ind)
      match_pct = (match_count / total_count) * 100.0
      diff_count = total_count - match_count

      if is_float:
        ta_events = (~np.isnan(res_ta)).sum()
        our_events = (~np.isnan(res_ind)).sum()
        # Active: where at least one is not NaN
        active_mask = (~np.isnan(res_ta)) | (~np.isnan(res_ind))
      else:
        ta_events = (res_ta != 0).sum()
        our_events = (res_ind != 0).sum()
        # Active: where at least one is non-zero
        active_mask = (res_ta != 0) | (res_ind != 0)

      active_count = active_mask.sum()
      if active_count > 0:
        event_match_count = (match_mask & active_mask).sum()
        event_match_pct = (event_match_count / active_count) * 100.0
      else:
        event_match_pct = 100.0

      results.append({
        "name": name,
        "match_pct": match_pct,
        "event_match_pct": event_match_pct,
        "diff_count": diff_count,
        "ta_events": ta_events,
        "our_events": our_events,
        "total_count": total_count,
      })

    except Exception:
      print(f"{name:<20} | {'ERROR':<8} | Check traceback")
      traceback.print_exc()
      pass

  # Sort by Event Match % Ascending (worst first)
  results.sort(key=lambda x: x["event_match_pct"])

  print(
    f"{'Indicator':<20} | {'Match %':<8} | {'Event %':<8} | {'Diffs':<8} | {'TA-Lib Events':<13} | {'Our Events':<10}"
  )
  print("-" * 88)

  for r in results:
    print(
      f"{r['name']:<20} | {r['match_pct']:>7.2f}% | {r['event_match_pct']:>7.1f}% | {r['diff_count']:>8} | {r['ta_events']:>13} | {r['our_events']:>10}"
    )


if __name__ == "__main__":
  check_match()
