"""PPO (Percentage Price Oscillator) indicator module."""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._results import IndicatorResult
from indikator.ema import ema
from indikator.sma import sma


@configurable
@validate
def ppo(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  fast_period: Hyper[int, Ge[2]] = 12,
  slow_period: Hyper[int, Ge[2]] = 26,
  matype: Hyper[int] = 0,  # 0=SMA, 1=EMA
) -> IndicatorResult:
  """Calculate Percentage Price Oscillator (PPO).

  PPO = (FastMA - SlowMA) / SlowMA * 100

  Args:
    data: Input Series.
    fast_period: Fast MA period (default 12).
    slow_period: Slow MA period (default 26).
    matype: Moving Average type (0=SMA, 1=EMA). Default 0.

  Returns:
    IndicatorResult
  """
  if matype == 1:
    fast_ma = ema(data, period=fast_period).value
    slow_ma = ema(data, period=slow_period).value
  else:
    # Default to SMA
    fast_ma = sma(data, period=fast_period).value
    slow_ma = sma(data, period=slow_period).value

  # Avoid div by zero?
  # If slow_ma is 0, result is likely NaN or Inf.
  # Pandas handles division by zero by returning Inf or NaN.
  # Numpy handles it by returning Inf or likely complaining if we don't watch out.
  # But assuming float inputs, it returns inf/nan.
  # We can silence warnings or let it be.
  with np.errstate(divide="ignore", invalid="ignore"):
    ppo_values = (fast_ma - slow_ma) / slow_ma * 100.0

  ppo_np = ppo_values

  return IndicatorResult(data_index=data.index, value=ppo_np, name="ppo")
