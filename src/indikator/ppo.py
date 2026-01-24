"""PPO (Percentage Price Oscillator) indicator module."""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._results import PPOResult
from indikator.ema import ema
from indikator.sma import sma

if TYPE_CHECKING:
  from numpy.typing import NDArray


@configurable
@validate
def ppo(
  data: Validated[pd.Series, Finite, NotEmpty],
  fast_period: Hyper[int, Ge[2]] = 12,
  slow_period: Hyper[int, Ge[2]] = 26,
  matype: int = 0,  # 0=SMA, 1=EMA
) -> PPOResult:
  """Calculate Percentage Price Oscillator (PPO).

  PPO = (FastMA - SlowMA) / SlowMA * 100

  Args:
    data: Input Series.
    fast_period: Fast MA period (default 12).
    slow_period: Slow MA period (default 26).
    matype: Moving Average type (0=SMA, 1=EMA). Default 0.

  Returns:
    PPOResult
  """
  if matype == 1:
    fast_ma = ema(data, period=fast_period).to_pandas()
    slow_ma = ema(data, period=slow_period).to_pandas()
  else:
    # Default to SMA
    fast_ma = sma(data, period=fast_period).to_pandas()
    slow_ma = sma(data, period=slow_period).to_pandas()

  # Avoid div by zero?
  # If slow_ma is 0, result is likely NaN or Inf.
  # Pandas handles division by zero by returning Inf or NaN.
  ppo_values = (fast_ma - slow_ma) / slow_ma * 100.0
  ppo_np = cast("NDArray[np.float64]", ppo_values.to_numpy(dtype=float, copy=False))

  return PPOResult(index=data.index, ppo=ppo_np)
