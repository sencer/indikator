"""APO (Absolute Price Oscillator) indicator module."""

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

from indikator._results import APOResult
from indikator.ema import ema
from indikator.sma import sma

if TYPE_CHECKING:
  from numpy.typing import NDArray


@configurable
@validate
def apo(
  data: Validated[pd.Series, Finite, NotEmpty],
  fast_period: Hyper[int, Ge[2]] = 12,
  slow_period: Hyper[int, Ge[2]] = 26,
  matype: int = 0,  # 0=SMA, 1=EMA
) -> APOResult:
  """Calculate Absolute Price Oscillator (APO).

  APO = FastMA - SlowMA

  Args:
    data: Input Series.
    fast_period: Fast MA period (default 12).
    slow_period: Slow MA period (default 26).
    matype: Moving Average type (0=SMA, 1=EMA). Default 0.

  Returns:
    APOResult
  """
  # TA-Lib logic:
  # If matype=0, use SMA. If 1, use EMA.

  if matype == 1:
    fast_ma = ema(data, period=fast_period).ema
    slow_ma = ema(data, period=slow_period).ema
  else:
    # Default to SMA
    fast_ma = sma(data, period=fast_period).sma
    slow_ma = sma(data, period=slow_period).sma

  apo_values = fast_ma - slow_ma
  apo_np = cast("NDArray[np.float64]", apo_values)

  return APOResult(index=data.index, apo=apo_np)
