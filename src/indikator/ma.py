"""Universal Moving Average wrapper."""

from typing import TYPE_CHECKING

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import Ge, Hyper, configurable
import pandas as pd

if TYPE_CHECKING:
  from indikator._results import MAResult
from indikator.dema import dema
from indikator.ema import ema
from indikator.kama import kama
from indikator.sma import sma
from indikator.t3 import t3
from indikator.tema import tema
from indikator.trima import trima
from indikator.wma import wma

# Type mapping for TA-Lib compatibility
MAType = {
  0: sma,
  1: ema,
  2: wma,
  3: dema,
  4: tema,
  5: trima,
  6: kama,
  7: None,  # MAMA (To be implemented)
  8: t3,
}


@configurable
@validate
def ma(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
  matype: Hyper[int, Ge[0]] = 0,
) -> "MAResult":
  """Universal Moving Average wrapper.

  Args:
    data: Input Series.
    period: Lookback period (default: 30)
    matype: Moving Average type (0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3)

  Returns:
    IndicatorResult: Result object (SMAResult, EMAResult, etc.)
  """
  ma_func = MAType.get(matype)
  if ma_func is None:
    if matype == 7:  # noqa: PLR2004
      from indikator.mesa import mama  # noqa: PLC0415

      return mama(data)  # Default mama parameters
    raise ValueError(f"Invalid matype: {matype}")

  # Note: T3 has different default/hyper for vfactor, but here we use period.
  return ma_func(data, period=period)
