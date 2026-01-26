from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.bop import compute_bop_numba
from indikator.utils import to_numpy


@configurable
@validate
def bop(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Balance of Power (BOP).

  BOP = (Close - Open) / (High - Low)

  Args:
      open_: Open prices
      high: High prices
      low: Low prices
      close: Close prices

  Returns:
      IndicatorResult: Balance of Power values
  """
  open_np = to_numpy(open_)
  high_np = to_numpy(high)
  low_np = to_numpy(low)
  close_np = to_numpy(close)

  result = compute_bop_numba(open_np, high_np, low_np, close_np)

  return IndicatorResult(data_index=close.index, value=result, name="bop")
