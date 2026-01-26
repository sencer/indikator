"""Triple Exponential Moving Average (TEMA) indicator module.

TEMA further reduces lag compared to DEMA by using triple-smoothed EMAs.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.tema import compute_tema_numba
from indikator.utils import to_numpy


@configurable
@validate
def tema(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> IndicatorResult:
  """Calculate Triple Exponential Moving Average (TEMA).

  TEMA minimizes lag using a combination of single, double, and triple
  smoothed EMAs with specific weights.

  Formula:
  TEMA = 3 * EMA1 - 3 * EMA2 + EMA3

  Where:
  - EMA1 = EMA(price, n)
  - EMA2 = EMA(EMA1, n)
  - EMA3 = EMA(EMA2, n)

  Interpretation:
  - Even faster response than DEMA
  - Best for capturing short-term trend changes
  - Can be combined with slower MAs for crossover strategies

  Features:
  - Fused Numba kernel: computes all three EMA stages in single loop
  - Values stay in registers, no intermediate arrays

  Args:
    data: Input price Series (typically close prices)
    period: Lookback period (default: 20)

  Returns:
    IndicatorResult with TEMA values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = tema(prices, period=5)
  """
  values = to_numpy(data)

  tema_values = compute_tema_numba(values, period)

  return IndicatorResult(data_index=data.index, value=tema_values, name="tema")
