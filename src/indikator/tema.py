"""Triple Exponential Moving Average (TEMA) indicator module.

TEMA further reduces lag compared to DEMA by using triple-smoothed EMAs.
"""

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

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import TEMAResult
from indikator._tema_numba import compute_tema_numba


@configurable
@validate
def tema(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> TEMAResult:
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
    TEMAResult with TEMA values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = tema(prices, period=5)
  """
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  tema_values = compute_tema_numba(values, period)

  return TEMAResult(data_index=data.index, tema=tema_values)
