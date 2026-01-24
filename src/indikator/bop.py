from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import (
  validate,
)
from nonfig import configurable
import numpy as np

from indikator._bop_numba import compute_bop_numba
from indikator._results import BOPResult

if TYPE_CHECKING:
  from datawarden import (
    Finite,
    NotEmpty,
    Validated,
  )
  from numpy.typing import NDArray
  import pandas as pd


@configurable
@validate
def bop(
  open_: Validated[pd.Series, Finite, NotEmpty],
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> BOPResult:
  """Balance of Power (BOP).

  BOP = (Close - Open) / (High - Low)

  Args:
      open_: Open prices
      high: High prices
      low: Low prices
      close: Close prices

  Returns:
      BOPResult: Balance of Power values
  """
  open_np = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))
  high_np = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  low_np = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))
  close_np = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))

  result = compute_bop_numba(open_np, high_np, low_np, close_np)

  return BOPResult(index=close.index, bop=result)
