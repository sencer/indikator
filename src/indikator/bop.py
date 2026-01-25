from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import numpy as np
import pandas as pd

from indikator._bop_numba import compute_bop_numba
from indikator._results import BOPResult

if TYPE_CHECKING:
  from numpy.typing import NDArray


@configurable
@validate
def bop(
  open_: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
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
  open_np = cast("NDArray[np.float64]", open_.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  high_np = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_np = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  close_np = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  result = compute_bop_numba(open_np, high_np, low_np, close_np)

  return BOPResult(data_index=close.index, bop=result)
