"""TYPPRICE - Typical Price indicator.

TYPPRICE = (High + Low + Close) / 3
"""

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._price_transform_numba import compute_typprice_numba
from indikator._results import TYPPRICEResult


@configurable
@validate
def typprice(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> TYPPRICEResult:
  """Calculate Typical Price.

  TYPPRICE = (High + Low + Close) / 3

  Args:
    high: High prices
    low: Low prices
    close: Close prices

  Returns:
    TYPPRICEResult
  """
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_np = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  result = compute_typprice_numba(h, low_np, c)

  return TYPPRICEResult(data_index=high.index, typprice=result)
