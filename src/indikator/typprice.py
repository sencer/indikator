"""TYPPRICE - Typical Price indicator.

TYPPRICE = (High + Low + Close) / 3
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import validate
from nonfig import configurable
import numpy as np

if TYPE_CHECKING:
  from datawarden import Finite, NotEmpty, Validated
  from numpy.typing import NDArray
  import pandas as pd

from indikator._price_transform_numba import compute_typprice_numba
from indikator._results import TYPPRICEResult


@configurable
@validate
def typprice(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
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
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))

  result = compute_typprice_numba(h, l, c)

  return TYPPRICEResult(index=high.index, typprice=result)
