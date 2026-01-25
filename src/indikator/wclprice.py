"""WCLPRICE - Weighted Close Price indicator.

WCLPRICE = (High + Low + 2*Close) / 4
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._price_transform_numba import compute_wclprice_numba
from indikator._results import WCLPRICEResult


@configurable
@validate
def wclprice(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
) -> WCLPRICEResult:
  """Calculate Weighted Close Price.

  WCLPRICE = (High + Low + 2*Close) / 4

  Args:
    high: High prices
    low: Low prices
    close: Close prices

  Returns:
    WCLPRICEResult
  """
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  low_np = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))

  result = compute_wclprice_numba(h, low_np, c)

  return WCLPRICEResult(index=high.index, wclprice=result)
