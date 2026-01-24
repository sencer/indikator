"""MEDPRICE - Median Price indicator.

MEDPRICE = (High + Low) / 2
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

from indikator._price_transform_numba import compute_medprice_numba
from indikator._results import MEDPRICEResult


@configurable
@validate
def medprice(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
) -> MEDPRICEResult:
  """Calculate Median Price.

  MEDPRICE = (High + Low) / 2

  Args:
    high: High prices
    low: Low prices

  Returns:
    MEDPRICEResult
  """
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))
  l = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))

  result = compute_medprice_numba(h, l)

  return MEDPRICEResult(index=high.index, medprice=result)
