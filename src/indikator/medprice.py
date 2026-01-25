"""MEDPRICE - Median Price indicator.

MEDPRICE = (High + Low) / 2
"""

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._price_transform_numba import compute_medprice_numba
from indikator._results import MEDPRICEResult


@configurable
@validate
def medprice(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
) -> MEDPRICEResult:
  """Calculate Median Price.

  MEDPRICE = (High + Low) / 2

  Args:
    high: High prices
    low: Low prices

  Returns:
    MEDPRICEResult
  """
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_np = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  result = compute_medprice_numba(h, low_np)

  return MEDPRICEResult(data_index=high.index, medprice=result)
