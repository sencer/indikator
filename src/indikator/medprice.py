"""MEDPRICE - Median Price indicator.

MEDPRICE = (High + Low) / 2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

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

  result = (h + l) / 2.0

  return MEDPRICEResult(index=high.index, medprice=result)
