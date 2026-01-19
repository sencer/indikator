"""TYPPRICE - Typical Price indicator.

TYPPRICE = (High + Low + Close) / 3
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

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

  result = (h + l + c) / 3.0

  return TYPPRICEResult(index=high.index, typprice=result)
