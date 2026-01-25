"""AVGPRICE - Average Price indicator.

AVGPRICE = (Open + High + Low + Close) / 4
"""

from typing import TYPE_CHECKING, cast

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._price_transform_numba import compute_avgprice_numba
from indikator._results import AVGPRICEResult


@configurable
@validate
def avgprice(
  open: Validated[pd.Series[float], Finite, NotEmpty],
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> AVGPRICEResult:
  """Calculate Average Price.

  AVGPRICE = (Open + High + Low + Close) / 4

  Args:
    open: Open prices
    high: High prices
    low: Low prices
    close: Close prices

  Returns:
    AVGPRICEResult
  """
  o = cast("NDArray[np.float64]", open.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  h = cast("NDArray[np.float64]", high.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  low_arr = cast("NDArray[np.float64]", low.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]
  c = cast("NDArray[np.float64]", close.to_numpy(dtype=np.float64, copy=False))  # pyright: ignore[reportUnknownMemberType]

  result = compute_avgprice_numba(o, h, low_arr, c)

  return AVGPRICEResult(data_index=high.index, avgprice=result)
