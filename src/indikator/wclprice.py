"""WCLPRICE - Weighted Close Price indicator.

WCLPRICE = (High + Low + 2*Close) / 4
"""

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import pandas as pd

from indikator._price_transform_numba import compute_wclprice_numba
from indikator._results import WCLPRICEResult
from indikator.utils import to_numpy


@configurable
@validate
def wclprice(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
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
  h = to_numpy(high)
  low_np = to_numpy(low)
  c = to_numpy(close)

  result = compute_wclprice_numba(h, low_np, c)

  return WCLPRICEResult(data_index=high.index, wclprice=result)
