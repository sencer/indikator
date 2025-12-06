from hipr import Ge, Hyper, Le
import pandas as pd
from pdval import Finite, Validated

MAX_NAN_RATIO: float

def sector_correlation(
  stock_data: Validated[pd.Series, Finite],
  sector_data: Validated[pd.Series, Finite] | None = ...,
  *,
  window: Hyper[int, Ge[2]] = ...,
  default_value: Hyper[float, Ge[-1.0], Le[1.0]] = ...,
) -> pd.Series: ...
