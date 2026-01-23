"""MAVP (Moving Average with Variable Period) indicator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

import numpy as np
import pandas as pd
from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable

from indikator._mavp_numba import compute_mavp_sma_numba

if TYPE_CHECKING:
  from numpy.typing import NDArray


@configurable
@validate
def mavp(
  data: Validated[pd.Series, Finite, NotEmpty],
  periods: Validated[pd.Series, Finite, NotEmpty],
  minperiod: Annotated[int, Hyper(Ge(2))] = 2,
  maxperiod: Annotated[int, Hyper(Ge(2))] = 30,
  matype: int = 0,
) -> pd.Series:
  """Moving Average with Variable Period.

  Calculates a moving average where the period varies per element.
  Currently supports SMA (Simple Moving Average) logic (matype=0).

  Args:
    data: Input price series.
    periods: Series containing the period to use for each element.
             Values are clamped between minperiod and maxperiod.
    minperiod: Minimum period allowed (default: 2).
    maxperiod: Maximum period allowed (default: 30).
    matype: Moving Average Type (0=SMA). Currently only 0 is supported.

  Returns:
    pd.Series: MAVP values.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  periods_arr = cast(
    "NDArray[np.float64]",
    periods.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Ensure lengths match
  if len(input_arr) != len(periods_arr):
    raise ValueError("data and periods must have the same length")

  # Dispatch based on matype
  # Currently only supporting SMA (0)
  if matype != 0:
    # Warn or fallback? TA-Lib behavior is usually to support multiple.
    # For now, we only implement optimized SMA.
    # Explicitly mention only SMA is supported in docs.
    pass

  result = compute_mavp_sma_numba(input_arr, periods_arr, minperiod, maxperiod)

  return pd.Series(result, index=data.index, name="mavp")
