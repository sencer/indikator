"""Rolling Minimum and Maximum indicators."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._rolling_numba import (
  compute_max_numba,
  compute_maxindex_numba,
  compute_min_numba,
  compute_minindex_numba,
  compute_minmax_numba,
  compute_minmaxindex_numba,
  compute_sum_numba,
)

if TYPE_CHECKING:
  from numpy.typing import NDArray


@configurable
@validate
def min_val(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> pd.Series:
  """Rolling Minimum of a series.

  Args:
    data: Input series used for min calculation.
    period: The lookback window size (default: 30).

  Returns:
    pd.Series: Rolling minimum values.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_min_numba(input_arr, period)

  return pd.Series(result, index=data.index, name="min")


@configurable
@validate
def max_val(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> pd.Series:
  """Rolling Maximum of a series.

  Args:
    data: Input series used for max calculation.
    period: The lookback window size (default: 30).

  Returns:
    pd.Series: Rolling maximum values.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_max_numba(input_arr, period)

  return pd.Series(result, index=data.index, name="max")


@configurable
@validate
def min_index(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> pd.Series:
  """Rolling Index of Minimum value (relative to start of series).

  Returns the integer index (0-based) where the minimum value occurred.

  Args:
    data: Input series used for min calculation.
    period: The lookback window size (default: 30).

  Returns:
    pd.Series: Rolling index of minimum values.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_minindex_numba(input_arr, period)

  return pd.Series(result, index=data.index, name="min_index")


@configurable
@validate
def max_index(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> pd.Series:
  """Rolling Index of Maximum value (relative to start of series).

  Returns the integer index (0-based) where the maximum value occurred.

  Args:
    data: Input series used for max calculation.
    period: The lookback window size (default: 30).

  Returns:
    pd.Series: Rolling index of maximum values.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_maxindex_numba(input_arr, period)

  return pd.Series(result, index=data.index, name="max_index")


@configurable
@validate
def sum_val(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> pd.Series:
  """Rolling Sum of a series.

  Args:
    data: Input series used for sum calculation.
    period: The lookback window size (default: 30).

  Returns:
    pd.Series: Rolling sum values.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_sum_numba(input_arr, period)

  return pd.Series(result, index=data.index, name="sum")


@configurable
@validate
def minmax(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> tuple[pd.Series, pd.Series]:
  """Rolling Minimum and Maximum of a series.

  Returns:
    tuple[pd.Series, pd.Series]: (min, max) series.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False), # pyright: ignore
  )

  m, x = compute_minmax_numba(input_arr, period)

  return (
    pd.Series(m, index=data.index, name="min"),
    pd.Series(x, index=data.index, name="max"),
  )


@configurable
@validate
def minmaxindex(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> tuple[pd.Series, pd.Series]:
  """Rolling Minimum and Maximum index of a series.

  Returns indices (0-based) relative to start of series.

  Returns:
    tuple[pd.Series, pd.Series]: (min_index, max_index) series.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False), # pyright: ignore
  )

  mi, xi = compute_minmaxindex_numba(input_arr, period)

  return (
    pd.Series(mi, index=data.index, name="min_index"),
    pd.Series(xi, index=data.index, name="max_index"),
  )
