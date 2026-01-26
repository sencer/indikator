"""Math Transform indicators."""

from typing import TYPE_CHECKING, cast

from typing import TYPE_CHECKING, cast

from datawarden import NotEmpty, Validated, validate
from nonfig import configurable
from numba import jit
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from indikator.utils import to_numpy


# --- Optimized Kernels ---


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _sin_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.sin(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _cos_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.cos(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _tan_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.tan(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _sinh_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.sinh(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _cosh_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.cosh(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _tanh_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.tanh(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _ceil_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.ceil(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _floor_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.floor(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _exp_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.exp(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _ln_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.log(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _log10_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.log10(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _sqrt_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.sqrt(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _acos_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.arccos(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _asin_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.arcsin(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _atan_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.arctan(data)


# --- Public API ---


@configurable
@validate
def sin(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Trigonometric Sin."""
  arr = to_numpy(data)
  return pd.Series(_sin_impl(arr), index=data.index, name="sin")


@configurable
@validate
def cos(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Trigonometric Cos."""
  arr = to_numpy(data)
  return pd.Series(_cos_impl(arr), index=data.index, name="cos")


@configurable
@validate
def tan(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Trigonometric Tan."""
  arr = to_numpy(data)
  return pd.Series(_tan_impl(arr), index=data.index, name="tan")


@configurable
@validate
def sinh(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Hyperbolic Sin."""
  arr = to_numpy(data)
  return pd.Series(_sinh_impl(arr), index=data.index, name="sinh")


@configurable
@validate
def cosh(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Hyperbolic Cos."""
  arr = to_numpy(data)
  return pd.Series(_cosh_impl(arr), index=data.index, name="cosh")


@configurable
@validate
def tanh(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Hyperbolic Tan."""
  arr = to_numpy(data)
  return pd.Series(_tanh_impl(arr), index=data.index, name="tanh")


@configurable
@validate
def ceil(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Ceil."""
  arr = to_numpy(data)
  return pd.Series(_ceil_impl(arr), index=data.index, name="ceil")


@configurable
@validate
def floor(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Floor."""
  arr = to_numpy(data)
  return pd.Series(_floor_impl(arr), index=data.index, name="floor")


@configurable
@validate
def exp(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Exponential."""
  arr = to_numpy(data)
  return pd.Series(_exp_impl(arr), index=data.index, name="exp")


@configurable
@validate
def ln(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Natural Log."""
  arr = to_numpy(data)
  return pd.Series(_ln_impl(arr), index=data.index, name="ln")


@configurable
@validate
def log10(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Log Base 10."""
  arr = to_numpy(data)
  return pd.Series(_log10_impl(arr), index=data.index, name="log10")


@configurable
@validate
def sqrt(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Square Root."""
  arr = to_numpy(data)
  return pd.Series(_sqrt_impl(arr), index=data.index, name="sqrt")


@configurable
@validate
def acos(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Arccos."""
  arr = to_numpy(data)
  return pd.Series(_acos_impl(arr), index=data.index, name="acos")


@configurable
@validate
def asin(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Arcsin."""
  arr = to_numpy(data)
  return pd.Series(_asin_impl(arr), index=data.index, name="asin")


@configurable
@validate
def atan(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Arctan."""
  arr = to_numpy(data)
  return pd.Series(_atan_impl(arr), index=data.index, name="atan")
