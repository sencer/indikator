"""Math Transform indicators."""

from datawarden import NotEmpty, Validated, validate
from nonfig import configurable
import pandas as pd

from indikator.numba.math_transform import (
  acos_impl,
  asin_impl,
  atan_impl,
  ceil_impl,
  ceil_impl_serial,
  cos_impl,
  cosh_impl,
  exp_impl,
  floor_impl,
  floor_impl_serial,
  ln_impl,
  log10_impl,
  sin_impl,
  sinh_impl,
  sqrt_impl,
  sqrt_impl_serial,
  tan_impl,
  tanh_impl,
)
from indikator.utils import to_numpy

# Constants
PARALLEL_THRESHOLD = 4096


# --- Public API ---


@configurable
@validate
def sin(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Trigonometric Sin."""
  arr = to_numpy(data)
  return pd.Series(sin_impl(arr), index=data.index, name="sin")


@configurable
@validate
def cos(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Trigonometric Cos."""
  arr = to_numpy(data)
  return pd.Series(cos_impl(arr), index=data.index, name="cos")


@configurable
@validate
def tan(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Trigonometric Tan."""
  arr = to_numpy(data)
  return pd.Series(tan_impl(arr), index=data.index, name="tan")


@configurable
@validate
def sinh(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Hyperbolic Sin."""
  arr = to_numpy(data)
  return pd.Series(sinh_impl(arr), index=data.index, name="sinh")


@configurable
@validate
def cosh(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Hyperbolic Cos."""
  arr = to_numpy(data)
  return pd.Series(cosh_impl(arr), index=data.index, name="cosh")


@configurable
@validate
def tanh(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Hyperbolic Tan."""
  arr = to_numpy(data)
  return pd.Series(tanh_impl(arr), index=data.index, name="tanh")


@configurable
@validate
def ceil(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Ceil."""
  arr = to_numpy(data)
  res = ceil_impl_serial(arr) if len(arr) < PARALLEL_THRESHOLD else ceil_impl(arr)
  return pd.Series(res, index=data.index, name="ceil")


@configurable
@validate
def floor(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Floor."""
  arr = to_numpy(data)
  res = floor_impl_serial(arr) if len(arr) < PARALLEL_THRESHOLD else floor_impl(arr)
  return pd.Series(res, index=data.index, name="floor")


@configurable
@validate
def exp(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Exponential."""
  arr = to_numpy(data)
  return pd.Series(exp_impl(arr), index=data.index, name="exp")


@configurable
@validate
def ln(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Natural Log."""
  arr = to_numpy(data)
  return pd.Series(ln_impl(arr), index=data.index, name="ln")


@configurable
@validate
def log10(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Log Base 10."""
  arr = to_numpy(data)
  return pd.Series(log10_impl(arr), index=data.index, name="log10")


@configurable
@validate
def sqrt(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Square Root."""
  arr = to_numpy(data)
  res = sqrt_impl_serial(arr) if len(arr) < PARALLEL_THRESHOLD else sqrt_impl(arr)
  return pd.Series(res, index=data.index, name="sqrt")


@configurable
@validate
def acos(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Arccos."""
  arr = to_numpy(data)
  return pd.Series(acos_impl(arr), index=data.index, name="acos")


@configurable
@validate
def asin(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Arcsin."""
  arr = to_numpy(data)
  return pd.Series(asin_impl(arr), index=data.index, name="asin")


@configurable
@validate
def atan(
  data: Validated[pd.Series[float], NotEmpty],
) -> pd.Series:
  """Vector Arctan."""
  arr = to_numpy(data)
  return pd.Series(atan_impl(arr), index=data.index, name="atan")
