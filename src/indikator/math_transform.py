"""Math Transform indicators."""

from __future__ import annotations

from datawarden import Finite, NotEmpty, Validated, validate
from nonfig import configurable
import numpy as np
import pandas as pd


@configurable
@validate
def sin(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Trigonometric Sin.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.sin(data), index=data.index, name="sin")


@configurable
@validate
def cos(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Trigonometric Cos.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.cos(data), index=data.index, name="cos")


@configurable
@validate
def tan(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Trigonometric Tan.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.tan(data), index=data.index, name="tan")


@configurable
@validate
def sinh(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Hyperbolic Sin.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.sinh(data), index=data.index, name="sinh")


@configurable
@validate
def cosh(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Hyperbolic Cos.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.cosh(data), index=data.index, name="cosh")


@configurable
@validate
def tanh(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Hyperbolic Tan.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.tanh(data), index=data.index, name="tanh")


@configurable
@validate
def ceil(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Ceil.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.ceil(data), index=data.index, name="ceil")


@configurable
@validate
def floor(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Floor.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.floor(data), index=data.index, name="floor")


@configurable
@validate
def exp(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Exponential.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.exp(data), index=data.index, name="exp")


@configurable
@validate
def ln(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Natural Log.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.log(data), index=data.index, name="ln")


@configurable
@validate
def log10(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Log Base 10.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.log10(data), index=data.index, name="log10")


@configurable
@validate
def sqrt(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Square Root.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.sqrt(data), index=data.index, name="sqrt")


@configurable
@validate
def acos(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Arccos.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.arccos(data), index=data.index, name="acos")


@configurable
@validate
def asin(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Arcsin.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.arcsin(data), index=data.index, name="asin")


@configurable
@validate
def atan(
  data: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
  """Vector Arctan.

  Args:
    data: Input series.

  Returns:
    pd.Series: Resulting series.
  """
  return pd.Series(np.arctan(data), index=data.index, name="atan")
