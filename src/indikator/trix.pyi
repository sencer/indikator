"""Type stubs for TRIX indicator."""

from datawarden import Finite, NotEmpty, Validated
from nonfig import Ge, Hyper
from pandas import Series

def trix(
  data: Validated[Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = ...,
) -> Series:
  """Calculate TRIX (Triple Exponential Average).

  Args:
    data: Input Series (typically closing prices)
    period: EMA period (default: 14)

  Returns:
    Series with TRIX values (percentage)
  """
  ...

class Config:
  """Configuration for TRIX indicator."""

  period: Hyper[int, Ge[2]]

  def make(self) -> trix:
    """Create configured TRIX calculator."""
    ...
