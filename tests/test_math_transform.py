import numpy as np
import pandas as pd
import pytest
import talib
from indikator.math_transform import (
  acos,
  asin,
  atan,
  ceil,
  cos,
  cosh,
  exp,
  floor,
  ln,
  log10,
  sin,
  sinh,
  sqrt,
  tan,
  tanh,
)

TRANSFORMS = [
  ("sin", sin, talib.SIN),
  ("cos", cos, talib.COS),
  ("tan", tan, talib.TAN),
  ("sinh", sinh, talib.SINH),
  ("cosh", cosh, talib.COSH),
  ("tanh", tanh, talib.TANH),
  ("ceil", ceil, talib.CEIL),
  ("floor", floor, talib.FLOOR),
  ("exp", exp, talib.EXP),
  ("ln", ln, talib.LN),
  ("log10", log10, talib.LOG10),
  ("sqrt", sqrt, talib.SQRT),
  ("acos", acos, talib.ACOS),
  ("asin", asin, talib.ASIN),
  ("atan", atan, talib.ATAN),
]


@pytest.mark.parametrize("name, func, talib_func", TRANSFORMS)
def test_math_transform_matches_talib(name, func, talib_func):
  np.random.seed(42)
  # Range -1 to 1 for trig safety (acos/asin)
  # Range > 0 for log/sqrt
  # Let's generate safe data suitable for ALL functions if possible,
  # or just uniform(0.1, 0.9) covers most except maybe tan singularity (rare in random float)
  data = pd.Series(np.random.uniform(0.1, 0.9, 100), name="data")

  result = func(data)
  expected = talib_func(data.values)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name=name), atol=1e-10
  )


def test_transform_with_nans():
  data = pd.Series([0.5, np.nan, 0.8])
  result = sin(data)
  # NumPy propagates NaNs by default
  assert np.isnan(result.iloc[1])
  assert not np.isnan(result.iloc[0])
