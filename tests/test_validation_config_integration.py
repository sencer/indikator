from hipr import Ge, Hyper, Lt, configurable
import numpy as np
import pandas as pd
from pdval import Finite, Validated, validated
import pytest

# Define functions with different decorator orders


@configurable
@validated
def fn_correct_order(
  data: Validated[pd.Series, Finite],
  param: Hyper[float, Ge[0], Lt[1]] = 0.5,  # noqa: ARG001
):
  """Correct order: @configurable (top) wraps @validated (bottom).
  Config.make() should return a function that includes validation.
  """
  return data


class TestValidationConfigIntegration:
  """Integration tests for @configurable and @validated decorators."""

  def setup_method(self):
    self.valid_data = pd.Series([1.0, 2.0, 3.0])
    self.invalid_data = pd.Series([1.0, np.inf, 3.0])

  def test_correct_order_validates_input(self):
    """Test that @configurable @validated (top-down) preserves validation in made function."""
    made_fn = fn_correct_order.Config(param=0.3).make()

    # Valid data should pass
    result = made_fn(self.valid_data)
    pd.testing.assert_series_equal(result, self.valid_data)

    # Invalid data should raise ValueError
    with pytest.raises(ValueError, match="must be finite"):
      made_fn(self.invalid_data)

  def test_incorrect_order_fails_at_definition(self):
    """Test that @validated @configurable (top-down) fails at definition time.

    The @configurable decorator returns a ConfigurableIndicator object,
    and @validated cannot wrap non-function objects (inspect.signature fails).
    """
    with pytest.raises(ValueError, match="no signature found for builtin"):

      @validated
      @configurable
      def fn_incorrect_order(
        data: Validated[pd.Series, Finite],
        param: Hyper[float, Ge[0], Lt[1]] = 0.5,  # noqa: ARG001
      ):
        return data
