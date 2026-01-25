import numpy as np
import pandas as pd
import pytest
import talib

from indikator.beta import beta, beta_statistical
from indikator.correl import correl


@pytest.mark.parametrize("period", [5, 14, 30])
def test_beta_matches_talib(period):
  np.random.seed(42)
  # TA-Lib BETA calculates returns internally?
  # Yes, TA-Lib BETA(high, low) = Cov(ROCP(high), ROCP(low)) / Var(ROCP(low))
  # where low is "benchmark" (independent, X) and high is "stock" (dependent, Y).
  # Wait, checking TA-Lib docs: BETA(real0, real1, timeperiod=5) returns real
  # Standard def: Beta of real0 relative to real1?
  # Usually Beta(Stock, Market).
  # TA-Lib: BETA(real0, real1) -> Cov(real0, real1) / Var(real1)
  # BUT TA-Lib documentation says "BETA function calculates the Beta ... of a stock's returns relative to a market's returns."
  # So it accepts PRICES and computes returns internally?
  # Let's verify with our implementation 'beta' which does exactly that.

  # We treat 'high' as dependent (Stock, Y) and 'low' as independent (Market, X).
  # TA-Lib: BETA(y, x) -> ?

  high = pd.Series(np.random.randn(100).cumsum() + 100, name="high")  # Stock
  low = pd.Series(np.random.randn(100).cumsum() + 100, name="low")  # Market

  # Our 'beta' function docstring says: x=Independent (Market), y=Dependent (Stock).

  # Let's see what TA-Lib expects: TA-Lib(real0, real1).
  # Convention usually is (asset, benchmark).
  # Result = Cov(asset_ret, bench_ret) / Var(bench_ret).

  result = beta(x=low, y=high, period=period)

  # TA-Lib call order: BETA(real0, real1) -> Cov/Var(real0).
  # We want Beta of 'high' (Stock) relative to 'low' (Market).
  # Denominator variance should be Market ('low').
  # So we must pass 'low' as first argument to TA-Lib.

  expected = talib.BETA(low.values, high.values, timeperiod=period)

  # Note: TA-Lib usually produces NaNs for the first (period + 1) bars
  # because of ROCP (1 bar) + Rolling (period bars).

  pd.testing.assert_series_equal(
    result.to_pandas(), pd.Series(expected, index=high.index, name="beta"), atol=1e-10
  )


@pytest.mark.parametrize("period", [5, 14, 30])
def test_correl_matches_talib(period):
  np.random.seed(42)
  # CORREL in TA-Lib: "Pearson's Correlation Coefficient (r)"
  # TA-Lib: CORREL(real0, real1, timeperiod=30)
  # It takes raw prices or returns?
  # TA-Lib CORREL docs say: "Pearson's Correlation Coefficient (r)".
  # Usually correl is done on prices directly or returns?
  # TA-Lib implementation usually does NOT internal ROCP for CORREL.
  # It works on the input arrays directly.
  # Our 'correl' implementation works on raw inputs.

  high = pd.Series(np.random.randn(100).cumsum(), name="high")
  low = pd.Series(np.random.randn(100).cumsum(), name="low")

  result = correl(x=low, y=high, period=period)

  # Order for correl is symmetric so (high, low) vs (low, high) shouldn't matter for value.
  expected = talib.CORREL(high.values, low.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.to_pandas(), pd.Series(expected, index=high.index, name="correl"), atol=1e-10
  )


def test_beta_statistical_raw_inputs():
  """Verify beta_statistical works on raw inputs (no implicit ROCP)."""
  np.random.seed(42)
  # If we feed returns directly to beta_statistical, it should match beta() output provided we feed returns to it.

  high = pd.Series(np.random.randn(100).cumsum() + 100)
  low = pd.Series(np.random.randn(100).cumsum() + 100)
  period = 10

  # Calculate returns manually
  high_ret = high.pct_change()
  low_ret = low.pct_change()

  # beta_statistical on returns
  res_stat = beta_statistical(x=low_ret.fillna(0), y=high_ret.fillna(0), period=period)

  # beta() on prices (should match substantially, ignoring how NaNs are handled at start)
  res_beta = beta(x=low, y=high, period=period)

  # Compare stable region (after period + 1)
  pd.testing.assert_series_equal(
    res_stat.to_pandas().iloc[period + 2 :],
    res_beta.to_pandas().iloc[period + 2 :],
    atol=1e-10,
  )
