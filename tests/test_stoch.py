from datawarden import config
import numpy as np
import pandas as pd
import talib

from indikator.stoch import stoch


def test_reproducible():
  print("Starting reproduction...")
  np.random.seed(42)
  high = pd.Series(np.random.uniform(105, 110, 100))
  low = pd.Series(np.random.uniform(95, 100, 100))
  close = pd.Series(np.random.uniform(95, 110, 100))

  k_period = 5
  k_slow = 3
  d_period = 3

  print("Running stoch...")
  with config.Overrides(skip_validation=True):
    result = stoch(
      high, low, close, k_period=k_period, k_slowing=k_slow, d_period=d_period
    )

  df = result.to_pandas()
  k_res = df["stoch_k"]

  print("Running TA-lib...")
  exp_k, exp_d = talib.STOCH(
    high.values,
    low.values,
    close.values,
    fastk_period=k_period,
    slowk_period=k_slow,
    slowk_matype=0,
    slowd_period=d_period,
    slowd_matype=0,
  )

  print("\nComparison:")
  print("Index | Our %K | TA-lib %K")
  for i in range(15):
    print(f"{i:5d} | {k_res.iloc[i]:8.4f} | {exp_k[i]:8.4f}")

  try:
    pd.testing.assert_series_equal(
      k_res,
      pd.Series(exp_k, index=high.index, name="stoch_k"),
      check_exact=False,
      rtol=1e-5,
    )
    print("\nPASS")
  except AssertionError as e:
    print("\nFAIL")
    print(e)


if __name__ == "__main__":
  reproducible_test()
