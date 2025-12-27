"""Intraday aggregation utilities.

This module provides functions for computing time-of-day based aggregations,
useful for indicators that need to account for intraday seasonality patterns.
"""

from collections.abc import Callable
from typing import Literal, cast

import pandas as pd
from validated import (
  Datetime,
  Index,
  NonEmpty,
  Validated,
  validated,
)

from indikator._constants import DEFAULT_MIN_SAMPLES

# Optimized aggregation functions (use these for best performance)
AggFunc = Literal["mean", "std", "median", "min", "max"]

__all__ = ["intraday_aggregate", "intraday_stats"]


def _get_lookback_data(
  data: pd.Series, lookback_days: int | None
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
  """Internal helper to prepare DataFrame and filter by lookback."""
  dt_index = cast("pd.DatetimeIndex", data.index)
  df = data.to_frame(name="__indikator_value__")
  df["__indikator_time_slot__"] = dt_index.time

  if lookback_days is not None:
    cutoff_date = cast("pd.Timestamp", df.index[-1] - pd.Timedelta(days=lookback_days))
    lookback_data = df[df.index >= cutoff_date]
  else:
    lookback_data = df

  return lookback_data, dt_index


@validated
def intraday_aggregate(
  data: Validated[pd.Series, Index(Datetime), NonEmpty],
  agg_func: AggFunc | Callable[[pd.Series], float],
  lookback_days: int | None = None,
  min_samples: int = DEFAULT_MIN_SAMPLES,
) -> pd.Series:
  """Generic intraday aggregation by time-of-day.

  For each bar, aggregates historical values for that specific time of day
  using a custom aggregation function (mean, std, median, etc.).

  Args:
    data: Series with DatetimeIndex
    agg_func: Aggregation function name ("mean", "std") or callable
    lookback_days: Days to look back (None = all history)
    min_samples: Minimum observations required before calculating aggregate (NaN until met)

  Returns:
    Series with aggregated values for each bar's time slot
  """
  lookback_data, _ = _get_lookback_data(data, lookback_days)

  def expanding_agg_shifted(group: pd.Series) -> pd.Series:
    exp = group.expanding(min_periods=min_samples)

    if agg_func == "mean" or agg_func is pd.Series.mean:
      return exp.mean().shift(1)
    if agg_func == "std" or agg_func is pd.Series.std:
      return exp.std().shift(1)
    if agg_func == "median" or agg_func is pd.Series.median:
      return exp.median().shift(1)
    if agg_func == "max" or agg_func is pd.Series.max:
      return exp.max().shift(1)  # pyright: ignore[reportUnknownMemberType]
    if agg_func == "min" or agg_func is pd.Series.min:
      return exp.min().shift(1)

    func = cast("Callable[[pd.Series], float]", agg_func)
    return exp.apply(func, raw=False).shift(1)

  agg_values = lookback_data.groupby(  # pyright: ignore[reportUnknownMemberType]
    "__indikator_time_slot__", group_keys=False
  )["__indikator_value__"].transform(expanding_agg_shifted)

  return pd.Series(agg_values, index=data.index, dtype=float)


@validated
def intraday_stats(
  data: Validated[pd.Series, Index(Datetime), NonEmpty],
  lookback_days: int | None = None,
  min_samples: int = DEFAULT_MIN_SAMPLES,
) -> tuple[pd.Series, pd.Series]:
  """Compute both mean and std by time-of-day in a single pass.

  More efficient than calling intraday_aggregate twice when both
  statistics are needed (e.g., for z-score calculation).

  Args:
    data: Series with DatetimeIndex
    lookback_days: Days to look back (None = all history)
    min_samples: Minimum observations required before calculating aggregate (NaN until met)

  Returns:
    Tuple of (mean_series, std_series)
  """
  lookback_data, _ = _get_lookback_data(data, lookback_days)

  grouped = lookback_data.groupby(  # pyright: ignore[reportUnknownMemberType]
    "__indikator_time_slot__", group_keys=False
  )["__indikator_value__"]

  mean_values = grouped.transform(
    lambda g: g.expanding(min_periods=min_samples).mean().shift(1)
  )
  std_values = grouped.transform(
    lambda g: g.expanding(min_periods=min_samples).std().shift(1)
  )

  return (
    pd.Series(mean_values, index=data.index, dtype=float),
    pd.Series(std_values, index=data.index, dtype=float),
  )
