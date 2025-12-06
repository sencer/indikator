"""Intraday aggregation utilities.

This module provides functions for computing time-of-day based aggregations,
useful for indicators that need to account for intraday seasonality patterns.
"""

from collections.abc import Callable
from typing import cast

import pandas as pd
from pdval import Datetime, Index, Validated, validated

# Minimum samples required per time slot before calculating aggregate
_MIN_SAMPLES_PER_SLOT = 3

__all__ = ["intraday_aggregate", "intraday_stats"]


@validated
def intraday_aggregate(
  data: Validated[pd.Series, Index[Datetime]],
  agg_func: str | Callable[[pd.Series], float],
  lookback_days: int | None = None,
  min_samples: int = _MIN_SAMPLES_PER_SLOT,
) -> pd.Series:
  """Generic intraday aggregation by time-of-day.

  For each bar, aggregates historical values for that specific time of day
  using a custom aggregation function (mean, std, median, etc.).

  Args:
    data: Series with DatetimeIndex
    agg_func: Aggregation function name ("mean", "std") or callable
    lookback_days: Days to look back (None = all history)
    min_samples: Minimum samples required per time slot

  Returns:
    Series with aggregated values for each bar's time slot

  Raises:
    ValueError: If index not DatetimeIndex
  """
  # Cast index to DatetimeIndex for type checker
  dt_index = cast("pd.DatetimeIndex", data.index)

  # We need to group by time, but Series groupby is less flexible for temporary columns
  # So we'll convert to DataFrame temporarily for the operation
  df = data.to_frame(name="_value")
  df["_time_slot"] = dt_index.time

  # Filter to lookback period if specified
  if lookback_days is not None:
    cutoff_date = cast("pd.Timestamp", df.index[-1] - pd.Timedelta(days=lookback_days))
    lookback_data = df[df.index >= cutoff_date]
  else:
    lookback_data = df

  # Optimized: Use pandas expanding with groupby transform
  # Calculate expanding aggregate within each time slot, shifted to exclude current bar
  def expanding_agg_shifted(group: pd.Series) -> pd.Series:
    # Create expanding object
    exp = group.expanding(min_periods=min_samples)

    # Use optimized Cython implementations for common operations
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

    # Fallback to slow generic apply for custom functions
    # expanding().apply() calculates aggregate including current, so shift by 1
    func = cast("Callable[[pd.Series], float]", agg_func)
    return exp.apply(func, raw=False).shift(1)

  agg_values = lookback_data.groupby(  # pyright: ignore[reportUnknownMemberType]
    "_time_slot", group_keys=False
  )["_value"].transform(expanding_agg_shifted)

  # Reindex to match original data index
  return pd.Series(agg_values, index=data.index, dtype=float)


@validated
def intraday_stats(
  data: Validated[pd.Series, Index[Datetime]],
  lookback_days: int | None = None,
  min_samples: int = _MIN_SAMPLES_PER_SLOT,
) -> tuple[pd.Series, pd.Series]:
  """Compute both mean and std by time-of-day in a single pass.

  More efficient than calling intraday_aggregate twice when both
  statistics are needed (e.g., for z-score calculation).

  Args:
    data: Series with DatetimeIndex
    lookback_days: Days to look back (None = all history)
    min_samples: Minimum samples required per time slot

  Returns:
    Tuple of (mean_series, std_series)

  Raises:
    ValueError: If index not DatetimeIndex
  """
  # Cast index to DatetimeIndex for type checker
  dt_index = cast("pd.DatetimeIndex", data.index)

  df = data.to_frame(name="_value")
  df["_time_slot"] = dt_index.time

  # Filter to lookback period if specified
  if lookback_days is not None:
    cutoff_date = cast("pd.Timestamp", df.index[-1] - pd.Timedelta(days=lookback_days))
    lookback_data = df[df.index >= cutoff_date]
  else:
    lookback_data = df

  # Compute both mean and std in single groupby operation
  def expanding_mean_shifted(group: pd.Series) -> pd.Series:
    return group.expanding(min_periods=min_samples).mean().shift(1)

  def expanding_std_shifted(group: pd.Series) -> pd.Series:
    return group.expanding(min_periods=min_samples).std().shift(1)

  grouped = lookback_data.groupby(  # pyright: ignore[reportUnknownMemberType]
    "_time_slot", group_keys=False
  )["_value"]

  mean_values = grouped.transform(expanding_mean_shifted)  # pyright: ignore[reportUnknownMemberType]
  std_values = grouped.transform(expanding_std_shifted)  # pyright: ignore[reportUnknownMemberType]

  return (
    pd.Series(mean_values, index=data.index, dtype=float),
    pd.Series(std_values, index=data.index, dtype=float),
  )
