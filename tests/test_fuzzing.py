"""Property-based fuzzing tests for all indicators."""

from typing import Literal

from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes
import numpy as np
import pandas as pd
import pytest

from indikator.atr import atr, atr_intraday
from indikator.rsi import rsi
from indikator.macd import macd
from indikator.bollinger import bollinger_bands
from indikator.mfi import mfi
from indikator.obv import obv
from indikator.vwap import vwap, vwap_anchored
from indikator.slope import slope
from indikator.zscore import zscore, zscore_intraday
from indikator.rvol import rvol, rvol_intraday
from indikator.churn_factor import churn_factor
from indikator.pivots import pivot_points
from indikator.legs import zigzag_legs
from indikator.sector_correlation import sector_correlation
from indikator.opening_range import opening_range


@st.composite
def ohlcv_data(draw, index_type="range"):
  """Generate valid OHLCV data constructively."""
  n = draw(st.integers(min_value=10, max_value=100))

  lows = np.array(
    draw(
      st.lists(
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
        min_size=n,
        max_size=n,
      )
    )
  )
  ranges = np.array(
    draw(
      st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        min_size=n,
        max_size=n,
      )
    )
  )
  highs = lows + ranges

  open_fracs = np.array(
    draw(
      st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=n, max_size=n
      )
    )
  )
  close_fracs = np.array(
    draw(
      st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=n, max_size=n
      )
    )
  )

  opens = lows + ranges * open_fracs
  closes = lows + ranges * close_fracs

  volumes = np.array(
    draw(
      st.lists(
        st.floats(min_value=0.1, max_value=1e6, allow_nan=False), min_size=n, max_size=n
      )
    )
  )

  if index_type == "datetime":
    index = pd.date_range("2020-01-01", periods=n, freq="D")
  else:
    index = pd.RangeIndex(n)

  return pd.DataFrame(
    {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
    index=index,
  )


@st.composite
def intraday_data(draw):
  """Generate intraday OHLCV data (multiple days, minute frequency)."""
  days = draw(st.integers(min_value=2, max_value=5))
  bars_per_day = 30  # Enough for opening range

  dates = []
  start_date = pd.Timestamp("2024-01-01")
  for i in range(days):
    day_start = start_date + pd.Timedelta(days=i) + pd.Timedelta(hours=9, minutes=30)
    day_dates = pd.date_range(start=day_start, periods=bars_per_day, freq="1min")
    dates.extend(day_dates)

  n = len(dates)

  # Generate price data similar to ohlcv_data
  lows = np.array(
    draw(
      st.lists(
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
        min_size=n,
        max_size=n,
      )
    )
  )
  ranges = np.array(
    draw(
      st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        min_size=n,
        max_size=n,
      )
    )
  )
  highs = lows + ranges

  open_fracs = np.array(
    draw(
      st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=n, max_size=n
      )
    )
  )
  close_fracs = np.array(
    draw(
      st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=n, max_size=n
      )
    )
  )

  opens = lows + ranges * open_fracs
  closes = lows + ranges * close_fracs
  volumes = np.array(
    draw(
      st.lists(
        st.floats(min_value=100.0, max_value=1e6, allow_nan=False),
        min_size=n,
        max_size=n,
      )
    )
  )

  return pd.DataFrame(
    {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
    index=pd.DatetimeIndex(dates),
  )


# --- DataFrame Indicators (OHLCV) ---


@settings(max_examples=10, deadline=None)
@given(data=ohlcv_data(), window=st.integers(min_value=2, max_value=50))
@pytest.mark.parametrize(
  "func, col_check",
  [
    (atr, "atr"),
    (mfi, "mfi"),
  ],
)
def test_ohlcv_window_indicators(func, col_check, data, window):
  """Fuzz test for indicators taking DataFrame + window."""
  result = func(data, window=window)
  assert isinstance(result, pd.DataFrame)
  assert col_check in result.columns
  assert len(result) == len(data)
  if not result[col_check].isna().all():
    assert np.isfinite(result[col_check].dropna()).all()


@settings(max_examples=10, deadline=None)
@given(data=ohlcv_data())
def test_obv_fuzz(data):
  """Fuzz test for OBV (no window)."""
  result = obv(data)
  assert "obv" in result.columns
  assert len(result) == len(data)


@settings(max_examples=10, deadline=None)
@given(data=ohlcv_data())
def test_churn_factor_fuzz(data):
  """Fuzz test for Churn Factor."""
  result = churn_factor(data)
  assert "churn_factor" in result.columns
  assert len(result) == len(data)


# --- Series Indicators ---


@settings(max_examples=10, deadline=None)
@given(data=ohlcv_data(), window=st.integers(min_value=2, max_value=50))
@pytest.mark.parametrize(
  "func",
  [
    rsi,
    slope,
    zscore,
    rvol,
  ],
)
def test_series_window_indicators(func, data, window):
  """Fuzz test for indicators taking Series + window."""
  # Use 'close' for most, 'volume' for rvol
  series = data["volume"] if func == rvol else data["close"]
  result = func(series, window=window)
  assert len(result) == len(data)


@settings(max_examples=10, deadline=None)
@given(data=ohlcv_data(), window=st.integers(min_value=2, max_value=50))
def test_bollinger_fuzz(data, window):
  result = bollinger_bands(data["close"], window=window)
  assert "bb_upper" in result.columns
  assert len(result) == len(data)


# --- Intraday & Time-based Indicators ---


@settings(max_examples=10, deadline=None)
@given(data=intraday_data())
def test_intraday_indicators(data):
  """Fuzz test for intraday variants."""
  # ATR Intraday
  res_atr = atr_intraday(data, min_samples=2)
  assert "atr_intraday" in res_atr.columns

  # Z-Score Intraday
  res_z = zscore_intraday(data["close"], min_samples=2)
  assert len(res_z) == len(data)

  # RVOL Intraday
  res_rvol = rvol_intraday(data["volume"], min_samples=2)
  assert len(res_rvol) == len(data)

  # Opening Range
  res_or = opening_range(data, minutes=15)  # using 15m since we generate 30m per day
  assert "or_high" in res_or.columns


@settings(max_examples=10, deadline=None)
@given(data=intraday_data())
@pytest.mark.parametrize("method", ["standard", "fibonacci", "woodie", "camarilla"])
def test_pivots_fuzz(data, method):
  """Fuzz test for Pivot Points with all methods."""
  # Need to specify period that makes sense for the data.
  # Our intraday data spans days, so period="D" works.
  try:
    result = pivot_points(data, method=method, period="D")
    assert "pp" in result.columns
  except ValueError as e:
    # If generated data covers less than one period boundary
    if "period" not in str(e):
      raise e


# --- Special Indicators ---


@settings(max_examples=10, deadline=None)
@given(
  data=ohlcv_data(),
  fast=st.integers(min_value=2, max_value=20),
  slow=st.integers(min_value=21, max_value=50),
  signal=st.integers(min_value=2, max_value=15),
)
def test_macd_fuzz(data, fast, slow, signal):
  """Fuzz test for MACD."""
  series = data["close"]
  if fast >= slow:
    return
  result = macd(series, fast_period=fast, slow_period=slow, signal_period=signal)
  assert "macd" in result.columns


@settings(max_examples=10, deadline=None)
@given(data=ohlcv_data(index_type="datetime"))
def test_vwap_fuzz(data):
  """Fuzz test for VWAP."""
  result = vwap(data)
  assert "vwap" in result.columns


@settings(max_examples=10, deadline=None)
@given(
  data=ohlcv_data(index_type="datetime"),
  anchor_idx=st.integers(min_value=0, max_value=9),
)
def test_vwap_anchored_fuzz(data, anchor_idx):
  """Fuzz test for Anchored VWAP."""
  if anchor_idx >= len(data):
    anchor_idx = 0
  result = vwap_anchored(data, anchor_index=anchor_idx)
  assert "vwap_anchored" in result.columns


@settings(max_examples=10, deadline=None)
@given(
  data=ohlcv_data(),
  threshold=st.floats(min_value=0.001, max_value=0.1),
  min_dist=st.floats(min_value=0.001, max_value=0.1),
)
def test_zigzag_legs_fuzz(data, threshold, min_dist):
  """Fuzz test for ZigZag Legs."""
  result = zigzag_legs(data["close"], threshold=threshold, min_distance_pct=min_dist)
  assert len(result) == len(data)


@settings(max_examples=10, deadline=None)
@given(data=ohlcv_data())
def test_sector_correlation_fuzz(data):
  """Fuzz test for Sector Correlation."""
  stock = data["close"]
  # Create a sector series that is somewhat correlated but maybe slightly different length or index
  # For fuzzing, just using a modified version of stock is fine to test mechanics
  sector = stock * np.random.uniform(0.9, 1.1, len(stock))

  result = sector_correlation(stock, sector)
  assert len(result) == len(data)

  # Test with missing sector
  result_none = sector_correlation(stock, None)
  assert (result_none == 0.0).all()
