"""Pytest fixtures and configuration for indikator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def empty_df() -> pd.DataFrame:
  """Empty DataFrame with required columns."""
  return pd.DataFrame({"close": []})


@pytest.fixture
def single_value_df() -> pd.DataFrame:
  """DataFrame with single close price."""
  return pd.DataFrame({"close": [100.0]})


@pytest.fixture
def flat_prices_df() -> pd.DataFrame:
  """DataFrame with constant prices (no trend)."""
  return pd.DataFrame({"close": [100.0] * 10})


@pytest.fixture
def simple_uptrend_df() -> pd.DataFrame:
  """Simple uptrend: 100 -> 110 (10% move)."""
  return pd.DataFrame({"close": [100.0, 102.0, 105.0, 108.0, 110.0]})


@pytest.fixture
def simple_downtrend_df() -> pd.DataFrame:
  """Simple downtrend: 100 -> 90 (10% move)."""
  return pd.DataFrame({"close": [100.0, 98.0, 95.0, 92.0, 90.0]})


@pytest.fixture
def zigzag_pattern_df() -> pd.DataFrame:
  """Zigzag pattern: up, down, up."""
  return pd.DataFrame({
    "close": [
      100.0,
      105.0,
      110.0,  # Up 10%
      108.0,
      105.0,
      100.0,  # Down 9%
      102.0,
      105.0,
      110.0,  # Up 10%
    ]
  })


@pytest.fixture
def noisy_uptrend_df() -> pd.DataFrame:
  """Uptrend with small noise/wicks."""
  return pd.DataFrame({
    "close": [
      100.0,
      101.5,
      101.0,  # Small noise
      102.0,
      103.5,
      103.0,  # Small noise
      105.0,
      106.5,
      106.0,  # Small noise
      110.0,  # Final high
    ]
  })


@pytest.fixture
def higher_highs_df() -> pd.DataFrame:
  """Bullish structure: higher highs and higher lows."""
  return pd.DataFrame({
    "close": [
      100.0,
      110.0,  # First high
      105.0,  # Higher low (pullback)
      115.0,  # Higher high
      110.0,  # Higher low
      120.0,  # Higher high
    ]
  })


@pytest.fixture
def lower_lows_df() -> pd.DataFrame:
  """Bearish structure: lower lows and lower highs."""
  return pd.DataFrame({
    "close": [
      100.0,
      90.0,  # First low
      95.0,  # Lower high (pullback)
      85.0,  # Lower low
      90.0,  # Lower high
      80.0,  # Lower low
    ]
  })


@pytest.fixture
def trend_change_bull_to_bear_df() -> pd.DataFrame:
  """Trend change from bullish to bearish."""
  return pd.DataFrame({
    "close": [
      100.0,
      110.0,  # Bullish leg
      105.0,  # Pullback (higher low - still bullish)
      115.0,  # Higher high - confirms bullish
      110.0,  # Start of reversal
      100.0,  # Breaks previous low (105) - bearish!
      95.0,  # Lower low confirmed
    ]
  })


@pytest.fixture
def nan_prices_df() -> pd.DataFrame:
  """DataFrame with NaN values."""
  return pd.DataFrame({"close": [100.0, 105.0, np.nan, 110.0]})


@pytest.fixture
def inf_prices_df() -> pd.DataFrame:
  """DataFrame with infinite values."""
  return pd.DataFrame({"close": [100.0, 105.0, np.inf, 110.0]})


@pytest.fixture
def large_dataset_df() -> pd.DataFrame:
  """Large dataset for performance testing (10k bars)."""
  np.random.seed(42)
  # Generate random walk
  returns = np.random.normal(0.0001, 0.02, 10000)
  prices = 100.0 * np.exp(np.cumsum(returns))
  return pd.DataFrame({"close": prices})
