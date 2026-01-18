"""Result types for multi-output indicators.

Uses dataclass with DataFrame-compatible interface.
Near-zero creation overhead while supporting `result.col`, `result["col"]`, and common DataFrame methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
  from collections.abc import Iterator

  import numpy as np
  from numpy.typing import NDArray


def _make_series(arr: NDArray[Any], index: pd.Index | None) -> pd.Series:
  """Create a Series from array with optional index."""
  return pd.Series(arr, index=index)


@dataclass(frozen=True, slots=True)
class ADXResult:
  """Result from ADX calculation."""

  adx: NDArray[np.float64]
  plus_di: NDArray[np.float64]
  minus_di: NDArray[np.float64]
  index: pd.Index | None = None

  @property
  def columns(self) -> list[str]:
    """DataFrame-compatible columns property."""
    return ["adx", "plus_di", "minus_di"]

  def __getitem__(self, key: str) -> pd.Series:
    """Support dict-like access, returns Series for compatibility."""
    if key == "adx":
      return _make_series(self.adx, self.index)
    if key == "plus_di":
      return _make_series(self.plus_di, self.index)
    if key == "minus_di":
      return _make_series(self.minus_di, self.index)
    raise KeyError(key)

  def __iter__(self) -> Iterator[str]:
    return iter(self.columns)

  def __len__(self) -> int:
    return len(self.adx)

  def dropna(self) -> pd.DataFrame:
    """Return DataFrame with NaN rows dropped."""
    return self.to_dataframe().dropna()

  def to_dataframe(self) -> pd.DataFrame:
    """Convert to DataFrame with proper index."""
    return pd.DataFrame(
      {"adx": self.adx, "plus_di": self.plus_di, "minus_di": self.minus_di},
      index=self.index,
    )


@dataclass(frozen=True, slots=True)
class AROONResult:
  """Result from AROON calculation."""

  aroon_up: NDArray[np.float64]
  aroon_down: NDArray[np.float64]
  aroon_osc: NDArray[np.float64]
  index: pd.Index | None = None

  @property
  def columns(self) -> list[str]:
    return ["aroon_up", "aroon_down", "aroon_osc"]

  def __getitem__(self, key: str) -> pd.Series:
    if key == "aroon_up":
      return _make_series(self.aroon_up, self.index)
    if key == "aroon_down":
      return _make_series(self.aroon_down, self.index)
    if key == "aroon_osc":
      return _make_series(self.aroon_osc, self.index)
    raise KeyError(key)

  def __iter__(self) -> Iterator[str]:
    return iter(self.columns)

  def __len__(self) -> int:
    return len(self.aroon_up)

  def dropna(self) -> pd.DataFrame:
    return self.to_dataframe().dropna()

  def to_dataframe(self) -> pd.DataFrame:
    return pd.DataFrame(
      {
        "aroon_up": self.aroon_up,
        "aroon_down": self.aroon_down,
        "aroon_osc": self.aroon_osc,
      },
      index=self.index,
    )


@dataclass(frozen=True, slots=True)
class StochResult:
  """Result from Stochastic Oscillator calculation."""

  stoch_k: NDArray[np.float64]
  stoch_d: NDArray[np.float64]
  index: pd.Index | None = None

  @property
  def columns(self) -> list[str]:
    return ["stoch_k", "stoch_d"]

  def __getitem__(self, key: str) -> pd.Series:
    if key == "stoch_k":
      return _make_series(self.stoch_k, self.index)
    if key == "stoch_d":
      return _make_series(self.stoch_d, self.index)
    raise KeyError(key)

  def __iter__(self) -> Iterator[str]:
    return iter(self.columns)

  def __len__(self) -> int:
    return len(self.stoch_k)

  def dropna(self) -> pd.DataFrame:
    return self.to_dataframe().dropna()

  def to_dataframe(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"stoch_k": self.stoch_k, "stoch_d": self.stoch_d}, index=self.index
    )


@dataclass(frozen=True, slots=True)
class MACDResult:
  """Result from MACD calculation."""

  macd: NDArray[np.float64]
  signal: NDArray[np.float64]
  histogram: NDArray[np.float64]
  index: pd.Index | None = None

  @property
  def columns(self) -> list[str]:
    return ["macd", "signal", "histogram"]

  def __getitem__(self, key: str) -> pd.Series:
    if key == "macd":
      return _make_series(self.macd, self.index)
    if key in {"signal", "macd_signal"}:
      return _make_series(self.signal, self.index)
    if key in {"histogram", "macd_histogram"}:
      return _make_series(self.histogram, self.index)
    raise KeyError(key)

  def __iter__(self) -> Iterator[str]:
    return iter(self.columns)

  def __len__(self) -> int:
    return len(self.macd)

  def dropna(self) -> pd.DataFrame:
    return self.to_dataframe().dropna()

  def to_dataframe(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"macd": self.macd, "signal": self.signal, "histogram": self.histogram},
      index=self.index,
    )


@dataclass(frozen=True, slots=True)
class BollingerResult:
  """Result from Bollinger Bands calculation."""

  bb_middle: NDArray[np.float64]
  bb_upper: NDArray[np.float64]
  bb_lower: NDArray[np.float64]
  bb_bandwidth: NDArray[np.float64]
  bb_percent: NDArray[np.float64]
  index: pd.Index | None = None

  @property
  def columns(self) -> list[str]:
    return ["bb_middle", "bb_upper", "bb_lower", "bb_bandwidth", "bb_percent"]

  def __getitem__(self, key: str) -> pd.Series:
    if key == "bb_middle":
      return _make_series(self.bb_middle, self.index)
    if key == "bb_upper":
      return _make_series(self.bb_upper, self.index)
    if key == "bb_lower":
      return _make_series(self.bb_lower, self.index)
    if key == "bb_bandwidth":
      return _make_series(self.bb_bandwidth, self.index)
    if key == "bb_percent":
      return _make_series(self.bb_percent, self.index)
    raise KeyError(key)

  def __iter__(self) -> Iterator[str]:
    return iter(self.columns)

  def __len__(self) -> int:
    return len(self.bb_middle)

  def dropna(self) -> pd.DataFrame:
    return self.to_dataframe().dropna()

  def to_dataframe(self) -> pd.DataFrame:
    return pd.DataFrame(
      {
        "bb_middle": self.bb_middle,
        "bb_upper": self.bb_upper,
        "bb_lower": self.bb_lower,
        "bb_bandwidth": self.bb_bandwidth,
        "bb_percent": self.bb_percent,
      },
      index=self.index,
    )


@dataclass(frozen=True, slots=True)
class OpeningRangeResult:
  """Result from Opening Range calculation."""

  or_high: NDArray[np.float64]
  or_low: NDArray[np.float64]
  or_mid: NDArray[np.float64]
  or_range: NDArray[np.float64]
  or_breakout: NDArray[np.int8]
  index: pd.Index | None = None

  @property
  def columns(self) -> list[str]:
    return ["or_high", "or_low", "or_mid", "or_range", "or_breakout"]

  def __getitem__(self, key: str) -> pd.Series:
    if key == "or_high":
      return _make_series(self.or_high, self.index)
    if key == "or_low":
      return _make_series(self.or_low, self.index)
    if key == "or_mid":
      return _make_series(self.or_mid, self.index)
    if key == "or_range":
      return _make_series(self.or_range, self.index)
    if key == "or_breakout":
      return _make_series(self.or_breakout, self.index)
    raise KeyError(key)

  def __iter__(self) -> Iterator[str]:
    return iter(self.columns)

  def __len__(self) -> int:
    return len(self.or_high)

  def dropna(self) -> pd.DataFrame:
    return self.to_dataframe().dropna()

  def to_dataframe(self) -> pd.DataFrame:
    return pd.DataFrame(
      {
        "or_high": self.or_high,
        "or_low": self.or_low,
        "or_mid": self.or_mid,
        "or_range": self.or_range,
        "or_breakout": self.or_breakout,
      },
      index=self.index,
    )
