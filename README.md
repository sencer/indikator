# Indikator

![CI](https://github.com/sencer/indikator/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/indikator)](https://pypi.org/project/indikator/)
[![Python](https://img.shields.io/pypi/pyversions/indikator)](https://pypi.org/project/indikator/)
[![License](https://img.shields.io/github/license/sencer/indikator)](https://github.com/sencer/indikator/blob/master/LICENSE)

**High-Performance Technical Indicators for Python**

`indikator` is a powerful, type-safe Python library for financial market analysis. It provides an extensive suite of technical indicators optimized with **Numba** for high performance, validated with **datawarden**, and configurable via **nonfig**.

## Key Features

*   🚀 **High Performance**: Critical calculations are JIT-compiled using `Numba` for near-C speeds, often outperforming TA-Lib in Python environments.
*   🛡️ **Type-Safe & Validated**: Built with strict type checking (`basedpyright`) and runtime data validation using `datawarden`.
*   ⚙️ **Configurable**: Flexible parameter management using `nonfig`'s hierarchical configuration system.
*   🐼 **Pandas Integration**: Seamlessly works with pandas DataFrames and Series. All results can be converted to pandas objects with `.to_pandas()`.
*   📦 **Modern Stack**: Managed with `uv`, linted with `ruff`, and tested with `pytest`.

## Installation

Install using `pip` or `uv`:

```bash
pip install indikator
# or
uv add indikator
```

## Usage

### Basic Usage

Indicators can be used directly as functions. They validate input data automatically.

```python
import pandas as pd
from indikator import rsi, sma

# Load your price data
prices = pd.Series([100, 102, 101, 103, 105, 104, 106], name="close")

# Calculate RSI (returns IndicatorResult)
result = rsi(prices, window=14)

# Convert to pandas Series
rsi_series = result.to_pandas()
print(rsi_series.head())
```

### Multi-Input Indicators

Indicators like ATR or VWAP take multiple Series or a DataFrame.

```python
from indikator import atr, vwap

# ATR takes separate high, low, close Series
result = atr(df['high'], df['low'], df['close'], period=14)
atr_series = result.to_pandas()

# VWAP takes high, low, close, and volume
vwap_result = vwap(df['high'], df['low'], df['close'], df['volume'])
vwap_series = vwap_result.to_pandas()
```

### Configuration (Factory Pattern)

You can create reusable configurations using `.Config().make()`. This is useful for building trading systems with fixed parameters.

```python
# Create a specialized ATR calculator
fast_atr = atr.Config(period=5).make()

# Apply it to multiple datasets
result1 = fast_atr(high1, low1, close1).to_pandas()
result2 = fast_atr(high2, low2, close2).to_pandas()
```

### Validation

`indikator` ensures your data is correct before calculation. It checks for finite values, non-empty data, and required columns.

```python
import numpy as np

# This will raise a ValidationError because of the infinity value
invalid_prices = pd.Series([100, np.inf, 102])
try:
    rsi(invalid_prices)
except ValueError as e:
    print(f"Validation Error: {e}")
```

## Available Indicators

`indikator` includes a comprehensive set of indicators, including most TA-Lib equivalents and modern intraday indicators.

| Category | Indicators |
|----------|------------|
| **Overlap Studies** | `sma`, `ema`, `wma`, `dema`, `tema`, `trima`, `kama`, `mama`, `t3`, `bollinger_bands`, `sar`, `midpoint`, `midprice` |
| **Momentum** | `rsi`, `stoch`, `macd`, `adx`, `cci`, `mfi`, `roc`, `willr`, `cmo`, `mom`, `ppo`, `ultosc` |
| **Volatility** | `atr`, `natr`, `trange`, `stddev`, `var`, `zscore` |
| **Volume** | `ad`, `adosc`, `obv`, `vwap`, `churn_factor`, `rvol` |
| **Intraday** | `atr_intraday`, `zscore_intraday`, `rvol_intraday`, `opening_range`, `pivots`, `vwap_anchored` |
| **Price Transform** | `avgprice`, `medprice`, `typprice`, `wclprice` |
| **Cycle/Math** | `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`, `ht_trendline`, `ht_trendmode`, `sin`, `cos`, `tan`, `sqrt`, `exp`, `log10` |
| **Pattern Recognition** | Over 60 candlestick patterns (`cdl_hammer`, `cdl_engulfing`, `cdl_doji`, etc.) |

*Check the documentation for the full list of over 150 functions.*

## License

MIT License. See `LICENSE` for details.
