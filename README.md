# Indikator

![CI](https://github.com/sencer/indikator/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/sencer/indikator/branch/master/graph/badge.svg)](https://app.codecov.io/github/sencer/indikator)
[![PyPI](https://img.shields.io/pypi/v/indikator)](https://pypi.org/project/indikator/)
[![Python](https://img.shields.io/pypi/pyversions/indikator)](https://pypi.org/project/indikator/)
[![License](https://img.shields.io/github/license/sencer/indikator)](https://github.com/sencer/indikator/blob/master/LICENSE)

**High-Performance Technical Indicators for Python**

`indikator` is a powerful, type-safe Python library for financial market analysis. It provides a comprehensive suite of technical indicators optimized with **Numba** for high performance, validated with **pdval**, and configurable via **hipr**.

## Key Features

*   🚀 **High Performance**: Critical calculations are JIT-compiled using `Numba` for near-C speeds.
*   🛡️ **Type-Safe & Validated**: Built with strict type checking (`basedpyright`) and runtime data validation (`pdval`).
*   ⚙️ **Configurable**: Flexible parameter management using `hipr`'s hierarchical configuration system.
*   🐼 **Pandas Integration**: Seamlessly works with pandas DataFrames and Series.
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
from indikator import atr

# Load your OHLCV data
data = pd.DataFrame({
    'high': [...],
    'low': [...],
    'close': [...]
})

# Calculate ATR with default parameters (window=14)
result = atr(data)

# Result is a DataFrame with 'atr' and 'true_range' columns
print(result.head())
```

### Configuration

You can override parameters directly or create reusable configurations using `.make()`.

```python
# 1. Direct override
result = atr(data, window=20)

# 2. Reusable configuration (Factory pattern)
# Create a specialized ATR calculator
fast_atr = atr.make(window=5)

# Apply it to multiple datasets
result1 = fast_atr(data1)
result2 = fast_atr(data2)
```

### Validation

`indikator` ensures your data is correct before calculation. It checks for:
*   Required columns (e.g., 'high', 'low', 'close')
*   Data types (numeric)
*   Data quality (non-empty, non-NaN where required)

```python
# This will raise a helpful error if 'high' column is missing
try:
    atr(data[['close', 'low']])
except ValueError as e:
    print(f"Validation Error: {e}")
```

## Available Indicators

| Indicator | Description |
|-----------|-------------|
| **ATR** | Average True Range (Volatility) |
| **Bollinger Bands** | Volatility bands based on SMA and standard deviation |
| **Churn Factor** | Volume efficiency measure |
| **Legs** | Zigzag/Swing point detection |
| **MACD** | Moving Average Convergence Divergence |
| **MFI** | Money Flow Index (Volume-weighted RSI) |
| **OBV** | On-Balance Volume |
| **Opening Range** | High/Low of the first N minutes |
| **Pivots** | Support/Resistance pivot points |
| **RSI** | Relative Strength Index |
| **RVOL** | Relative Volume (Standard & Intraday) |
| **Sector Correlation** | Correlation with a benchmark/sector |
| **Slope** | Linear regression slope |
| **VWAP** | Volume Weighted Average Price (Standard & Anchored) |
| **Z-Score** | Standard deviation from mean (Standard & Intraday) |

## Development

This project uses `uv` for dependency management and `poe` for task running.

### Setup

1.  Install `uv`:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  Clone and sync:
    ```bash
    git clone https://github.com/yourusername/indikator.git
    cd indikator
    uv sync
    ```
3.  Install pre-commit hooks:
    ```bash
    uv run pre-commit install
    ```

### Common Tasks

*   **Test**: `uv run pytest`
*   **Lint**: `uv run ruff check`
*   **Format**: `uv run ruff format`
*   **Type Check**: `uv run basedpyright`
*   **Run All Checks**: `uv run poe quality`

## License

MIT License. See `LICENSE` for details.
