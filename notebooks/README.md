# Indicator Visualizations

This folder contains interactive Jupyter notebooks demonstrating each indicator with visual examples, hyperparameter effects, and real-world use cases.

## Notebooks

### 1. [zscore_visualization.ipynb](zscore_visualization.ipynb)
**Z-Score Indicator** - Detects price/volume anomalies

**Scenarios covered:**
- Price spike detection
- Window parameter effect (sensitivity vs stability)
- Volume analysis for breakout detection
- Mean reversion trading strategy

**Key parameters:**
- `window`: Rolling window size (larger = smoother, better anomaly detection)
- `epsilon`: Division by zero protection

**Best for:** Overbought/oversold detection, anomaly identification, mean reversion strategies

---

### 2. [slope_visualization.ipynb](slope_visualization.ipynb)
**Linear Regression Slope** - Trend direction and strength

**Scenarios covered:**
- Trend direction detection (up/down/consolidation)
- Window parameter effect on smoothness
- Trend acceleration and deceleration
- Divergence detection (price vs momentum)
- Trend-following trading strategy

**Key parameters:**
- `window`: Window size (larger = smoother trends)

**Best for:** Trend following, momentum analysis, divergence trading

**Performance:** 50-100x faster than scipy (Numba-optimized)

---

### 3. [churn_factor_visualization.ipynb](churn_factor_visualization.ipynb)
**Churn Factor** - Volume / (High - Low)

**Scenarios covered:**
- Support/resistance detection (high churn zones)
- Fill strategy comparison for zero-range bars

**Key parameters:**
- `fill_strategy`: How to handle zero-range bars (`nan`, `forward_fill`, `zero`)
- `epsilon`: Division by zero protection

**Best for:** Accumulation/distribution detection, support/resistance identification

---

### 4. [sector_correlation_visualization.ipynb](sector_correlation_visualization.ipynb)
**Sector Correlation** - Stock vs Sector ETF correlation

**Scenarios covered:**
- High correlation (stock moves with sector)
- Decoupling detection (stock goes its own way)
- Window parameter effect on correlation stability

**Key parameters:**
- `window`: Rolling correlation window
- `default_value`: Value when sector data missing

**Best for:** Risk management, diversification analysis, identifying stock-specific opportunities

---

### 5. [rvol_visualization.ipynb](rvol_visualization.ipynb)
**Relative Volume (RVOL)** - Current Volume / Average Volume

**Scenarios covered:**
- Breakout volume detection
- Low volume consolidation warnings
- Window parameter effect on sensitivity
- Real trading strategy example

**Key parameters:**
- `window`: Average volume calculation window

**Interpretation:**
- RVOL > 3: Exceptional volume
- RVOL 2-3: High volume
- RVOL < 0.5: Very low volume (avoid)

**Best for:** Breakout confirmation, liquidity assessment, volume-based filtering

---

### 6. [zigzag_visualization.ipynb](zigzag_visualization.ipynb)
**ZigZag Legs** - Market structure tracking (Higher Highs/Lower Lows)

**Scenarios covered:**
- Multiple market scenarios (trends, consolidation, reversals)
- Parameter sensitivity analysis
- Real-world pattern examples

**Key parameters:**
- `threshold`: Minimum % change for reversal
- `min_distance_pct`: Noise filter
- `confirmation_bars`: Reversal confirmation period

**Best for:** Elliott Wave analysis, structure-based trading, trend change detection

---

## Running the Notebooks

### Prerequisites
```bash
# Ensure you're in the project directory
cd /path/to/indikator

# Install with notebook dependencies
uv pip install -e ".[dev]"

# Or install jupyter separately
uv pip install jupyter matplotlib scipy
```

### Launch Jupyter
```bash
jupyter notebook notebooks/
```

Or use JupyterLab:
```bash
jupyter lab notebooks/
```

### Running Individual Notebooks
```bash
jupyter notebook notebooks/zscore_visualization.ipynb
```

## Notebook Features

Each notebook includes:
- ✅ **Clear explanations** of what the indicator measures
- ✅ **Multiple scenarios** showing different market conditions
- ✅ **Parameter comparisons** to understand hyperparameter effects
- ✅ **Visual plots** with proper labeling and color coding
- ✅ **Real-world examples** applicable to actual trading
- ✅ **Key takeaways** summarizing best practices
- ✅ **Interpretation guides** for indicator values

## Tips for Exploration

1. **Start with zscore or rvol** - These are the most intuitive
2. **Run cells sequentially** - Each builds on previous examples
3. **Modify parameters** - Change window sizes, thresholds to experiment
4. **Combine indicators** - Try using multiple indicators together
5. **Use your own data** - Replace sample data with real market data

## Common Parameters Across Indicators

| Parameter | Purpose | Typical Range | Effect |
|-----------|---------|---------------|--------|
| `window` | Rolling calculation size | 5-30 | Larger = smoother, slower |
| `threshold` | Signal trigger level | 0.01-0.05 | Asset/timeframe dependent |
| `epsilon` | Numerical stability | 1e-9 | Prevents division by zero |

## Support

For questions or issues:
- Check the indicator source code in `src/indikator/`
- Review tests in `tests/` for more usage examples
- See docstrings for detailed parameter descriptions

## Contributing

To add a new visualization notebook:
1. Copy an existing notebook as template
2. Follow the same structure (scenarios, parameters, takeaways)
3. Include at least 3-4 scenarios showing different behaviors
4. Add visual plots with clear labels
5. Document key parameters and interpretation
