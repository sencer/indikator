"""Tests for sector_correlation indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indikator.sector_correlation import sector_correlation


class TestBasicFunctionality:
    """Test core functionality of sector_correlation."""

    @pytest.fixture
    def stock_series(self) -> pd.Series:
        """Sample stock data."""
        return pd.Series([100.0, 102.0, 105.0, 108.0, 110.0], name="close")

    @pytest.fixture
    def sector_series(self) -> pd.Series:
        """Sample sector data."""
        return pd.Series([1000.0, 1020.0, 1050.0, 1080.0, 1100.0], name="close")

    def test_returns_series_with_name(
        self, stock_series: pd.Series, sector_series: pd.Series
    ) -> None:
        """Should return Series with correct name."""
        result = sector_correlation(stock_series, sector_series, window=3)

        assert isinstance(result, pd.Series)
        assert result.name == "sector_correlation"

    def test_output_length_matches_input(
        self, stock_series: pd.Series, sector_series: pd.Series
    ) -> None:
        """Output should have same length as input."""
        result = sector_correlation(stock_series, sector_series, window=3)
        assert len(result) == len(stock_series)

    def test_empty_series_returns_empty(self) -> None:
        """Empty input should raise ValueError."""
        empty_series = pd.Series([], dtype=float)
        sector_series = pd.Series([100.0, 105.0])

        with pytest.raises(ValueError, match="Data must not be empty"):
            sector_correlation(empty_series, sector_series)


class TestCorrelationCalculation:
    """Test correlation calculation logic."""

    def test_perfect_positive_correlation(self) -> None:
        """Perfectly correlated data should return correlation near +1."""
        stock = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
        # Sector moves exactly the same (scaled)
        sector = pd.Series([1000.0, 1050.0, 1100.0, 1150.0, 1200.0])

        result = sector_correlation(stock, sector, window=3)

        # Should be very close to 1.0 (perfect positive correlation)
        # Skip the first (window-1) elements which are filled with default_value
        corr_values = result.iloc[2:]
        assert (corr_values > 0.99).all()

    def test_perfect_negative_correlation(self) -> None:
        """Inversely correlated data should return correlation near -1."""
        stock = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
        # Sector moves opposite direction
        sector = pd.Series([1200.0, 1150.0, 1100.0, 1050.0, 1000.0])

        result = sector_correlation(stock, sector, window=3)

        # Should be very close to -1.0 (perfect negative correlation)
        # Skip the first (window-1) elements which are filled with default_value
        corr_values = result.iloc[2:]
        assert (corr_values < -0.99).all()

    def test_no_correlation_returns_near_zero(self) -> None:
        """Uncorrelated data should return correlation near 0."""
        stock = pd.Series([100.0, 105.0, 100.0, 105.0, 100.0, 105.0])
        # Sector has completely different pattern
        sector = pd.Series([1000.0, 1000.0, 1050.0, 1050.0, 1000.0, 1000.0])

        result = sector_correlation(stock, sector, window=3)

        # Should be close to 0 (no correlation)
        corr_values = result.dropna()
        # Allow some variation due to window effects
        assert (corr_values.abs() < 0.5).any()

    def test_window_parameter_affects_smoothness(self) -> None:
        """Larger window should produce smoother correlation values."""
        np.random.seed(42)
        n = 50

        stock = pd.Series(100 + np.cumsum(np.random.randn(n)))
        sector = pd.Series(1000 + np.cumsum(np.random.randn(n)))

        result_small = sector_correlation(stock, sector, window=3)
        result_large = sector_correlation(stock, sector, window=20)

        # Larger window should have less variation
        std_small = result_small.dropna().std()
        std_large = result_large.dropna().std()

        assert std_small > std_large


class TestMissingDataHandling:
    """Test handling of missing or incomplete data."""

    @pytest.fixture
    def stock_series(self) -> pd.Series:
        """Sample stock data."""
        return pd.Series([100.0, 102.0, 105.0, 108.0, 110.0])

    def test_none_sector_data_returns_default(self, stock_series: pd.Series) -> None:
        """Should return default value when sector_data is None."""
        result = sector_correlation(stock_series, sector_data=None, default_value=0.5)

        # All values should be the default
        assert (result == 0.5).all()

    def test_empty_sector_data_raises_error(self, stock_series: pd.Series) -> None:
        """Should raise ValueError when sector_data is empty."""
        empty_sector = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Data must not be empty"):
            sector_correlation(stock_series, empty_sector)

    def test_insufficient_window_data_returns_default(self) -> None:
        """Should return default value when data is shorter than window."""
        stock = pd.Series([100.0, 102.0])
        sector = pd.Series([1000.0, 1020.0])

        result = sector_correlation(stock, sector, window=10, default_value=0.0)

        assert (result == 0.0).all()


class TestTimestampAlignment:
    """Test timestamp alignment functionality."""

    def test_aligned_timestamps_work_correctly(self) -> None:
        """Should work correctly when timestamps are aligned."""
        index = pd.date_range("2024-01-01", periods=5, freq="1h")

        stock = pd.Series([100.0, 102.0, 105.0, 108.0, 110.0], index=index)
        sector = pd.Series([1000.0, 1020.0, 1050.0, 1080.0, 1100.0], index=index)

        result = sector_correlation(stock, sector, window=3)

        # Should have valid correlations after window
        assert result.notna().any()

    def test_misaligned_timestamps_use_forward_fill(self) -> None:
        """Should forward fill sector data when timestamps don't match."""
        stock_index = pd.date_range("2024-01-01", periods=5, freq="1h")
        # Sector data has different timestamps (every 2 hours)
        sector_index = pd.date_range("2024-01-01", periods=3, freq="2h")

        stock = pd.Series([100.0, 102.0, 105.0, 108.0, 110.0], index=stock_index)
        sector = pd.Series([1000.0, 1050.0, 1100.0], index=sector_index)

        result = sector_correlation(stock, sector, window=3)

        # Should successfully align and compute correlation
        # Should not return all defaults (alignment should work)
        assert result.nunique() > 1

    def test_poor_alignment_returns_default(self) -> None:
        """Should return default when too many NaN after alignment."""
        stock_index = pd.date_range("2024-01-01", periods=10, freq="1h")
        # Sector data only has 1 point - will result in mostly NaN after reindex
        sector_index = pd.date_range("2024-01-10", periods=1, freq="1h")

        stock = pd.Series(np.arange(100.0, 110.0), index=stock_index)
        sector = pd.Series([1000.0], index=sector_index)

        result = sector_correlation(stock, sector, window=3, default_value=0.0)

        # Should return default due to poor alignment
        assert (result == 0.0).all()


class TestParameterValidation:
    """Test parameter validation and constraints."""

    @pytest.fixture
    def stock_series(self) -> pd.Series:
        """Sample stock data."""
        return pd.Series([100.0, 102.0, 105.0, 108.0, 110.0])

    @pytest.fixture
    def sector_series(self) -> pd.Series:
        """Sample sector data."""
        return pd.Series([1000.0, 1020.0, 1050.0, 1080.0, 1100.0])

    def test_default_parameters_work(
        self, stock_series: pd.Series, sector_series: pd.Series
    ) -> None:
        """Should work with all default parameters."""
        result = sector_correlation(stock_series, sector_series)
        assert isinstance(result, pd.Series)

    def test_custom_window(
        self, stock_series: pd.Series, sector_series: pd.Series
    ) -> None:
        """Should accept custom window parameter."""
        result = sector_correlation(stock_series, sector_series, window=3)
        assert isinstance(result, pd.Series)

    def test_custom_default_value(
        self, stock_series: pd.Series, sector_series: pd.Series
    ) -> None:
        """Should accept custom default_value parameter."""
        result = sector_correlation(stock_series, sector_series, default_value=0.5)
        assert isinstance(result, pd.Series)

    def test_schema_validation_error(self) -> None:
        """Should raise SchemaError for invalid data."""
        stock = pd.Series([100.0, np.inf, 105.0])
        sector = pd.Series([1000.0, 1020.0, 1050.0])

        with pytest.raises(ValueError):
            sector_correlation(stock, sector)


class TestCorrelationBounds:
    """Test that correlation values are within valid bounds."""

    def test_correlation_within_valid_range(self) -> None:
        """Correlation should always be between -1 and 1 for non-NaN values."""
        np.random.seed(42)
        stock = pd.Series(100 + np.cumsum(np.random.randn(50)))
        sector = pd.Series(1000 + np.cumsum(np.random.randn(50)))

        result = sector_correlation(stock, sector, window=10, default_value=0.0)

        # Check only non-NaN correlations (after rolling window is satisfied)
        corr = result.dropna()
        assert (corr >= -1.0).all()
        assert (corr <= 1.0).all()
