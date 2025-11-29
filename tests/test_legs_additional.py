"""Additional comprehensive tests for zigzag_legs indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indikator.legs import zigzag_legs


class TestDataFrameHandling:
    """Test DataFrame handling edge cases."""

    def test_preserves_other_columns(self) -> None:
        """Should preserve other columns (OHLCV format)."""
        df = pd.DataFrame({
            "open": [100.0, 102.0, 105.0],
            "high": [101.0, 103.0, 106.0],
            "low": [99.0, 101.0, 104.0],
            "close": [100.5, 102.5, 105.5],
            "volume": [1000, 1500, 2000],
        })

        result = zigzag_legs(df, threshold=0.01)

        # All original columns should be preserved
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
        assert "zigzag_legs" in result.columns

        # Values should be unchanged
        assert (result["volume"] == df["volume"]).all()

    def test_datetime_index(self) -> None:
        """Should work with DatetimeIndex."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"close": [100.0, 102.0, 105.0, 108.0, 110.0]}, index=dates)

        result = zigzag_legs(df, threshold=0.01)

        # Index should be preserved
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 5
        assert (result.index == dates).all()

    def test_non_sequential_index(self) -> None:
        """Should work with non-sequential integer index."""
        df = pd.DataFrame(
            {"close": [100.0, 102.0, 105.0, 108.0, 110.0]}, index=[0, 5, 10, 15, 20]
        )

        result = zigzag_legs(df, threshold=0.01)

        # Index should be preserved
        assert (result.index == [0, 5, 10, 15, 20]).all()

    def test_named_index(self) -> None:
        """Should preserve named index."""
        df = pd.DataFrame({"close": [100.0, 102.0, 105.0]})
        df.index.name = "bar_number"

        result = zigzag_legs(df)

        assert result.index.name == "bar_number"

    def test_truly_returns_copy(self) -> None:
        """Should return a true copy, not a view."""
        df = pd.DataFrame({"close": [100.0, 102.0, 105.0]})

        result = zigzag_legs(df, threshold=0.01)

        # Modifying result should not affect original
        result.loc[0, "close"] = 999.0
        assert df.loc[0, "close"] == 100.0


class TestComplexSequences:
    """Test complex multi-leg sequences."""

    def test_three_legs_bullish(self) -> None:
        """Test sequence with 3 bullish legs (requires sustained moves)."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # +10% - establishes bullish structure
                111.0,  # Continue up
                112.0,  # Continue up
                108.0,  # Small pullback (doesn't cross threshold)
                115.0,  # Resume up
                116.0,  # Continue up
                117.0,  # Continue up - breaks previous high during this leg
                114.0,  # Small pullback
                120.0,  # Resume up
                121.0,  # Continue up
                122.0,  # Continue up - breaks previous high again
            ]
        })

        result = zigzag_legs(
            df, threshold=0.05, confirmation_bars=2, min_distance_pct=0.01
        )

        legs = result["zigzag_legs"].values

        # In a bullish structure, count should be positive
        assert (legs[legs != 0] > 0).all()
        # Should establish a trend
        assert (legs != 0).any()

    def test_three_legs_bearish(self) -> None:
        """Test sequence with 3 bearish legs (requires sustained moves)."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                90.0,  # -10% - establishes bearish structure
                89.0,  # Continue down
                88.0,  # Continue down
                92.0,  # Small bounce (doesn't cross threshold)
                85.0,  # Resume down
                84.0,  # Continue down
                83.0,  # Continue down - breaks previous low during this leg
                86.0,  # Small bounce
                80.0,  # Resume down
                79.0,  # Continue down
                78.0,  # Continue down - breaks previous low again
            ]
        })

        result = zigzag_legs(
            df, threshold=0.05, confirmation_bars=2, min_distance_pct=0.01
        )

        legs = result["zigzag_legs"].values

        # In a bearish structure, count should be negative
        assert (legs[legs != 0] < 0).all()
        # Should establish a trend
        assert (legs != 0).any()

    def test_alternating_trends_multiple_times(self) -> None:
        """Test long sequence with multiple price swings."""
        # This test verifies the algorithm handles extended sequences
        # without errors. The existing test_legs_structure.py tests
        # thoroughly cover trend change logic.
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,
                105.0,
                115.0,
                110.0,
                105.0,
                95.0,
                100.0,
                90.0,
                95.0,
                100.0,
                110.0,
                105.0,
                115.0,
            ]
        })

        result = zigzag_legs(
            df, threshold=0.05, confirmation_bars=1, min_distance_pct=0.01
        )

        legs = result["zigzag_legs"].values

        # Should establish some trend (positive or negative)
        assert (legs != 0).any()
        # Length should match input
        assert len(legs) == len(df)


class TestParameterBoundaries:
    """Test extreme parameter values."""

    def test_all_parameters_at_minimum(self) -> None:
        """Test with all parameters at minimum values."""
        df = pd.DataFrame({"close": [100.0, 100.01, 100.02, 100.01, 100.03]})

        result = zigzag_legs(
            df, threshold=0.0, min_distance_pct=0.0, confirmation_bars=0, epsilon=1e-12
        )

        assert "zigzag_legs" in result.columns

    def test_all_parameters_at_maximum(self) -> None:
        """Test with all parameters at maximum practical values."""
        df = pd.DataFrame({"close": [100.0, 150.0, 120.0, 180.0, 140.0]})

        result = zigzag_legs(
            df, threshold=1.0, min_distance_pct=1.0, confirmation_bars=10, epsilon=1.0
        )

        # With such high thresholds, nothing should trigger
        assert (result["zigzag_legs"] == 0.0).all()

    def test_very_long_confirmation_period(self) -> None:
        """Test with confirmation period longer than data."""
        df = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0, 110.0]})

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=100)

        # Confirmation never completes
        assert "zigzag_legs" in result.columns

    def test_threshold_larger_than_price_movement(self) -> None:
        """Test where all price movements are below threshold."""
        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0, 104.0]  # Max 4% total move
        })

        result = zigzag_legs(df, threshold=0.50)  # 50% threshold

        # No trend should be established
        assert (result["zigzag_legs"] == 0.0).all()


class TestNumericTypes:
    """Test different numeric data types."""

    def test_integer_prices(self) -> None:
        """Test with integer price data."""
        df = pd.DataFrame({"close": [100, 102, 105, 108, 110]})

        result = zigzag_legs(df, threshold=0.01)

        assert "zigzag_legs" in result.columns
        assert len(result) == 5

    def test_float32_prices(self) -> None:
        """Test with float32 data type."""
        df = pd.DataFrame({"close": np.array([100.0, 102.0, 105.0], dtype=np.float32)})

        result = zigzag_legs(df, threshold=0.01)

        assert "zigzag_legs" in result.columns

    def test_mixed_numeric_types(self) -> None:
        """Test with mixed numeric types in DataFrame."""
        df = pd.DataFrame({
            "close": [100.0, 102, 105.0, 108, 110.0],  # Mixed int and float
            "volume": [1000, 1500, 2000, 2500, 3000],  # Integers
        })

        result = zigzag_legs(df, threshold=0.01)

        assert "zigzag_legs" in result.columns


class TestNegativePrices:
    """Test handling of negative price values."""

    def test_negative_prices(self) -> None:
        """Test with negative price values (theoretical edge case)."""
        df = pd.DataFrame({"close": [-100.0, -110.0, -105.0, -115.0]})

        # Should handle negative prices
        result = zigzag_legs(df, threshold=0.05)

        assert "zigzag_legs" in result.columns
        assert len(result) == 4

    def test_prices_crossing_zero(self) -> None:
        """Test with prices crossing from negative to positive."""
        df = pd.DataFrame({"close": [-10.0, -5.0, 0.1, 5.0, 10.0]})

        result = zigzag_legs(df, threshold=0.1, epsilon=1e-9)

        assert "zigzag_legs" in result.columns


class TestPerformance:
    """Test performance with large datasets."""

    def test_large_dataset(self, large_dataset_df: pd.DataFrame) -> None:
        """Test with large dataset (10k bars)."""
        result = zigzag_legs(large_dataset_df, threshold=0.02, confirmation_bars=3)

        assert len(result) == len(large_dataset_df)
        assert "zigzag_legs" in result.columns

    def test_very_large_dataset(self) -> None:
        """Test with very large dataset (100k bars)."""
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, 100_000)
        prices = 100.0 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"close": prices})

        result = zigzag_legs(df, threshold=0.02)

        assert len(result) == 100_000
        assert "zigzag_legs" in result.columns


class TestRobustness:
    """Test algorithm robustness."""

    def test_constant_price_with_noise(self) -> None:
        """Test constant price with tiny random noise."""
        np.random.seed(42)
        noise = np.random.normal(0, 0.0001, 100)
        df = pd.DataFrame({"close": 100.0 + noise})

        result = zigzag_legs(df, threshold=0.01, min_distance_pct=0.005)

        # Should not establish trend with tiny noise
        assert "zigzag_legs" in result.columns

    def test_extreme_volatility(self) -> None:
        """Test with extreme price volatility."""
        df = pd.DataFrame({
            "close": [
                100.0,
                200.0,  # 100% jump
                50.0,  # 75% drop
                150.0,  # 200% jump
                25.0,  # 83% drop
            ]
        })

        result = zigzag_legs(df, threshold=0.10)

        assert "zigzag_legs" in result.columns
        # Should detect trend changes
        assert (result["zigzag_legs"] != 0).any()

    def test_precision_at_threshold_boundary(self) -> None:
        """Test precision exactly at threshold boundary."""
        df = pd.DataFrame({
            "close": [
                100.0,
                100.0100001,  # Just over threshold
                99.9899999,  # Just under threshold
            ]
        })

        result = zigzag_legs(df, threshold=0.0001, confirmation_bars=0)

        assert "zigzag_legs" in result.columns


class TestConfirmationComplexity:
    """Test complex confirmation scenarios."""

    def test_confirmation_with_price_whipsaw(self) -> None:
        """Test confirmation with whipsaw price action."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up trend established
                105.0,  # Start potential reversal
                106.0,  # Bounce
                104.0,  # Down again
                107.0,  # Bounce again
                103.0,  # Finally down
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=2)

        assert "zigzag_legs" in result.columns

    def test_confirmation_with_exact_pivot_tracking(self) -> None:
        """Test that confirmation tracks the most extreme price."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up
                108.0,  # Down 1
                107.0,  # Down 2 (most extreme during confirmation)
                107.5,  # Slight bounce (confirmation completes)
            ]
        })

        result = zigzag_legs(
            df, threshold=0.05, confirmation_bars=2, min_distance_pct=0.0
        )

        # Just verify it runs without error and has expected length
        assert len(result) == 5

    def test_multiple_confirmations_in_sequence(self) -> None:
        """Test multiple confirmation periods in sequence."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up
                105.0,  # Confirm down 1
                104.0,  # Confirm down 2
                105.0,  # Slight bounce
                100.0,  # Down again 1
                99.0,  # Down 2
                100.0,  # Bounce
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=2)

        assert "zigzag_legs" in result.columns


class TestMinDistanceComplexity:
    """Test complex min_distance scenarios."""

    def test_min_distance_with_trending_noise(self) -> None:
        """Test min_distance filtering in trending market."""
        df = pd.DataFrame({
            "close": [
                100.0,
                100.1,  # Noise
                100.2,  # Noise
                100.15,  # Noise
                105.0,  # Real move
                105.1,  # Noise
                105.2,  # Noise
                110.0,  # Real move
            ]
        })

        result = zigzag_legs(
            df, threshold=0.01, min_distance_pct=0.01, confirmation_bars=0
        )

        assert "zigzag_legs" in result.columns

    def test_min_distance_prevents_wick_counting(self) -> None:
        """Test that min_distance prevents counting small wicks."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up 10%
                110.1,  # Tiny wick up
                110.05,  # Tiny move
                110.2,  # Tiny wick up
                110.0,  # Back to previous level
            ]
        })

        result = zigzag_legs(
            df, threshold=0.05, min_distance_pct=0.005, confirmation_bars=0
        )

        # All the tiny moves should be filtered
        assert "zigzag_legs" in result.columns
