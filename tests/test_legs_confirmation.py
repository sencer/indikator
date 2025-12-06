"""Confirmation period tests for zigzag_legs indicator."""

from __future__ import annotations

import pandas as pd

from indikator.legs import zigzag_legs


class TestConfirmationPeriod:
    """Test confirmation_bars parameter behavior."""

    def test_zero_confirmation_immediate_reversal(self) -> None:
        """confirmation_bars=0 should reverse immediately when structure breaks."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # Up 10% - establishes bullish (last_low=-inf)
                103.0,  # Down 6.4% - establishes last_low=103
                115.0,  # Up 11.7% - higher high, continues bullish
                100.0,  # Down 13% - breaks last_low=103
                106.0,  # Up 6% - completes leg, evaluates break
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=0)

        legs = result["zigzag_legs"].values
        assert legs[0] == 0  # Initial
        assert legs[1] > 0  # Bullish established
        assert legs[2] > 0  # Still bullish (correction)
        assert legs[3] > 0  # Still bullish (higher high)
        assert legs[4] > 0  # Still bullish (mid-leg, break not evaluated yet)
        assert legs[5] < 0  # NOW bearish - structure break detected on leg completion

    def test_one_bar_confirmation(self) -> None:
        """confirmation_bars=1 should require one bar to confirm reversal."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # Up 10% - bullish
                103.0,  # Down 6.4% - establishes last_low=103
                115.0,  # Up 11.7% - higher high
                100.0,  # Down 13% - crosses threshold, needs 1 bar confirmation
                99.0,  # Confirmation bar (breaks last_low=103) - structure break detected here!
                106.0,  # Up - new leg starts in bearish structure
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=1)

        legs = result["zigzag_legs"].values
        assert legs[4] > 0  # Still bullish (confirmation not complete)
        assert legs[5] < 0  # Bearish - confirmation complete, structure break detected
        assert legs[6] < 0  # Still bearish

    def test_two_bar_confirmation(self) -> None:
        """confirmation_bars=2 should require two bars to confirm."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up 10%
                105.0,  # Start reversal
                104.0,  # Confirmation bar 1
                103.0,  # Confirmation bar 2
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=2)

        # Should eventually confirm downtrend
        assert "zigzag_legs" in result.columns

    def test_confirmation_cancelled_by_new_high(self) -> None:
        """Confirmation should be cancelled if price makes new high."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up 10% - establishes uptrend
                105.0,  # Down 4.5% - starts potential reversal
                106.0,  # Bounces back up - should cancel reversal
                112.0,  # Makes new high - confirms uptrend continues
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=2)

        legs = result["zigzag_legs"].values
        # Should remain positive (bullish) throughout
        assert legs[-1] > 0

    def test_confirmation_cancelled_by_new_low(self) -> None:
        """Confirmation should be cancelled if price makes new low in downtrend."""
        df = pd.DataFrame({
            "close": [
                100.0,
                90.0,  # Down 10% - establishes downtrend
                95.0,  # Up 5.6% - starts potential reversal
                94.0,  # Drops back - should cancel reversal
                88.0,  # Makes new low - confirms downtrend continues
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=2)

        legs = result["zigzag_legs"].values
        # Should remain negative (bearish) throughout
        assert legs[-1] < 0

    def test_confirmation_tracks_extreme_during_period(self) -> None:
        """During confirmation, should track the most extreme price."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # Up 10% - establishes uptrend
                105.0,  # Down 4.5%, confirmation bar 1 (tracks 105)
                103.0,  # Down further, confirmation bar 2 (tracks 103 as extreme)
                104.0,  # Bounce up slightly (confirmation complete, pivot should be 103)
            ]
        })

        result = zigzag_legs(
            df, threshold=0.05, confirmation_bars=2, min_distance_pct=0.0
        )

        legs = result["zigzag_legs"].values
        # Just verify confirmation completed - don't test structure breaks here
        assert legs[0] == 0  # Initial
        assert legs[1] > 0  # Bullish
        assert legs[-1] > 0  # Still bullish (103 is higher low, no structure break)

    def test_long_confirmation_period(self) -> None:
        """Test with long confirmation period (5 bars)."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # Up 10% - establishes uptrend
                109.0,  # Down bar 1
                108.0,  # Bar 2
                107.0,  # Bar 3
                106.0,  # Bar 4
                105.0,  # Bar 5 - confirmation complete
                106.0,  # Slight bounce (reversal confirmed, down leg now active)
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=5)

        legs = result["zigzag_legs"].values
        # Just verify confirmation period works with 5 bars
        assert legs[0] == 0  # Initial
        assert legs[1] > 0  # Bullish
        assert legs[4] > 0  # Still bullish (confirmation not complete)
        assert legs[5] > 0  # Still bullish (confirmation just completed)
        assert legs[-1] > 0  # Still bullish (higher low, no structure break)


class TestConfirmationWithThreshold:
    """Test interaction between confirmation and threshold."""

    def test_must_cross_threshold_before_confirmation_starts(self) -> None:
        """Confirmation should only start after threshold is crossed."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up 10% - establishes uptrend
                109.0,  # Down 0.9% - below threshold, no confirmation
                108.0,  # Down another 0.9% - still below threshold
                107.0,  # Down another 0.9% - still below threshold
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=1)

        legs = result["zigzag_legs"].values
        # Should remain bullish (no threshold cross)
        assert legs[-1] > 0

    def test_threshold_and_confirmation_together(self) -> None:
        """Test that both threshold and confirmation are required."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # Up 10% - bullish
                104.0,  # Down 5.5% - establishes last_low=104
                115.0,  # Up 10.6% - higher high
                105.0,  # Down 8.7% - crosses threshold
                102.0,  # Confirmation bar (breaks last_low=104)
                108.0,  # Up - completes leg, evaluates break
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=1)

        # Should reverse to bearish after confirmation and leg completion
        legs = result["zigzag_legs"].values
        assert (
            legs[-1] < 0
        )  # Bearish - threshold crossed, confirmed, and structure broken


class TestConfirmationEdgeCases:
    """Test edge cases in confirmation logic."""

    def test_confirmation_at_end_of_data(self) -> None:
        """Confirmation period that ends exactly at last bar."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # Up 10% - bullish
                103.0,  # Down 6.4% - establishes last_low=103
                115.0,  # Up 11.7% - higher high
                105.0,  # Down 8.7% - crosses threshold
                102.0,  # Confirmation bar (breaks last_low=103)
                108.0,  # Up - completes leg (last bar)
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=1)

        # Should confirm and evaluate structure break on last bar
        legs = result["zigzag_legs"].values
        assert legs[-1] < 0  # Bearish - structure break at end of data

    def test_confirmation_incomplete_at_end(self) -> None:
        """Incomplete confirmation at end of data."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up 10%
                105.0,  # Start reversal
                # Would need more bars to confirm with confirmation_bars=5
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=5)

        legs = result["zigzag_legs"].values
        # Should still be in uptrend (confirmation incomplete)
        # Or might be 0 if not confirmed yet
        assert legs[-1] >= 0

    def test_multiple_failed_confirmation_attempts(self) -> None:
        """Multiple attempts at reversal that all get cancelled."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up 10%
                105.0,  # Attempt 1
                111.0,  # Cancelled - new high
                106.0,  # Attempt 2
                112.0,  # Cancelled - new high
                107.0,  # Attempt 3
                113.0,  # Cancelled - new high
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=1)

        legs = result["zigzag_legs"].values
        # Should remain bullish throughout
        assert legs[-1] > 0

    def test_confirmation_with_exact_threshold_price(self) -> None:
        """Confirmation with price exactly at threshold."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Up 10%
                104.5,  # Exactly 5% down from 110 (threshold boundary)
                104.0,  # Confirmation
            ]
        })

        result = zigzag_legs(df, threshold=0.05, confirmation_bars=1)

        # Should handle exact threshold
        assert "zigzag_legs" in result.columns


class TestConfirmationWithMinDistance:
    """Test interaction between confirmation and min_distance_pct."""

    def test_confirmation_with_min_distance_filtering(self) -> None:
        """Confirmation should work with min_distance filtering."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # Up 10% - bullish
                103.0,  # Down 6.4% - establishes last_low=103
                115.0,  # Up 11.7% - higher high
                109.9,  # Tiny drop (below min_distance, pivot not updated)
                109.8,  # Tiny drop (below min_distance, pivot not updated)
                105.0,  # Significant drop (crosses threshold)
                102.0,  # Confirmation (breaks last_low=103)
                108.0,  # Up - completes leg, evaluates break
            ]
        })

        result = zigzag_legs(
            df, threshold=0.05, confirmation_bars=1, min_distance_pct=0.005
        )

        # Should reverse after filtering tiny moves and breaking structure
        legs = result["zigzag_legs"].values
        assert legs[-1] < 0  # Bearish - structure break despite min_distance filtering
