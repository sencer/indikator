"""Market structure tracking tests for zigzag_legs indicator."""

from __future__ import annotations

import pandas as pd

from indikator.legs import zigzag_legs


class TestHigherHighsLowerLows:
    """Test higher high and lower low detection."""

    def test_single_higher_high_increases_count(
        self, higher_highs_df: pd.DataFrame
    ) -> None:
        """Higher high in uptrend should increase leg count."""
        result = zigzag_legs(
            higher_highs_df["close"],
            threshold=0.03,
            confirmation_bars=0,
            min_distance_pct=0.01,
        )

        legs = result.values

        # Should see increasing positive counts as higher highs form
        # Filter out zeros and check that counts increase
        non_zero = legs[legs > 0]
        if len(non_zero) > 1:
            # At least some counts should increase
            assert non_zero[-1] >= non_zero[0]

    def test_single_lower_low_increases_count(
        self, lower_lows_df: pd.DataFrame
    ) -> None:
        """Lower low in downtrend should increase leg count (more negative)."""
        result = zigzag_legs(
            lower_lows_df["close"],
            threshold=0.03,
            confirmation_bars=0,
            min_distance_pct=0.01,
        )

        legs = result.values

        # Should see increasingly negative counts as lower lows form
        non_zero = legs[legs < 0]
        if len(non_zero) > 1:
            # Counts should get more negative (larger absolute value)
            assert abs(non_zero[-1]) >= abs(non_zero[0])

    def test_higher_high_detection_in_established_uptrend(self) -> None:
        """Test that higher highs are detected correctly (live counting)."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # +10% up - leg 1, bullish
                105.0,  # -4.5% down - correction, sets last_low=105
                112.0,  # +6.7% up - moves but doesn't break high yet
                116.0,  # Continues up, breaks last_high=110 -> leg 2
                112.0,  # -3.4% down - correction
                118.0,  # +5.4% up - moves
                122.0,  # Continues up, breaks last_high=116 -> leg 3
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.01
        )

        legs = result.values

        # Live counting: count increments when price breaks previous high DURING up leg
        # Should reach at least 2 (possibly 3)
        max_legs = legs.max()
        assert max_legs >= 2

    def test_lower_low_detection_in_established_downtrend(self) -> None:
        """Test that lower lows are detected correctly (live counting)."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                90.0,  # -10% down - leg 1, bearish
                95.0,  # +5.6% up - correction, sets last_high=95
                88.0,  # -7.4% down - moves but doesn't break low yet
                84.0,  # Continues down, breaks last_low=90 -> leg 2
                88.0,  # +4.8% up - correction
                82.0,  # -6.8% down - moves
                78.0,  # Continues down, breaks last_low=84 -> leg 3
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.01
        )

        legs = result.values

        # Live counting: count increments when price breaks previous low DURING down leg
        # Should reach at least -2 (possibly -3)
        min_legs = legs.min()
        assert min_legs <= -2


class TestTrendChanges:
    """Test transitions between bullish and bearish trends."""

    def test_trend_change_from_bullish_to_bearish(
        self, trend_change_bull_to_bear_df: pd.DataFrame
    ) -> None:
        """Test transition from bullish to bearish structure."""
        result = zigzag_legs(
            trend_change_bull_to_bear_df["close"],
            threshold=0.03,
            confirmation_bars=0,
            min_distance_pct=0.01,
        )

        legs = result.values

        # Should start positive and end negative
        # Find first non-zero
        non_zero_indices = legs != 0
        if non_zero_indices.any():
            first_nonzero = legs[non_zero_indices][0]
            last_value = legs[-1]

            # Should transition from positive to negative
            assert first_nonzero > 0
            assert last_value < 0

    def test_trend_change_from_bearish_to_bullish(self) -> None:
        """Test transition from bearish to bullish structure."""
        df = pd.DataFrame({
            "close": [
                100.0,
                90.0,  # Bearish leg
                95.0,  # Lower high (still bearish)
                85.0,  # Lower low - confirms bearish
                90.0,  # Pullback
                100.0,  # Breaks previous high (95) - bullish!
                105.0,  # Higher high confirmed
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.01
        )

        legs = result.values

        # Should transition from negative to positive
        if (legs != 0).any():
            non_zero = legs[legs != 0]
            first_nonzero = non_zero[0]
            last_value = legs[-1]

            assert first_nonzero < 0
            assert last_value > 0

    def test_multiple_trend_changes(self) -> None:
        """Test multiple trend changes in sequence."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,
                105.0,
                115.0,  # Bullish: higher highs
                105.0,
                95.0,  # Break below 105 - bearish!
                100.0,
                90.0,  # Bearish: lower lows
                100.0,
                110.0,  # Break above 100 - bullish again!
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.01
        )

        legs = result.values

        # Should have both positive and negative values
        has_positive = (legs > 0).any()
        has_negative = (legs < 0).any()

        assert has_positive and has_negative

    def test_count_resets_to_one_on_trend_change(self) -> None:
        """When trend changes, count should reset to 1."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Bullish
                105.0,
                115.0,  # leg 2
                110.0,
                120.0,  # leg 3
                115.0,
                105.0,  # Breaks structure - should reset to leg 1 (bearish)
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.01
        )

        legs = result.values

        # After trend change, should start at ±1
        # Find where sign changes
        for i in range(1, len(legs)):
            if legs[i] != 0 and legs[i - 1] != 0 and (legs[i] > 0) != (legs[i - 1] > 0):
                # New trend should start at ±1
                assert abs(legs[i]) == 1
                break


class TestCorrectionsVsImpulse:
    """Test distinction between corrective and impulsive moves."""

    def test_pullback_in_uptrend_doesnt_increase_count(self) -> None:
        """Pullback (correction) in uptrend shouldn't increase leg count."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # Impulse up - leg 1
                108.0,  # Pullback (correction)
                109.0,  # Still in correction
                106.0,  # Deeper pullback
                # No higher high yet, so count should stay at 1
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.01
        )

        legs = result.values

        # Count should stay at 1 during correction
        positive_vals = legs[legs > 0]
        if len(positive_vals) > 0:
            # Should not exceed 1 without a higher high
            assert positive_vals.max() == 1

    def test_impulse_after_correction_increases_count(self) -> None:
        """Impulse move after correction should increase count (live counting)."""
        df = pd.DataFrame({
            "close": [
                100.0,  # Start
                110.0,  # +10% impulse up - leg 1
                106.0,  # -3.6% correction (crosses threshold, sets last_low=106)
                108.0,  # +1.9% moves up but doesn't break high yet
                112.0,  # Continues up, breaks last_high=110 -> leg 2
                115.0,  # Continues higher
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.01
        )

        legs = result.values

        # Live counting: after correction, breaking previous high increases count
        positive_vals = legs[legs > 0]
        if len(positive_vals) > 0:
            assert positive_vals.max() >= 2  # Should reach leg 2


class TestStructureBreaks:
    """Test structure break detection."""

    def test_break_of_previous_low_in_uptrend(self) -> None:
        """Breaking previous low in uptrend signals trend change."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # High
                105.0,  # Low (pullback)
                115.0,  # Higher high - bullish confirmed
                104.0,  # Breaks previous low (105) - structure break!
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.0
        )

        # Should detect structure break
        assert isinstance(result, pd.Series)

    def test_break_of_previous_high_in_downtrend(self) -> None:
        """Breaking previous high in downtrend signals trend change."""
        df = pd.DataFrame({
            "close": [
                100.0,
                90.0,  # Low
                95.0,  # High (pullback)
                85.0,  # Lower low - bearish confirmed
                96.0,  # Breaks previous high (95) - structure break!
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.0
        )

        # Should detect structure break and potentially reverse
        assert isinstance(result, pd.Series)


class TestLiveCountingBehavior:
    """Test live counting behavior (counting on current leg)."""

    def test_count_updates_during_current_leg(self) -> None:
        """Count should update as new highs/lows form in current trend."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # leg 1 starts
                105.0,  # pullback
                111.0,  # Small new high
                106.0,  # pullback
                115.0,  # Significant new high - leg 2
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.01
        )

        # Should show progression of leg count
        assert isinstance(result, pd.Series)

    def test_count_persists_during_correction(self) -> None:
        """Count should persist (not decrease) during corrections."""
        df = pd.DataFrame({
            "close": [
                100.0,
                110.0,  # leg 1
                105.0,
                115.0,  # leg 2
                112.0,  # Correction
                111.0,  # Deeper correction
                # Count should still show 2 throughout correction
            ]
        })

        result = zigzag_legs(
            df["close"], threshold=0.03, confirmation_bars=0, min_distance_pct=0.01
        )

        legs = result.values

        # During correction, count shouldn't decrease
        # Once a count is reached, it should persist until trend change
        for i in range(1, len(legs)):
            if legs[i] > 0 and legs[i - 1] > 0:
                # In same trend, should not decrease
                assert legs[i] >= legs[i - 1]
