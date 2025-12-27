"""Shared constants for indikator.

These constants are used across multiple indicator modules to ensure
consistent behavior and easy maintenance.
"""

# Default minimum periods for intraday aggregation functions.
# This is the minimum number of observations required before
# the expanding window starts producing non-NaN values.
DEFAULT_MIN_SAMPLES = 3

# Default epsilon for division-by-zero protection.
# Used in indicators that involve division operations.
DEFAULT_EPSILON = 1e-9

# Maximum ratio of NaN values allowed after alignment before
# falling back to default values (used in sector_correlation).
MAX_NAN_RATIO = 0.5
