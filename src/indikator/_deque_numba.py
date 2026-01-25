"""Numba-optimized monotonic deque for O(n) rolling max/min calculations.

Provides inline helper functions that operate on pre-allocated numpy buffers.
Use `@jit(inline='always')` for zero-overhead function calls.

Usage pattern:
  # Allocate buffers once
  dq = np.zeros(period + 1, dtype=np.int64)
  head, tail = 0, 0

  for i in range(n):
    # Remove expired
    head = deque_expire(dq, head, tail, period + 1, i - period + 1)
    # Push (for max-deque)
    head, tail = deque_push_max(dq, head, tail, period + 1, data, i)
    # Get front
    result[i] = data[deque_front(dq, head, period + 1)]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

# =============================================================================
# Inline deque helper functions (zero overhead when inlined)
# =============================================================================


@jit(nopython=True, inline="always")  # pragma: no cover
def deque_is_empty(head: int, tail: int) -> bool:
  """Check if deque is empty."""
  return head == tail


@jit(nopython=True, inline="always")  # pragma: no cover
def deque_front(dq: NDArray[np.int64], head: int, capacity: int) -> int:
  """Get the index stored at the front of the deque."""
  return dq[head % capacity]


@jit(nopython=True, inline="always")  # pragma: no cover
def deque_expire(
  dq: NDArray[np.int64],
  head: int,
  tail: int,
  capacity: int,
  min_valid_idx: int,
) -> int:
  """Remove expired elements from front. Returns new head."""
  while head != tail and dq[head % capacity] < min_valid_idx:
    head += 1
  return head


@jit(nopython=True, inline="always")  # pragma: no cover
def deque_push_max(
  dq: NDArray[np.int64],
  head: int,
  tail: int,
  capacity: int,
  data: NDArray[np.float64],
  idx: int,
) -> tuple[int, int]:
  """Push index for max-deque: remove smaller elements from back."""
  while head != tail and data[dq[(tail - 1) % capacity]] <= data[idx]:
    tail -= 1
  dq[tail % capacity] = idx
  return head, tail + 1


@jit(nopython=True, inline="always")  # pragma: no cover
def deque_push_min(
  dq: NDArray[np.int64],
  head: int,
  tail: int,
  capacity: int,
  data: NDArray[np.float64],
  idx: int,
) -> tuple[int, int]:
  """Push index for min-deque: remove larger elements from back."""
  while head != tail and data[dq[(tail - 1) % capacity]] >= data[idx]:
    tail -= 1
  dq[tail % capacity] = idx
  return head, tail + 1


# =============================================================================
# Convenience functions for common use cases
# =============================================================================


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def rolling_max_indices(
  data: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
  """Compute rolling maximum values and their indices using monotonic deque.

  O(n) complexity instead of O(n * period).

  Args:
    data: Input array
    period: Window size

  Returns:
    Tuple of (indices of max, max values)
  """
  n = len(data)
  max_indices = np.zeros(n, dtype=np.int64)
  max_values = np.zeros(n, dtype=np.float64)

  # Allocate deque buffer
  capacity = period + 1
  dq = np.zeros(capacity, dtype=np.int64)
  head, tail = 0, 0

  for i in range(n):
    min_valid_idx = i - period + 1
    head = deque_expire(dq, head, tail, capacity, min_valid_idx)
    head, tail = deque_push_max(dq, head, tail, capacity, data, i)

    front_idx = deque_front(dq, head, capacity)
    max_indices[i] = front_idx
    max_values[i] = data[front_idx]

  return max_indices, max_values


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def rolling_min_indices(
  data: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
  """Compute rolling minimum values and their indices using monotonic deque.

  O(n) complexity instead of O(n * period).

  Args:
    data: Input array
    period: Window size

  Returns:
    Tuple of (indices of min, min values)
  """
  n = len(data)
  min_indices = np.zeros(n, dtype=np.int64)
  min_values = np.zeros(n, dtype=np.float64)

  # Allocate deque buffer
  capacity = period + 1
  dq = np.zeros(capacity, dtype=np.int64)
  head, tail = 0, 0

  for i in range(n):
    min_valid_idx = i - period + 1
    head = deque_expire(dq, head, tail, capacity, min_valid_idx)
    head, tail = deque_push_min(dq, head, tail, capacity, data, i)

    front_idx = deque_front(dq, head, capacity)
    min_indices[i] = front_idx
    min_values[i] = data[front_idx]

  return min_indices, min_values
