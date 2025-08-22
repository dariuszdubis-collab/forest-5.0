from __future__ import annotations


def _to_sign(value: int | float) -> int:
    """Convert numeric values to -1, 0 or 1."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
