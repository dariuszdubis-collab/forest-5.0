from __future__ import annotations


def should_halt_for_drawdown(start_eq: float, cur_eq: float, max_dd: float) -> bool:
    """Return True if drawdown from ``start_eq`` to ``cur_eq`` is at least ``max_dd``.

    ``max_dd`` is expressed as a fraction (e.g. 0.20 for 20%). If ``start_eq`` or
    ``max_dd`` are non-positive the function returns ``False``.
    """
    if start_eq <= 0 or max_dd <= 0:
        return False
    dd = (start_eq - cur_eq) / start_eq
    return dd >= max_dd
