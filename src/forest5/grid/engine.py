from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Any, Callable

import pandas as pd


def plan_param_grid(
    param_ranges: Dict[str, Iterable[Any]],
    *,
    filter_fn: Callable[[Dict[str, Any]], bool] | None = None,
) -> pd.DataFrame:
    """Enumerate parameter combinations and assign a ``combo_id``.

    Parameters
    ----------
    param_ranges:
        Mapping from parameter name to an iterable of possible values.
    filter_fn:
        Optional predicate to discard combinations.  It receives a dict of
        parameters and should return ``True`` to keep the combination.

    Returns
    -------
    pandas.DataFrame
        DataFrame where each row represents a parameter combination and the
        first column ``combo_id`` is a unique identifier.
    """

    keys = sorted(param_ranges.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[param_ranges[k] for k in keys])]
    if filter_fn is not None:
        combos = [c for c in combos if filter_fn(c)]
    df = pd.DataFrame(combos)
    df.insert(0, "combo_id", range(len(df)))
    return df
