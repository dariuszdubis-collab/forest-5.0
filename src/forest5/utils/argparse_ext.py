from __future__ import annotations

import argparse
import re
from typing import Callable, List, TypeVar


T = TypeVar("T", int, float)


class PercentAction(argparse.Action):
    """Argparse action that validates a percentage in a given range."""

    def __init__(
        self, option_strings, dest, *, min_value: float = 0.0, max_value: float = 1.0, **kwargs
    ):
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        try:
            is_percent = isinstance(value, str) and "%" in value
            if isinstance(value, str):
                value = value.replace("%", "").replace(",", ".")
            val = float(value)
            if is_percent:
                val /= 100.0
        except ValueError:
            parser.error(f"{option_string} expects a number")
        if not (self.min_value <= val <= self.max_value):
            parser.error(f"{option_string} must be between {self.min_value} and {self.max_value}")
        setattr(namespace, self.dest, val)


def span_or_list(spec: str, type_fn: Callable[[float], T] = float) -> List[T]:
    """Parse comma separated list or inclusive numeric range specification.

    Parameters
    ----------
    spec:
        String containing either comma separated values or a range in one of
        the following forms::

            a,b,c
            a:b:step
            a-b[:step]

    type_fn:
        Function used to cast parsed numeric values. Defaults to ``float`` but
        can be set to ``int``.

    Returns
    -------
    list[``T``]
        Parsed numeric values converted with ``type_fn``.

    Raises
    ------
    argparse.ArgumentTypeError
        If the specification is malformed, the range is descending or step is
        not positive.
    """

    spec = str(spec).strip()

    # Comma separated list
    if "," in spec:
        try:
            return [type_fn(x.strip()) for x in spec.split(",") if x.strip()]
        except ValueError as ex:
            raise argparse.ArgumentTypeError(f"Invalid list: {spec}") from ex

    # a:b:step form
    m = re.fullmatch(
        r"\s*([+-]?\d+(?:\.\d+)?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*",
        spec,
    )
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        step = float(m.group(3))
        if step <= 0:
            raise argparse.ArgumentTypeError(f"Step must be > 0 (given: {step})")
        if hi < lo:
            raise argparse.ArgumentTypeError(f"Upper bound < lower: {spec}")
        vals: List[T] = []
        v = lo
        while v <= hi:
            vals.append(type_fn(v))
            v += step
        if vals and vals[-1] != type_fn(hi):
            vals.append(type_fn(hi))
        return vals

    # Extract optional step for a-b[:step]
    core, step_str = (spec.split(":", 1) + ["1"])[:2]
    try:
        step = float(step_str)
    except ValueError as ex:
        raise argparse.ArgumentTypeError(f"Invalid step: {step_str}") from ex
    if step <= 0:
        raise argparse.ArgumentTypeError(f"Step must be > 0 (given: {step})")

    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*", core)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        if hi < lo:
            raise argparse.ArgumentTypeError(f"Upper bound < lower: {spec}")
        vals: List[T] = []
        v = lo
        while v <= hi:
            vals.append(type_fn(v))
            v += step
        if vals and vals[-1] != type_fn(hi):
            vals.append(type_fn(hi))
        return vals

    # Single value
    try:
        return [type_fn(spec)]
    except ValueError as ex:
        raise argparse.ArgumentTypeError(
            f"Invalid range: {spec}. Expected formats: lo-hi[:step] or lo:hi:step"
        ) from ex
