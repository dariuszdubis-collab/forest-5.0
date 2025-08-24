from __future__ import annotations

import argparse
import re


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


def span_or_list(spec: str, type_fn=float) -> list:
    """Parse a numeric span or comma separated list.

    Parameters
    ----------
    spec:
        String specification in one of the forms ``lo-hi[:step]`` or
        ``lo:hi:step`` or a comma separated list.
    type_fn:
        Callable used to convert parsed numbers. Typically :class:`int` or
        :class:`float`.

    Returns
    -------
    list
        List of numbers converted by ``type_fn``.

    Raises
    ------
    argparse.ArgumentTypeError
        If the specification is malformed or violates bounds/step rules.
    """

    spec = str(spec).strip()

    def _cast(value: str):
        return int(float(value)) if type_fn is int else float(value)

    def _range(lo: float, hi: float, step: float) -> list:
        if type_fn is int:
            lo_i, hi_i, step_i = int(lo), int(hi), int(step)
            vals = list(range(lo_i, hi_i + 1, step_i))
            if vals[-1] != hi_i:
                vals.append(hi_i)
            return vals

        vals: list[float] = []
        cur = lo
        eps = step / 10_000_000.0
        while cur <= hi + eps:
            vals.append(cur)
            cur += step
        if abs(vals[-1] - hi) > eps:
            vals.append(hi)
        return [float(v) for v in vals]

    # Comma separated list
    if "," in spec:
        try:
            return [type_fn(_cast(x.strip())) for x in spec.split(",") if x.strip()]
        except ValueError as ex:
            raise argparse.ArgumentTypeError(f"Invalid list: {spec}") from ex

    # ``lo:hi:step`` form
    m = re.fullmatch(
        r"\s*([+-]?\d+(?:\.\d+)?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*",
        spec,
    )
    if m:
        lo, hi, step = map(_cast, m.groups())
        if step <= 0:
            raise argparse.ArgumentTypeError(f"Step must be > 0 (given: {step})")
        if hi < lo:
            raise argparse.ArgumentTypeError(f"Upper bound < lower: {spec}")
        return [type_fn(v) for v in _range(lo, hi, step)]

    # ``lo-hi[:step]`` form
    core, step_str = (spec.split(":", 1) + ["1"])[:2]
    try:
        step = _cast(step_str)
    except ValueError as ex:
        raise argparse.ArgumentTypeError(f"Invalid step: {step_str}") from ex
    if step <= 0:
        raise argparse.ArgumentTypeError(f"Step must be > 0 (given: {step})")

    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*", core)
    if m:
        lo, hi = map(_cast, m.groups())
        if hi < lo:
            raise argparse.ArgumentTypeError(f"Upper bound < lower: {spec}")
        return [type_fn(v) for v in _range(lo, hi, step)]

    # Single value
    try:
        return [type_fn(_cast(spec))]
    except ValueError as ex:
        raise argparse.ArgumentTypeError(
            f"Invalid range: {spec}. Expected formats: lo-hi[:step] or lo:hi:step"
        ) from ex
