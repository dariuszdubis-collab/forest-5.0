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


def parse_span_or_list(spec: str) -> list[float]:
    """Parse a numeric span (``lo-hi[:step]`` or ``lo:hi:step``) or a
    comma-separated list into a list of floats.

    Examples
    --------
    ``"1-3"`` -> ``[1.0, 2.0, 3.0]``
    ``"1-5:2"`` -> ``[1.0, 3.0, 5.0]``
    ``"1,2,10"`` -> ``[1.0, 2.0, 10.0]``

    The range is inclusive and supports negative numbers. Raises
    :class:`argparse.ArgumentTypeError` when the specification is malformed,
    ``lo > hi`` or ``step <= 0``.
    """

    spec = str(spec).strip()
    if "," in spec:
        try:
            return [float(x.strip()) for x in spec.split(",") if x.strip()]
        except ValueError as ex:  # pragma: no cover
            raise argparse.ArgumentTypeError(f"Invalid list: {spec}") from ex

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
        vals: list[float] = []
        v = lo
        while v <= hi + 1e-9:
            vals.append(v)
            v += step
        if vals[-1] != hi:
            vals.append(hi)
        return vals

    core, step_str = (spec.split(":", 1) + ["1"])[0:2]
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
        vals = []
        v = lo
        while v <= hi + 1e-9:
            vals.append(v)
            v += step
        if vals[-1] != hi:
            vals.append(hi)
        return vals

    try:
        return [float(spec)]
    except ValueError as ex:  # pragma: no cover
        raise argparse.ArgumentTypeError(
            f"Invalid range: {spec}. Expected formats: lo-hi[:step] or lo:hi:step"
        ) from ex
