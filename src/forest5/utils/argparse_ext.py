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


class EnumAction(argparse.Action):
    """Map string choices to canonical values.

    Parameters
    ----------
    choices:
        Mapping of accepted strings to canonical values. Matching is
        case-insensitive.
    """

    def __init__(self, option_strings, dest, *, choices: dict[str, object], **kwargs):
        self._mapping = {str(k).lower(): v for k, v in choices.items()}
        if "default" in kwargs:
            default = kwargs["default"]
            if isinstance(default, str):
                kwargs["default"] = self._mapping.get(default.lower(), default)
        kwargs.setdefault("choices", tuple(self._mapping.keys()))
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        key = str(value).lower()
        if key not in self._mapping:
            parser.error(
                f"{option_string} must be one of {', '.join(sorted(self._mapping.keys()))}"
            )
        setattr(namespace, self.dest, self._mapping[key])


def positive_int(value: str) -> int:
    """Parse a positive integer (>= 1)."""

    try:
        iv = int(value)
    except ValueError as ex:  # pragma: no cover - argparse handles message
        raise argparse.ArgumentTypeError("must be an integer") from ex
    if iv < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return iv


def span_or_list(spec: str, type_fn: type | None = None) -> list:
    """Parse a numeric span or comma separated list.

    Parameters
    ----------
    spec:
        String specification in one of the forms ``lo-hi[:step]`` or
        ``lo:hi:step`` or a comma separated list.
    type_fn:
        Optional explicit type to cast values to. If ``None`` the type is
        inferred (``int`` if all numbers are whole, otherwise ``float``).

    Returns
    -------
    list
        List of parsed numbers.

    Raises
    ------
    argparse.ArgumentTypeError
        If the specification is malformed or violates bounds/step rules.
    """

    spec = str(spec).strip()

    def _cast(value: str) -> float:
        return float(value)

    def _range(lo: float, hi: float, step: float) -> list[float]:
        vals: list[float] = []
        cur = lo
        eps = step / 10_000_000.0
        while cur <= hi + eps:
            vals.append(cur)
            cur += step
        if abs(vals[-1] - hi) > eps:
            vals.append(hi)
        return vals

    def _finalize(vals: list[float]) -> list:
        target = type_fn
        if target is None:
            if all(float(v).is_integer() for v in vals):
                target = int
            else:
                target = float
        if target is int:
            return [int(round(v)) for v in vals]
        return [float(v) for v in vals]

    # Comma separated list
    if "," in spec:
        try:
            vals = [_cast(x.strip()) for x in spec.split(",") if x.strip()]
        except ValueError as ex:
            raise argparse.ArgumentTypeError(f"Invalid list: {spec}") from ex
        return _finalize(vals)

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
        return _finalize(_range(lo, hi, step))

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
        return _finalize(_range(lo, hi, step))

    # Single value
    try:
        return _finalize([_cast(spec)])
    except ValueError as ex:
        raise argparse.ArgumentTypeError(
            f"Invalid range: {spec}. Expected formats: lo-hi[:step] or lo:hi:step",
        ) from ex
