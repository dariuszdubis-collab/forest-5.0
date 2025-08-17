from __future__ import annotations

import re

# aliasy i mapowanie na standardowe "Xm", "Xh", "Xd"
_ALIASES = {
    "M": "1m",
    "H": "1h",
    "D": "1d",
    "1M": "1m",
    "1H": "1h",
    "1D": "1d",
    "60min": "1h",
    "240": "4h",
    "1440": "1d",
}
_VALID = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "1w",
}
# pomocnicza mapka „tf -> minuty”
_TF_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
}


def _alias_to_tf(raw: str) -> str | None:
    """Zwróć znormalizowane TF dla prostych aliasów, gdy występują."""

    return _ALIASES.get(raw.strip())


def _minutes_to_tf(s: str) -> str | None:
    """Zwróć TF na podstawie liczby minut zapisanej jako string."""

    if s.isdigit():
        minutes = int(s)
        for k, v in _TF_MINUTES.items():
            if v == minutes:
                return k
    return None


def normalize_timeframe(tf: str) -> str:
    """Znormalizuj zapis TF do postaci 'Xm'/'Xh'/'Xd' (np. 'H' -> '1h', '60min' -> '1h')."""

    raw = tf.strip()
    alias = _alias_to_tf(raw)
    if alias:
        return alias

    s = raw.lower().replace(" ", "")
    if s in _VALID:
        return s

    numeric = _minutes_to_tf(s)
    if numeric:
        return numeric

    # "1H", "15M" itp.
    m = re.fullmatch(r"(\d+)([mhdwMHDW])", s)
    if m:
        num, unit = int(m.group(1)), m.group(2).lower()
        if unit == "m":
            cand = f"{num}m"
        elif unit == "h":
            cand = f"{num}h"
        elif unit == "d":
            cand = f"{num}d"
        elif unit == "w":
            cand = f"{num}w"
        else:
            raise ValueError(f"Unsupported unit: {tf}")
        if cand in _VALID or cand in _TF_MINUTES:
            return cand if cand in _VALID else next(k for k, v in _TF_MINUTES.items() if k == cand)

    raise ValueError(f"Nieobsługiwany timeframe: {tf!r}")
