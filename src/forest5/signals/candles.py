from __future__ import annotations

import numpy as np
import pandas as pd


def _body(open_: pd.Series, close_: pd.Series) -> pd.Series:
    """Bezwzględna wielkość korpusu świecy."""
    return (close_ - open_).abs()


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bycze objęcie: bieżąca świeca bycza, a jej korpus całkowicie obejmuje
    korpus poprzedniej świecy niedźwiedziej.
    Zwraca serię 0/1 (1 tam, gdzie wystąpił wzorzec).
    """
    o, c = df["open"], df["close"]
    o_prev, c_prev = o.shift(1), c.shift(1)

    prev_bear = c_prev < o_prev
    now_bull = c > o
    engulf = (o <= c_prev) & (c >= o_prev)

    out = (prev_bear & now_bull & engulf).astype(int)
    # pierwsza obserwacja nie ma poprzedniej świecy
    out = out.where(~o_prev.isna(), 0)
    return out


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Niedźwiedzie objęcie: bieżąca świeca niedźwiedzia, a jej korpus całkowicie
    obejmuje korpus poprzedniej świecy byczej.
    Zwraca serię 0/1 (1 tam, gdzie wystąpił wzorzec).
    """
    o, c = df["open"], df["close"]
    o_prev, c_prev = o.shift(1), c.shift(1)

    prev_bull = c_prev > o_prev
    now_bear = c < o
    engulf = (o >= c_prev) & (c <= o_prev)

    out = (prev_bull & now_bear & engulf).astype(int)
    out = out.where(~o_prev.isna(), 0)
    return out


def doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """
    Doji: bardzo mały korpus względem całego zakresu świecy.
    Jako kryterium używamy body/range <= threshold (domyślnie 10%).
    Zwraca serię 0/1.
    """
    o, hi, lo, c = df["open"], df["high"], df["low"], df["close"]
    rng = (hi - lo).where((hi - lo) != 0, np.nan)  # unikamy dzielenia przez zero
    body = _body(o, c)
    res = ((body / rng) <= threshold).astype(int).fillna(0)
    return res


def candles_signal(df: pd.DataFrame) -> pd.Series:
    """Połącz sygnały świecowe w jedną serię +1/0/-1.

    +1 – bycze objęcie, -1 – niedźwiedzie objęcie, 0 – doji lub brak wzorca.
    """
    bull = bullish_engulfing(df)
    bear = bearish_engulfing(df)
    dj = doji(df).astype(bool)

    out = bull - bear
    out = out.where(~dj, 0)
    return out.astype("int8")
