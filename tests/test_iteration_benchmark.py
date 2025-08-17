import numpy as np
import pandas as pd


def iterrows_loop(df: pd.DataFrame, sig: pd.Series) -> float:
    total = 0.0
    for t, row in df.iterrows():
        total += float(row["close"]) * int(sig.loc[t])
    return total


def numpy_loop(prices: np.ndarray, sig_array: np.ndarray) -> float:
    total = 0.0
    for price, s in zip(prices, sig_array):
        total += price * s
    return total


def _make_data() -> tuple[pd.DataFrame, pd.Series]:
    n = 10000
    idx = pd.RangeIndex(n)
    df = pd.DataFrame({"close": np.random.rand(n)}, index=idx)
    sig = pd.Series(np.random.randint(-1, 2, size=n), index=idx)
    return df, sig


def test_iterrows_loop_benchmark(benchmark) -> None:
    df, sig = _make_data()
    benchmark(iterrows_loop, df, sig)


def test_numpy_loop_benchmark(benchmark) -> None:
    df, sig = _make_data()
    benchmark(numpy_loop, df["close"].to_numpy(), sig.to_numpy())
