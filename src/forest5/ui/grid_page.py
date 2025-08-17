from __future__ import annotations

import plotly.express as px
import streamlit as st

from forest5.backtest.grid import run_grid
from forest5.cli import _parse_range
from forest5.utils.io import read_ohlc_csv


def _parse_range_safe(spec: str) -> list[int]:
    """Parse a range specification, showing an error on failure."""
    try:
        return list(_parse_range(spec))
    except Exception as exc:  # pragma: no cover - UI validation
        st.error(f"Błędny zakres '{spec}': {exc}")
        return []


def render() -> None:
    """Render the Streamlit page for EMA grid search."""
    st.title("Grid Search")

    csv_path = st.text_input("Ścieżka do CSV", "demo.csv")
    symbol = st.text_input("Symbol", "SYMB")

    with st.form("grid_form"):
        fast_spec = st.text_input("Fast values", "5:20:1")
        slow_spec = st.text_input("Slow values", "10:60:2")
        capital = st.number_input("Initial capital", value=100_000.0)
        risk = st.number_input("Risk per trade", value=0.01, min_value=0.0, max_value=1.0, step=0.001)
        n_jobs = st.number_input("Parallel jobs", value=1, min_value=1, step=1)
        submitted = st.form_submit_button("Run grid")

    if not submitted:
        return

    try:
        df = read_ohlc_csv(csv_path)
    except Exception as exc:  # pragma: no cover - UI validation
        st.error(f"Nie można wczytać CSV: {exc}")
        return

    fast_values = _parse_range_safe(fast_spec)
    slow_values = _parse_range_safe(slow_spec)
    if not fast_values or not slow_values:
        return

    with st.spinner("Running grid..."):
        results = run_grid(
            df,
            symbol=symbol,
            fast_values=fast_values,
            slow_values=slow_values,
            capital=capital,
            risk=risk,
            n_jobs=int(n_jobs),
        )

    st.success(f"Zakończono. {len(results)} kombinacji.")
    st.dataframe(results)

    metric_options = [c for c in ("rar", "sharpe") if c in results.columns]
    if metric_options:
        metric = st.selectbox("Metric", metric_options)
        pivot = results.pivot(index="slow", columns="fast", values=metric)
        fig = px.imshow(
            pivot,
            labels={"x": "Fast", "y": "Slow", "color": metric.upper()},
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", csv_bytes, file_name="grid_results.csv", mime="text/csv")


if __name__ == "__main__":  # pragma: no cover
    render()
