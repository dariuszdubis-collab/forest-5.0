"""Streamlit page for running a simple EMA crossover backtest."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from forest5.backtest.engine import run_backtest
from forest5.config import BacktestSettings, RiskSettings, StrategySettings


def _load_df(uploaded_file: bytes | None) -> pd.DataFrame | None:
    """Load DataFrame from uploaded CSV file.

    The CSV is expected to contain at least the columns
    ``time``, ``open``, ``high``, ``low`` and ``close``.
    """
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    return df


def render() -> None:
    """Render the backtest page with a parameter form."""
    st.header("EMA Backtest")

    uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
    df = _load_df(uploaded_file)

    with st.form("backtest-form"):
        fast = st.number_input("Fast EMA", min_value=1, max_value=200, value=12, step=1)
        slow = st.number_input("Slow EMA", min_value=1, max_value=400, value=26, step=1)
        capital = st.number_input(
            "Initial capital", min_value=0.0, value=100_000.0, step=1_000.0, format="%.2f"
        )
        risk = st.number_input(
            "Risk per trade", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.3f"
        )
        max_dd = st.number_input(
            "Max drawdown", min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f"
        )
        atr_period = st.number_input("ATR period", min_value=1, max_value=200, value=14, step=1)
        atr_multiple = st.number_input(
            "ATR multiple", min_value=0.1, value=2.0, step=0.1, format="%.1f"
        )
        submitted = st.form_submit_button("Run backtest")

    if submitted:
        if df is None or df.empty:
            st.warning("Please upload a valid CSV file with OHLC data before running the backtest.")
            return

        settings = BacktestSettings(
            strategy=StrategySettings(fast=int(fast), slow=int(slow)),
            risk=RiskSettings(
                initial_capital=float(capital),
                risk_per_trade=float(risk),
                max_drawdown=float(max_dd),
            ),
            atr_period=int(atr_period),
            atr_multiple=float(atr_multiple),
        )

        result = run_backtest(df, settings)
        st.subheader("Equity curve")
        fig = px.line(result.equity_curve, title="Equity")
        st.plotly_chart(fig, use_container_width=True)

        if result.trades.trades:
            trades_df = pd.DataFrame([t.__dict__ for t in result.trades.trades])
            st.subheader("Trades")
            st.dataframe(trades_df)


if __name__ == "__main__":
    render()
