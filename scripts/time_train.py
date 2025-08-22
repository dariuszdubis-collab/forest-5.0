#!/usr/bin/env python
"""Train and save a :class:`TimeOnlyModel` from CSV data."""

import argparse
from pathlib import Path

import pandas as pd

from forest5.time_only import TimeOnlyModel, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TimeOnlyModel and export to JSON with metrics"
    )
    parser.add_argument("--input", required=True, help="CSV file with 'time' and 'y' columns")
    parser.add_argument("--output", required=True, help="Path to save model JSON")
    parser.add_argument("--q-low", type=float, default=0.2, help="Lower quantile")
    parser.add_argument("--q-high", type=float, default=0.8, help="Upper quantile")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if not {"time", "y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'time' and 'y' columns")

    model, metrics = train(df, q_low=args.q_low, q_high=args.q_high, return_metrics=True)
    model.save(args.output)
    # ensure model can be loaded back
    TimeOnlyModel.load(args.output)
    print(f"Saved model to {args.output}")
    for horizon, m in metrics.items():
        acc = m["accuracy"]
        brier = m["brier"]
        print(f"h{horizon}: acc={acc:.3f} brier={brier:.3f}")


if __name__ == "__main__":
    main()
