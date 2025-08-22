#!/usr/bin/env python
"""Train and save a TimeOnlyModel from CSV data."""

import argparse
from pathlib import Path

import pandas as pd

from forest5.time_only import TimeOnlyModel, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TimeOnlyModel and export to JSON")
    parser.add_argument("--input", required=True, help="CSV file with 'time' and 'y' columns")
    parser.add_argument("--output", required=True, help="Path to save model JSON")
    parser.add_argument("--q-low", type=float, default=0.1, help="Lower quantile")
    parser.add_argument("--q-high", type=float, default=0.9, help="Upper quantile")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if not {"time", "y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'time' and 'y' columns")
    df["time"] = pd.to_datetime(df["time"])

    model = train(df, q_low=args.q_low, q_high=args.q_high)
    Path(args.output).write_text(model.to_json())
    # ensure model can be loaded back and exposes the decision API
    loaded = TimeOnlyModel.load(args.output)
    # exercise the API (decision + weight) on a single sample
    _ = loaded.decide(df["time"].iloc[0], float(df["y"].iloc[0]))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
