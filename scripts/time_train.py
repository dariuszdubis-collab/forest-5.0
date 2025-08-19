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

    df = pd.read_csv(args.input, parse_dates=["time"])
    model = train(df, q_low=args.q_low, q_high=args.q_high)
    Path(args.output).write_text(model.to_json())
    # ensure model can be loaded back
    TimeOnlyModel.load(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
