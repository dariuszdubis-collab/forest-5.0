#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from forest5.cli import main as forest5_main


def main() -> int:
    p = argparse.ArgumentParser(
        prog="batch-normalize-dir",
        description="Example: normalize all CSVs in a directory to 1H schema",
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--policy", choices=("strict", "pad"), default="pad")
    args = p.parse_args()

    # Delegate to forest5 CLI data normalize command
    return forest5_main(
        [
            "data",
            "normalize",
            "--input-dir",
            str(args.input_dir),
            "--out-dir",
            str(args.out_dir),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
