#!/usr/bin/env python
"""Collects <results>/<method>_seed<seed>/metrics.json into one csv and prints mean and sd per method.

    python scripts/summarize.py results/wordnet --out results/wordnet/summary.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("*_seed*/metrics.json")):
        method, seed = path.parent.name.rsplit("_seed", 1)
        test = json.loads(path.read_text())["test"]
        rows.append({"method": method, "seed": int(seed), **{f"test_{k}": v for k, v in test.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print(df.drop(columns="seed").groupby("method").agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
