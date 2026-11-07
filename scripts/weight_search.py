#!/usr/bin/env python
"""Grid search of the view weights on the simplex, ranked by mean validation AUROC over seeds.

Arguments after `--` are passed to run.py for every grid point (e.g. --kg-csv, --model-ids, --epochs).

    python scripts/weight_search.py --num-views 4 --step 0.25 --seeds 101 103 --out-dir results/wordnet/weight_search \
        -- --kg-csv data/wordnet/wordnet_kg.csv --model-ids "bert-base-uncased;roberta-base;..."
"""

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def simplex(num_views: int, step: float):
    k = round(1 / step)
    for c in product(range(k + 1), repeat=num_views):
        if sum(c) == k:
            yield tuple(x / k for x in c)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-views", type=int, required=True)
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 103])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("INDEX", "COUNT"))
    p.add_argument("run_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    run_args = args.run_args[1:] if args.run_args[:1] == ["--"] else args.run_args

    jobs = [(w, s) for w in simplex(args.num_views, args.step) for s in args.seeds]
    for weights, seed in jobs[args.shard[0]::args.shard[1]]:
        tag = ";".join(f"{x:g}" for x in weights)
        out = args.out_dir / f"w{'_'.join(f'{x:g}' for x in weights)}_seed{seed}"
        if not (out / "metrics.json").exists():
            print(f"weights {tag}, seed {seed}", flush=True)
            subprocess.run([sys.executable, str(ROOT / "run.py"), *run_args, "--init-mode", "multiview",
                            "--model-weights", tag, "--seed", str(seed), "--out-dir", str(out)], check=True)

    rows = []
    for path in args.out_dir.glob("w*_seed*/metrics.json"):
        m = json.loads(path.read_text())
        name, seed = path.parent.name.rsplit("_seed", 1)
        rows.append({"weights": name[1:].replace("_", ";"), "seed": int(seed),
                     "val_auc": m["val"]["auc"], "test_auc": m["test"]["auc"]})
    table = pd.DataFrame(rows).groupby("weights").agg(val_auc=("val_auc", "mean"), seeds=("seed", "count"))
    table = table.sort_values("val_auc", ascending=False)
    table.to_csv(args.out_dir / "summary.csv")
    print(table.head(10).to_string())
    complete = table[table["seeds"] == len(args.seeds)]
    if len(complete):
        print("selected weights:", complete.index[0])
        (args.out_dir / "best_weights.txt").write_text(complete.index[0] + "\n")


if __name__ == "__main__":
    main()
