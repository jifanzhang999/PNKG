#!/usr/bin/env python
"""Shared multi-view coordinates of RENKI-KG entities from the encoder views of scripts/embed_renki.py.

Rows of each view are L2-normalized and combined by multi-view kernel PCA (pnkg.embeddings.shared_coordinates).
The orthonormal coordinates are rescaled to the mean row norm of the raw --scale-ref view. With a single
view in --views, this gives that view's top --dim principal components.

    python scripts/build_renki_views.py --views bert roberta nerbert mentalbert scibert biobert clinicalbert \
        --out data/renki/emb/cxr_mv7_d128.pq
    python scripts/build_renki_views.py --views bert --out data/renki/emb/cxr_pca128_bert.pq
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnkg.embeddings import shared_coordinates  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=Path("data/renki/emb"))
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", default=None, help="one per view; default uniform")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--scale-ref", default="bert")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    read = lambda name: pd.read_parquet(args.emb_dir / f"cxr_{name}.pq")
    views = [read(name) for name in args.views]
    index = views[0].index
    if any(not v.index.equals(index) for v in views):
        raise ValueError("views must share the same entity order")

    torch.manual_seed(args.seed)
    tensors = [F.normalize(torch.from_numpy(v.values.astype("float32")), p=2, dim=1) for v in views]
    z = shared_coordinates(tensors, d=args.dim, weights=args.weights)
    ref = torch.from_numpy(read(args.scale_ref).loc[index].values.astype("float32"))
    z = z * (ref.norm(dim=1).mean() / z.norm(dim=1).mean())

    out = pd.DataFrame(z.numpy(), index=index)
    out.columns = out.columns.astype(str)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"wrote {args.out} {out.shape}")


if __name__ == "__main__":
    main()
