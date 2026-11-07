#!/usr/bin/env python
"""RENKI-KG link prediction with the RENKI head network or a KGE baseline.

Each seed draws its own 70/10/20 edge split. Training uses the relation-balanced MSE on sigmoid
scores with mini-batches, AdamW, and the checkpoint with the best validation AUC (the per-relation
AUC averaged with weights proportional to the number of validation triples).

    python scripts/train_renki.py --embed-path data/renki/emb/cxr_mv7_d128.pq --freeze-embed --name mv7_d128
    python scripts/train_renki.py --dim 128 --name rand_d128
    python scripts/train_renki.py --model transe --name transe
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pnkg.data import relation_balance_weights  # noqa: E402
from pnkg.kge import build_kge_model, kge_logits, kge_optimizer  # noqa: E402
from pnkg.renki import RENKI, FastSampler, auc_by_relation, load_renki_kg  # noqa: E402
from pnkg.training import weighted_mse_loss  # noqa: E402


def rel_weights(rel, n_rel):
    return relation_balance_weights(rel.cpu().numpy(), n_rel).to(rel.device)


def split_edges(X, Y, train_frac, val_frac, seed):
    split = np.random.default_rng(seed).choice(3, size=len(X), p=[train_frac, val_frac, 1 - train_frac - val_frac])
    split = torch.as_tensor(split, device=X.device)
    return [(X[split == k], Y[split == k]) for k in range(3)]


def build_model(args, n, n_rel, embed, device):
    """(model, logits(model, pairs, rel), optimizer)."""
    if args.model == "renki":
        dim = args.dim if embed is None else embed.shape[1]
        model = RENKI(n, n_rel, dim, hidden=tuple(args.hidden), embed=embed,
                      freeze=args.freeze_embed and embed is not None).to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                weight_decay=args.weight_decay, fused=device.type == "cuda")
        return model, lambda m, x, rel: m(x[:, 0], rel, x[:, 1]), opt
    model = build_kge_model(args.model, n, n_rel, args.dim).to(device)
    return model, lambda m, x, rel: kge_logits(m, args.model, x[:, 0], rel, x[:, 1]), kge_optimizer(model, args.model)


@torch.no_grad()
def classification_metrics(model, logits_fn, X, Y, sampler, n_rel, seed, chunk=20000):
    """Relation-balanced AUROC, accuracy and F1 (sigmoid >= 0.5) over the triples and one negative each."""
    model.eval()
    torch.manual_seed(10_000 + seed)
    neg = sampler(X, Y)
    score = lambda x: torch.cat([logits_fn(model, x[i:i + chunk], Y[i:i + chunk]) for i in range(0, len(x), chunk)])
    prob = torch.sigmoid(torch.cat([score(X), score(neg)])).cpu().numpy()
    y_true = np.r_[np.ones(len(X)), np.zeros(len(X))]
    w = rel_weights(Y, n_rel)[Y].repeat(2).cpu().numpy()
    y_pred = (prob >= 0.5).astype(int)
    return dict(auroc_rb=roc_auc_score(y_true, prob, sample_weight=w),
                acc_rb=accuracy_score(y_true, y_pred, sample_weight=w),
                f1_rb=f1_score(y_true, y_pred, sample_weight=w))


def train_one(args, X, Y, n, n_rel, rtype_idx, tset, embed, seed, device):
    torch.manual_seed(seed)
    (Xtr, Ytr), (Xval, Yval), (Xts, Yts) = split_edges(X, Y, args.train_frac, args.val_frac, seed)
    print(f"seed {seed}: train/valid/test triples {len(Xtr)}/{len(Xval)}/{len(Xts)}")

    weights = rel_weights(Ytr, n_rel)
    sampler = FastSampler(rtype_idx, tset, n_rel, device)
    model, logits_fn, opt = build_model(args, n, n_rel, embed, device)

    best_auc, best_epoch, best_state = -1.0, 0, None
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        total = torch.zeros((), device=device)
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            pos, rel = Xtr[idx], Ytr[idx]
            neg = sampler(pos, rel)
            logits = torch.cat([logits_fn(model, pos, rel), logits_fn(model, neg, rel)])
            labels = torch.cat([torch.ones(len(idx), device=device), torch.zeros(len(idx), device=device)])
            loss = weighted_mse_loss(logits, labels, weights[rel].repeat(2))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * len(idx)
        _, val_auc = auc_by_relation(model, Xval, Yval, rtype_idx, tset)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch}: loss {total.item() / len(Xtr):.4f}, val AUC {val_auc:.4f} "
                  f"(best {best_auc:.4f} @ {best_epoch}), {time.perf_counter() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    auc_ts, avg_ts = auc_by_relation(model, Xts, Yts, rtype_idx, tset)
    cls = classification_metrics(model, logits_fn, Xts, Yts, FastSampler(rtype_idx, tset, n_rel, device), n_rel, seed)
    print(f"seed {seed}: best epoch {best_epoch}, val AUC {best_auc:.4f}, test AUC {avg_ts:.4f}, "
          + ", ".join(f"{k} {v:.4f}" for k, v in cls.items()))
    return auc_ts, dict(seed=seed, best_epoch=best_epoch, val_auc=best_auc, test_auc=avg_ts, **cls)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-path", type=Path, default=ROOT / "data" / "renki" / "cxr_all.csv")
    p.add_argument("--embed-path", type=Path, default=None,
                   help="parquet of initial entity embeddings indexed by entity code; none: random of size --dim")
    p.add_argument("--freeze-embed", action="store_true", help="keep the --embed-path embeddings fixed")
    p.add_argument("--name", required=True)
    p.add_argument("--model", choices=["renki", "transe", "distmult", "complex", "rotate"], default="renki")
    p.add_argument("--dim", type=int, default=128, help="embedding size without --embed-path, and of KGE models")
    p.add_argument("--hidden", type=int, nargs="+", default=[256])
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "renki")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    X, Y, n, rdict, rtype_idx, tset, embed, _ = load_renki_kg(args.csv_path, args.embed_path)
    X, Y = X.to(device), Y.to(device)
    tset = {t: v.to(device) for t, v in tset.items()}
    label = {v: k for k, v in rdict.items()}
    if args.model != "renki":
        embed = None
    print(f"{args.name}: {args.model}, init {args.embed_path}, frozen {args.freeze_embed and embed is not None}, "
          f"entities {n}, relations {len(rdict)}, device {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"{args.name}.csv"
    rows = []
    for seed in args.seeds:
        auc_ts, row = train_one(args, X, Y, n, len(rdict), rtype_idx, tset, embed, seed, device)
        rows.append({**{label[k]: v for k, v in sorted(auc_ts.items(), key=lambda kv: label[kv[0]])}, **row})
        pd.DataFrame(rows).set_index("seed").to_csv(out_csv)

    df = pd.DataFrame(rows).set_index("seed")
    print(f"wrote {out_csv}")
    print(df.agg(["mean", "std"]).T.round(4))


if __name__ == "__main__":
    main()
