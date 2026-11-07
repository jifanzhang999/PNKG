import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


@dataclass
class Config:
    model_type: str
    kg_csv: Path
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    relation_balanced: bool
    init_mode: str
    model_ids: List[str]
    model_weights: Optional[List[float]]
    pooling: str
    max_length: int
    batch_size_emb: int
    emb_dir: Path
    d: int
    mlp_hidden: List[int]
    dropout: float
    epochs: int
    lr: Optional[float]
    weight_decay: float
    batch_size: int
    num_workers: int
    out_dir: Path


def parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description="PNKG and KGE baselines for triple classification on a KG.")
    parser.add_argument("--model-type", choices=["pnkg", "transe", "complex", "distmult", "rotate"], default="pnkg")
    parser.add_argument("--kg-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--no-relation-balanced", action="store_false", dest="relation_balanced")
    parser.add_argument(
        "--init-mode",
        choices=["random", "multiview"],
        default="multiview",
        help="random: frozen N(0, 0.02^2) coordinates; multiview: frozen shared coordinates from text encoders",
    )
    parser.add_argument("--model-ids", default="", help="semicolon-separated Hugging Face encoders (one view each)")
    parser.add_argument("--model-weights", default=None, help="semicolon-separated view weights; default uniform")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--batch-size-emb", type=int, default=32)
    parser.add_argument("--emb-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--mlp-hidden", default="256,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None, help="default: 2e-3 for PNKG, model-specific for KGE")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    model_ids = [item.strip() for item in args.model_ids.split(";") if item.strip()]
    if args.model_type == "pnkg" and args.init_mode == "multiview" and not model_ids:
        parser.error("--model-ids is required with --init-mode multiview")
    model_weights = None
    if args.model_weights is not None:
        model_weights = [float(item) for item in args.model_weights.replace(",", ";").split(";") if item.strip()]
        if len(model_weights) != len(model_ids):
            parser.error("--model-weights must have one entry per --model-ids entry")
        if any(w < 0 for w in model_weights) or sum(model_weights) <= 0:
            parser.error("--model-weights must be non-negative with a positive sum")
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        parser.error("train/val/test ratios must sum to 1")

    return Config(
        model_type=args.model_type,
        kg_csv=args.kg_csv,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        relation_balanced=args.relation_balanced,
        init_mode=args.init_mode,
        model_ids=model_ids,
        model_weights=model_weights,
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size_emb=args.batch_size_emb,
        emb_dir=args.emb_dir,
        d=args.d,
        mlp_hidden=[int(x) for x in args.mlp_hidden.split(",") if x.strip()],
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
