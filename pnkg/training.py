import copy
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import Config
from .data import TripleDataset, TypeSampler, relation_balance_weights, sample_negatives

ScoreFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def weighted_mse_loss(logits: torch.Tensor, labels: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    per_sample = F.mse_loss(torch.sigmoid(logits), labels, reduction="none")
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)


@torch.no_grad()
def evaluate(
    score: ScoreFn,
    batches: Iterable[torch.Tensor],
    relations: np.ndarray,
    num_relations: int,
    sampler: TypeSampler,
    relation_balanced: bool,
    device: torch.device,
) -> Dict[str, float]:
    """AUROC, AP, accuracy and F1 (sigmoid >= 0.5) over the positives and one type-matched negative each.

    batches iterates over the evaluation triples; relations holds their relation ids (for the weights).
    """
    rel_weights = relation_balance_weights(relations, num_relations).to(device) if relation_balanced else None
    logits, labels, weights = [], [], []
    for batch in batches:
        batch = batch.to(device)
        pos = score(batch[:, 0], batch[:, 1], batch[:, 2])
        neg_triples = sample_negatives(batch, sampler)
        neg = score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        logits.append(torch.cat([pos, neg]).cpu().numpy())
        labels.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        w = rel_weights[batch[:, 1]].repeat(2).cpu().numpy() if rel_weights is not None else np.ones(2 * len(pos))
        weights.append(w)
    y_prob = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
    y_true = np.concatenate(labels)
    w = np.concatenate(weights)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return {
        "auc": float(roc_auc_score(y_true, y_prob, sample_weight=w)),
        "ap": float(average_precision_score(y_true, y_prob, sample_weight=w)),
        "acc": float(accuracy_score(y_true, y_pred, sample_weight=w)),
        "f1": float(f1_score(y_true, y_pred, sample_weight=w)),
    }


def write_metrics(out_dir: Path, best_epoch: int, val: Dict[str, float], test: Dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"best_epoch": best_epoch, "val": val, "test": test}, indent=2))
    print("TEST:", test)


def train_pnkg(
    cfg: Config,
    model: torch.nn.Module,
    train_tr: np.ndarray,
    val_tr: np.ndarray,
    test_tr: np.ndarray,
    num_relations: int,
    sampler: TypeSampler,
    device: torch.device,
) -> None:
    loader_kwargs = {"num_workers": cfg.num_workers, "pin_memory": device.type == "cuda"}
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(TripleDataset(train_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(TripleDataset(val_tr), batch_size=cfg.batch_size, **loader_kwargs)
    test_loader = DataLoader(TripleDataset(test_tr), batch_size=cfg.batch_size, **loader_kwargs)
    rel_weights = relation_balance_weights(train_tr[:, 1], num_relations).to(device) if cfg.relation_balanced else None
    lr = cfg.lr if cfg.lr is not None else 2e-3

    best_auc, best_epoch, best_val, best_state = -1.0, 0, None, None
    for epoch in range(1, cfg.epochs + 1):
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.weight_decay)
        model.train()
        losses: List[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False):
            batch = batch.to(device)
            pos_logits = model.score(batch[:, 0], batch[:, 1], batch[:, 2])
            neg = sample_negatives(batch, sampler)
            logits = torch.cat([pos_logits, model.score(neg[:, 0], neg[:, 1], neg[:, 2])])
            labels = torch.cat([torch.ones(len(batch), device=device), torch.zeros(len(batch), device=device)])
            weights = rel_weights[batch[:, 1]].repeat(2) if rel_weights is not None else None
            loss = weighted_mse_loss(logits, labels, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        val = evaluate(model.score, val_loader, val_tr[:, 1], num_relations, sampler, cfg.relation_balanced, device)
        print(f"Epoch {epoch} | loss={np.mean(losses):.4f} | val: {val}")
        if val["auc"] > best_auc:
            best_auc, best_epoch, best_val = val["auc"], epoch, val
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    test = evaluate(model.score, test_loader, test_tr[:, 1], num_relations, sampler, cfg.relation_balanced, device)
    write_metrics(cfg.out_dir, best_epoch, best_val, test)
