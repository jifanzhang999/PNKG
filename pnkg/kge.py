"""TransE, ComplEx, DistMult and RotatE (PyTorch Geometric) trained as triple classifiers."""

import copy
from typing import Optional

import torch
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE
from tqdm.auto import tqdm

from .config import Config
from .data import TypeSampler, relation_balance_weights, sample_negatives
from .training import evaluate, weighted_mse_loss, write_metrics

KGE_MODELS = {"transe": TransE, "complex": ComplEx, "distmult": DistMult, "rotate": RotatE}

# (optimizer, default learning rate, weight decay)
KGE_OPTIM = {
    "transe": (torch.optim.Adam, 1e-2, 0.0),
    "complex": (torch.optim.Adagrad, 1e-3, 1e-6),
    "distmult": (torch.optim.Adam, 1e-4, 1e-6),
    "rotate": (torch.optim.Adam, 1e-3, 0.0),
}


def build_kge_model(model_type: str, num_entities: int, num_relations: int, dim: int) -> torch.nn.Module:
    kwargs = {"margin": 9.0} if model_type == "rotate" else {}
    return KGE_MODELS[model_type](num_nodes=num_entities, num_relations=num_relations, hidden_channels=dim, **kwargs)


def kge_logits(model: torch.nn.Module, model_type: str, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """PyG scores as classification logits; TransE (score = -distance) is shifted by its margin."""
    scores = model(h, r, t)
    if model_type == "transe":
        return scores + float(model.margin)
    return scores


def kge_optimizer(model: torch.nn.Module, model_type: str, lr: Optional[float] = None) -> torch.optim.Optimizer:
    optim_cls, default_lr, weight_decay = KGE_OPTIM[model_type]
    return optim_cls(model.parameters(), lr=default_lr if lr is None else lr, weight_decay=weight_decay)


def train_kge(
    cfg: Config,
    train_triples: torch.Tensor,
    val_triples: torch.Tensor,
    test_triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
    sampler: TypeSampler,
    device: torch.device,
) -> None:
    model = build_kge_model(cfg.model_type, num_entities, num_relations, cfg.d).to(device)
    optimizer = kge_optimizer(model, cfg.model_type, cfg.lr)
    loader = model.loader(
        head_index=train_triples[:, 0], rel_type=train_triples[:, 1], tail_index=train_triples[:, 2],
        batch_size=cfg.batch_size, shuffle=True,
    )
    rel_weights = None
    if cfg.relation_balanced:
        rel_weights = relation_balance_weights(train_triples[:, 1].cpu().numpy(), num_relations).to(device)
    score = lambda h, r, t: kge_logits(model, cfg.model_type, h, r, t)
    eval_bs = min(cfg.batch_size * 4, 20000)
    chunks = lambda triples: (triples[i : i + eval_bs] for i in range(0, len(triples), eval_bs))
    eval_args = (num_relations, sampler, cfg.relation_balanced, device)

    best_auc, best_epoch, best_val, best_state = float("-inf"), 0, None, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for h, r, t in tqdm(loader, desc=f"{cfg.model_type} epoch {epoch}/{cfg.epochs}", leave=False):
            neg = sample_negatives(torch.stack([h, r, t], dim=1), sampler)
            logits = torch.cat([score(h, r, t), score(neg[:, 0], neg[:, 1], neg[:, 2])])
            labels = torch.cat([torch.ones(len(h), device=device), torch.zeros(len(h), device=device)])
            weights = rel_weights[r].repeat(2) if rel_weights is not None else None
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse_loss(logits, labels, weights)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(h)
            count += len(h)

        model.eval()
        val = evaluate(score, chunks(val_triples), val_triples[:, 1].cpu().numpy(), *eval_args)
        print(f"Epoch {epoch} | loss={total / max(count, 1):.4f} | val: {val}")
        if val["auc"] > best_auc:
            best_auc, best_epoch, best_val = val["auc"], epoch, val
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    test = evaluate(score, chunks(test_triples), test_triples[:, 1].cpu().numpy(), *eval_args)
    write_metrics(cfg.out_dir, best_epoch, best_val, test)
