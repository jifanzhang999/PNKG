"""RENKI head network, relation types and per-relation AUC of Liu et al. (2024), used for RENKI-KG."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn.kge.base import KGEModel

# Entity types required by each relation; the entity type is the first character of its code
# (C: UMLS concept, X: phecode, R: RxNorm).
RELATION_TYPES = {
    "XMP": ("C", "X"),
    "RMP": ("C", "R"),
    "CSM": ("C", "C"),
    "CRL": ("C", "C"),
    "CBR": ("C", "C"),
    "XSM": ("X", "X"),
    "XRL": ("X", "X"),
    "IND": ("R", "X"),
    "SID": ("R", "X"),
}
ENTITY_TYPES = ("C", "X", "R")


def mlp(d_in, hidden, d_out=1):
    dims = [d_in, *hidden, d_out]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(ReLU())
    return Sequential(*layers)


class RENKI(KGEModel):
    """One ReLU network g_k: R^{2d} -> R per relation k, scoring [u_h, u_t]."""

    def __init__(self, n_nodes, n_rel, dim, hidden=(256,), embed=None, freeze=False):
        super().__init__(n_nodes, n_rel, dim)
        if embed is not None:
            self.node_emb = Embedding.from_pretrained(torch.tensor(embed).float(), freeze=freeze)
        self.rel = ModuleList(mlp(2 * dim, hidden) for _ in range(n_rel))

    def forward(self, head, rel, tail):
        h = torch.cat([self.node_emb(head), self.node_emb(tail)], dim=1)
        out = torch.cat([net(h) for net in self.rel], dim=1)
        return out.gather(1, rel[:, None]).squeeze(1)


def load_renki_kg(csv_path, embed_path=None):
    """Triples of a csv with columns X (head), Y (tail), R (relation), with entities in sorted order.

    Returns pairs [n, 2], relation ids, #entities, relation-name -> id, relation id -> (head type, tail type),
    entity ids by type, the initial embedding (rows aligned with the entities, or None) and the entity codes.
    """
    df = pd.read_csv(csv_path)
    codes = sorted(set(df["X"]).union(df["Y"]))
    cdict = dict(zip(codes, range(len(codes))))
    rdict = dict(zip(df["R"].unique(), range(df["R"].nunique())))
    X = torch.as_tensor(np.stack([df["X"].map(cdict).values, df["Y"].map(cdict).values], axis=1), dtype=torch.long)
    y = torch.as_tensor(df["R"].map(rdict).values, dtype=torch.long)
    rtype_idx = {rdict[k]: v for k, v in RELATION_TYPES.items() if k in rdict}
    tset = {t: torch.as_tensor([cdict[c] for c in codes if c[0] == t], dtype=torch.long) for t in ENTITY_TYPES}
    embed = pd.read_parquet(embed_path).loc[codes].values if embed_path else None
    return X, y, len(cdict), rdict, rtype_idx, tset, embed, codes


def neg_sample(pos, rel, rtype_idx, tset):
    """Replaces the head or the tail (fair coin) of each pair by a uniform entity of the type its relation requires."""
    neg = pos.clone()
    for r, (a, b) in rtype_idx.items():
        idx = rel == r
        n = int(idx.sum())
        if n == 0:
            continue
        ht = torch.randint(high=2, size=(n,), device=pos.device)
        ra = torch.randint(high=len(tset[a]), size=(n,), device=pos.device)
        rb = torch.randint(high=len(tset[b]), size=(n,), device=pos.device)
        rnd = torch.stack([tset[a][ra], tset[b][rb]], dim=1)
        neg[idx, ht] = rnd[range(n), ht]
    return neg


class FastSampler:
    """Vectorized neg_sample with the same distribution, for training batches."""

    def __init__(self, rtype_idx, tset, n_rel, device):
        types = sorted(tset)
        self.pool = torch.cat([tset[t] for t in types])
        sizes = torch.as_tensor([len(tset[t]) for t in types], device=device)
        offsets = torch.cumsum(sizes, 0) - sizes
        tid = {t: i for i, t in enumerate(types)}
        head = torch.as_tensor([tid[rtype_idx[k][0]] for k in range(n_rel)], device=device)
        tail = torch.as_tensor([tid[rtype_idx[k][1]] for k in range(n_rel)], device=device)
        self.size = torch.stack([sizes[head], sizes[tail]], 1)
        self.offset = torch.stack([offsets[head], offsets[tail]], 1)

    def __call__(self, pos, rel):
        side = torch.randint(2, (len(pos),), device=pos.device)
        size = self.size[rel, side]
        pick = self.offset[rel, side] + (torch.rand(len(pos), device=pos.device) * size).long().clamp_max(size - 1)
        neg = pos.clone()
        neg[torch.arange(len(pos), device=pos.device), side] = self.pool[pick]
        return neg


@torch.no_grad()
def score_cpu(model, x, rel, chunk=20000):
    was_training = model.training
    model.eval()
    s = torch.cat([model(x[i:i + chunk, 0], rel[i:i + chunk], x[i:i + chunk, 1]).cpu() for i in range(0, len(x), chunk)])
    model.train(was_training)
    return s


def auc_by_relation(model, pos, rel, rtype_idx, tset):
    """({relation id: AUC}, the average AUC weighted by the number of triples per relation)."""
    neg = neg_sample(pos, rel, rtype_idx, tset)
    ps, ns = score_cpu(model, pos, rel), score_cpu(model, neg, rel)
    rel = rel.cpu()
    auc, total = {}, 0.0
    for r in rtype_idx:
        idx = rel == r
        n = int(idx.sum())
        if n > 0:
            auc[r] = roc_auc_score([1] * n + [0] * n, torch.cat([ps[idx], ns[idx]]))
            total += auc[r] * n
    return auc, total / len(rel)
