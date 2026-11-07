from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def node_text(node_id: str, name: str, node_type: str, source: str) -> str:
    parts = [node_type] if node_type else []
    parts.append(name or node_id)
    if source:
        parts.append(f"source={source}")
    return " | ".join(parts)


def load_kg(csv_path: Path) -> Tuple[np.ndarray, List[str], List[str], Dict[str, str], List[str]]:
    """Reads a KG csv with columns relation, x_id, y_id and optional x_/y_ name, type, source.

    Returns triples (h, r, t) as indices, entity ids, relation names, entity text and entity types.
    Entities and relations are indexed in order of first appearance.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df[(df["x_id"] != "") & (df["y_id"] != "") & (df["relation"] != "")]

    def side(prefix: str) -> pd.DataFrame:
        get = lambda col: df[f"{prefix}_{col}"] if f"{prefix}_{col}" in df else pd.Series("", index=df.index)
        return pd.DataFrame({"id": df[f"{prefix}_id"], "name": get("name"), "type": get("type"), "source": get("source")})

    x, y = side("x"), side("y")
    nodes = pd.concat([x, y]).iloc[np.ravel(np.column_stack([np.arange(len(df)), np.arange(len(df)) + len(df)]))]
    nodes = nodes.drop_duplicates("id", keep="first")

    node_ids = nodes["id"].tolist()
    rel_names = pd.unique(df["relation"]).tolist()
    ent2idx = {e: i for i, e in enumerate(node_ids)}
    rel2idx = {r: i for i, r in enumerate(rel_names)}
    triples = np.stack(
        [df["x_id"].map(ent2idx).to_numpy(), df["relation"].map(rel2idx).to_numpy(), df["y_id"].map(ent2idx).to_numpy()],
        axis=1,
    ).astype(np.int64)

    texts = {
        row.id: node_text(row.id, row.name.strip(), row.type.strip(), row.source.strip())
        for row in nodes.itertuples(index=False)
    }
    types = [t or "unknown" for t in nodes["type"].tolist()]
    return triples, node_ids, rel_names, texts, types


def split_edges(
    triples: np.ndarray, seed: int, train_ratio: float, val_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = triples[np.random.default_rng(seed).permutation(len(triples))]
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


class TripleDataset(Dataset):
    def __init__(self, triples: np.ndarray) -> None:
        self.triples = triples.astype(np.int64)

    def __len__(self) -> int:
        return self.triples.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.triples[idx]


def relation_balance_weights(relations: np.ndarray, num_relations: int) -> torch.Tensor:
    """Per-relation weights proportional to 1 / n_r, scaled to mean one over the given triples."""
    counts = np.bincount(relations, minlength=num_relations).astype(np.float32)
    nonzero = counts > 0
    weights = np.zeros(num_relations, dtype=np.float32)
    weights[nonzero] = float(nonzero.sum()) / counts[nonzero]
    mean_weight = float(weights[relations].mean()) if relations.size else 1.0
    if mean_weight > 0:
        weights /= mean_weight
    return torch.tensor(weights, dtype=torch.float32)


class TypeSampler:
    """Replaces an entity by a uniformly drawn different entity of the same type."""

    def __init__(self, node_types: List[str], device: torch.device) -> None:
        type2idx: Dict[str, int] = {}
        ids = [type2idx.setdefault(t, len(type2idx)) for t in node_types]
        self.type_ids = torch.tensor(ids, dtype=torch.long, device=device)
        counts = torch.bincount(self.type_ids, minlength=len(type2idx))
        self.offsets = torch.zeros(len(type2idx) + 1, dtype=torch.long, device=device)
        self.offsets[1:] = torch.cumsum(counts, dim=0)
        self.sorted_entities = torch.argsort(self.type_ids)
        self.num_entities = len(node_types)
        self.num_types = len(type2idx)

    def __call__(self, entity_idx: torch.Tensor) -> torch.Tensor:
        sampled = torch.randint(0, self.num_entities, entity_idx.shape, device=entity_idx.device)
        ent_types = self.type_ids[entity_idx]
        for type_id in torch.unique(ent_types):
            mask = ent_types == type_id
            start, end = int(self.offsets[type_id]), int(self.offsets[type_id + 1])
            count = end - start
            if count <= 1:
                continue
            pool = self.sorted_entities[start:end]
            candidates = pool[torch.randint(0, count, (int(mask.sum()),), device=entity_idx.device)]
            same = candidates == entity_idx[mask]
            for _ in range(5):
                if not same.any():
                    break
                candidates[same] = pool[torch.randint(0, count, (int(same.sum()),), device=entity_idx.device)]
                same = candidates == entity_idx[mask]
            sampled[mask] = candidates
        return sampled


def sample_negatives(pos: torch.Tensor, sampler: TypeSampler) -> torch.Tensor:
    """One negative per positive: the head or the tail (fair coin) replaced by an entity of the same type."""
    neg = pos.clone()
    corrupt_head = torch.rand(pos.shape[0], device=pos.device) < 0.5
    replacement = torch.randint(0, sampler.num_entities, (pos.shape[0],), device=pos.device)
    if corrupt_head.any():
        replacement[corrupt_head] = sampler(pos[corrupt_head, 0])
    if (~corrupt_head).any():
        replacement[~corrupt_head] = sampler(pos[~corrupt_head, 2])
    neg[corrupt_head, 0] = replacement[corrupt_head]
    neg[~corrupt_head, 2] = replacement[~corrupt_head]
    return neg
