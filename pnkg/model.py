from typing import List, Optional

import torch
import torch.nn as nn


class RelationMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.1) -> None:
        super().__init__()
        dims = [in_dim] + hidden_dims + [1]
        layers: List[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PNKG(nn.Module):
    """Frozen entity coordinates with one MLP head per relation on the concatenated (head, tail) coordinates."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        mlp_hidden: List[int],
        dropout: float = 0.1,
        init_entity_emb: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if init_entity_emb is None:
            self.entity = nn.Embedding(num_entities, dim)
            nn.init.normal_(self.entity.weight, mean=0.0, std=0.02)
        else:
            if init_entity_emb.shape != (num_entities, dim):
                raise ValueError(f"entity init has shape {tuple(init_entity_emb.shape)}, expected ({num_entities}, {dim})")
            self.entity = nn.Embedding.from_pretrained(init_entity_emb.float(), freeze=False)
        self.entity.weight.requires_grad = False
        self.mlps = nn.ModuleList(RelationMLP(2 * dim, mlp_hidden, dropout) for _ in range(num_relations))

    def score(self, h_idx: torch.Tensor, r_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([self.entity(h_idx), self.entity(t_idx)], dim=1)
        out = torch.empty(pair.shape[0], device=pair.device, dtype=torch.float32)
        for rel in torch.unique(r_idx):
            mask = r_idx == rel
            out[mask] = self.mlps[int(rel)](pair[mask])
        return out
