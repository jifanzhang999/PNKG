import re
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import Config


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


@torch.no_grad()
def encode_texts(
    model_id: str, texts: List[str], batch_size: int, max_length: int, pooling: str, device: torch.device
) -> torch.Tensor:
    """L2-normalized last-layer embeddings (mean-pooled or [CLS]) of each text."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()
    out: List[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"embed {model_id}"):
        enc = tokenizer(texts[i : i + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc.pop("token_type_ids", None)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state
        emb = hidden[:, 0, :] if pooling == "cls" else mean_pool(hidden, enc["attention_mask"])
        out.append(F.normalize(emb, p=2, dim=1).cpu())
    return torch.cat(out, dim=0)


def load_or_encode(model_id: str, node_ids: List[str], texts: Dict[str, str], cfg: Config, device: torch.device) -> torch.Tensor:
    cfg.emb_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.emb_dir / f"node_emb_{re.sub(r'[^a-zA-Z0-9_.-]+', '_', model_id)}_{cfg.pooling}.pt"
    if path.exists():
        print("loading cached:", path)
        return torch.load(path, map_location="cpu")
    emb = encode_texts(model_id, [texts[n] for n in node_ids], cfg.batch_size_emb, cfg.max_length, cfg.pooling, device)
    torch.save(emb, path)
    print("saved:", path, tuple(emb.shape))
    return emb


@torch.no_grad()
def shared_coordinates(
    views: Sequence[torch.Tensor],
    d: int,
    weights: Optional[Sequence[float]] = None,
    oversample: int = 32,
    niter: int = 3,
) -> torch.Tensor:
    """Multi-view kernel PCA.

    With column-centered views E_m of dimension p_m and weights alpha_m summing to one, returns the
    top-d eigenvectors (orthonormal columns, n x d) of sum_m (alpha_m / p_m) E_m E_m^T, computed by
    randomized subspace iteration. A single view gives its top-d principal components.
    """
    n = views[0].shape[0]
    if weights is None:
        weights = [1.0] * len(views)
    total = float(sum(weights))
    alphas = [float(w) / total for w in weights]
    centered = [v.float() - v.float().mean(dim=0, keepdim=True) for v in views]

    def apply_kernel(mat: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(mat)
        for alpha, emb in zip(alphas, centered):
            out += (alpha / emb.shape[1]) * (emb @ (emb.T @ mat))
        return out

    q = torch.randn(n, min(d + oversample, n), dtype=torch.float32)
    for _ in range(niter):
        q, _ = torch.linalg.qr(apply_kernel(q), mode="reduced")
    b = q.T @ apply_kernel(q)
    evals, evecs = torch.linalg.eigh(0.5 * (b + b.T))
    order = torch.argsort(evals, descending=True)
    z, _ = torch.linalg.qr(q @ evecs[:, order[:d]], mode="reduced")
    return z


def build_entity_init(cfg: Config, node_ids: List[str], texts: Dict[str, str], device: torch.device) -> Optional[torch.Tensor]:
    if cfg.init_mode == "random":
        return None
    views = [load_or_encode(m, node_ids, texts, cfg, device).float() for m in cfg.model_ids]
    print("view weights:", dict(zip(cfg.model_ids, cfg.model_weights or [1.0] * len(views))))
    return shared_coordinates(views, d=cfg.d, weights=cfg.model_weights).contiguous()
