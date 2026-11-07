import torch

from .config import parse_args, set_seed
from .data import TypeSampler, load_kg, split_edges


def main(argv=None) -> None:
    cfg = parse_args(argv)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    triples, node_ids, rel_names, texts, node_types = load_kg(cfg.kg_csv)
    sampler = TypeSampler(node_types, device)
    train_tr, val_tr, test_tr = split_edges(triples, cfg.seed, cfg.train_ratio, cfg.val_ratio)
    print(f"device: {device} | entities: {len(node_ids)} | relations: {len(rel_names)} | entity types: {sampler.num_types}")
    print(f"train/val/test triples: {len(train_tr)}/{len(val_tr)}/{len(test_tr)}")

    if cfg.model_type != "pnkg":
        from .kge import train_kge

        to_dev = lambda a: torch.from_numpy(a).long().to(device)
        train_kge(cfg, to_dev(train_tr), to_dev(val_tr), to_dev(test_tr), len(node_ids), len(rel_names), sampler, device)
        return

    from .embeddings import build_entity_init
    from .model import PNKG
    from .training import train_pnkg

    init = build_entity_init(cfg, node_ids, texts, device)
    model = PNKG(len(node_ids), len(rel_names), cfg.d, cfg.mlp_hidden, cfg.dropout, init).to(device)
    train_pnkg(cfg, model, train_tr, val_tr, test_tr, len(rel_names), sampler, device)
