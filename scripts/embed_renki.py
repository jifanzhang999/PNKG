#!/usr/bin/env python
"""Text views of RENKI-KG entities: the last-layer [CLS] state of each entity name, one name at a time.

Reads cxr_all.csv (X, Y, R) and cxr_str.csv (X, str) and writes <out-dir>/cxr_<name>.pq per encoder
(rows indexed by entity code, in order of first appearance in cxr_all.csv).

    python scripts/embed_renki.py --model bert roberta nerbert mentalbert scibert biobert clinicalbert
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

ENCODERS = {
    "bert": "google-bert/bert-base-uncased",
    "roberta": "FacebookAI/roberta-base",
    "nerbert": "dslim/bert-base-NER",
    "mentalbert": "ourafla/mental-health-bert-finetuned",
    "scibert": "allenai/scibert_scivocab_uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}


def entity_names(kg_csv: Path, str_csv: Path) -> pd.DataFrame:
    edges = pd.read_csv(kg_csv, usecols=["X", "Y"], dtype=str)
    names = pd.read_csv(str_csv, usecols=["X", "str"], dtype=str, keep_default_na=False)
    names = dict(zip(names["X"], names["str"].str.strip()))
    codes = pd.unique(np.column_stack([edges["X"].values, edges["Y"].values]).ravel())
    return pd.DataFrame({"X": codes, "str": [names.get(c) or c for c in codes]})


@torch.no_grad()
def embed_cls(texts, repo: str, device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(device).eval()
    out = []
    for s in tqdm(texts, desc=repo):
        ids = tokenizer(s, return_tensors="pt").input_ids.to(device)
        out.append(model(ids).last_hidden_state[0, 0].cpu().numpy())
    return np.stack(out).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kg-csv", type=Path, default=Path("data/renki/cxr_all.csv"))
    p.add_argument("--str-csv", type=Path, default=Path("data/renki/cxr_str.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/renki/emb"))
    p.add_argument("--model", nargs="+", choices=list(ENCODERS), default=list(ENCODERS))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes = entity_names(args.kg_csv, args.str_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.model:
        path = args.out_dir / f"cxr_{name}.pq"
        if path.exists() and not args.overwrite:
            print(f"{name}: {path} exists")
            continue
        emb = pd.DataFrame(embed_cls(nodes["str"].tolist(), ENCODERS[name], device), index=pd.Index(nodes["X"], name="X"))
        emb.columns = emb.columns.astype(str)
        emb.to_parquet(path)
        print(f"wrote {path} {emb.shape}")


if __name__ == "__main__":
    main()
