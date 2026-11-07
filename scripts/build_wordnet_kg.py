#!/usr/bin/env python
"""Synset-level WordNet KG from the Princeton WordNet dict files (data.noun, data.verb, data.adj, data.adv).

Each synset is an entity; each pointer between retained synsets is a triple. Lexical pointers keep a
`lexical_` prefix. Entity text is the synset type, its lemmas and its gloss.

    python scripts/build_wordnet_kg.py --dict-dir /path/to/WordNet-3.0/dict --out-dir data/wordnet
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


POS_FILES = {
    "noun": "n",
    "verb": "v",
    "adj": "a",
    "adv": "r",
}

POS_LABELS = {
    "n": "noun_synset",
    "v": "verb_synset",
    "a": "adjective_synset",
    "s": "adjective_satellite_synset",
    "r": "adverb_synset",
}

RELATION_LABELS = {
    "!": "antonym",
    "@": "hypernym",
    "@i": "instance_hypernym",
    "~": "hyponym",
    "~i": "instance_hyponym",
    "#m": "member_holonym",
    "#s": "substance_holonym",
    "#p": "part_holonym",
    "%m": "member_meronym",
    "%s": "substance_meronym",
    "%p": "part_meronym",
    "=": "attribute",
    "+": "derivationally_related_form",
    ";c": "domain_topic",
    "-c": "member_topic",
    ";r": "domain_region",
    "-r": "member_region",
    ";u": "domain_usage",
    "-u": "member_usage",
    "*": "entailment",
    ">": "cause",
    "^": "also_see",
    "$": "verb_group",
    "&": "similar_to",
    "<": "participle_of_verb",
    "\\": "pertainym_or_derived_from_adjective",
}


def synset_id(pos: str, offset: str) -> str:
    return f"wn:{pos}:{offset}"


def clean_text(value: str) -> str:
    return " ".join(value.replace("_", " ").split())


def parse_data_line(line: str, source_pos: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    record, _, gloss = line.rstrip("\n").partition("|")
    fields = record.split()
    offset = fields[0]
    lex_filenum = fields[1]
    ss_type = fields[2]
    word_count = int(fields[3], 16)

    idx = 4
    words: List[str] = []
    for _ in range(word_count):
        words.append(fields[idx])
        idx += 2

    pointer_count = int(fields[idx])
    idx += 1

    node = {
        "id": synset_id(source_pos, offset),
        "offset": offset,
        "pos": source_pos,
        "ss_type": ss_type,
        "type": POS_LABELS.get(ss_type, f"{ss_type}_synset"),
        "name": "; ".join(clean_text(word) for word in words),
        "lemmas": "|".join(words),
        "lex_filenum": lex_filenum,
        "gloss": gloss.strip(),
        "source": "WordNet",
    }

    edges: List[Dict[str, str]] = []
    for _ in range(pointer_count):
        pointer_symbol = fields[idx]
        target_offset = fields[idx + 1]
        target_pos = fields[idx + 2]
        source_target = fields[idx + 3]
        idx += 4

        relation = RELATION_LABELS.get(pointer_symbol, f"wordnet_pointer_{pointer_symbol}")
        edge_scope = "semantic" if source_target == "0000" else "lexical"
        if edge_scope == "lexical":
            relation = f"lexical_{relation}"

        edges.append(
            {
                "relation": relation,
                "display_relation": relation,
                "x_id": node["id"],
                "y_id": synset_id(target_pos, target_offset),
                "source_target": source_target,
                "pointer_symbol": pointer_symbol,
                "edge_scope": edge_scope,
            }
        )

    return node, edges


def iter_data_lines(dict_dir: Path) -> Iterable[Tuple[str, str]]:
    for file_stem, pos in POS_FILES.items():
        data_path = dict_dir / f"data.{file_stem}"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing expected WordNet data file: {data_path}")
        with data_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip() or not line[:1].isdigit():
                    continue
                yield pos, line


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def build_wordnet_kg(dict_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes: Dict[str, Dict[str, str]] = {}
    raw_edges: List[Dict[str, str]] = []

    for source_pos, line in iter_data_lines(dict_dir):
        node, edges = parse_data_line(line, source_pos)
        nodes[node["id"]] = node
        raw_edges.extend(edges)

    kg_rows: List[Dict[str, str]] = []
    skipped_missing_target = 0
    for edge_index, edge in enumerate(raw_edges):
        source = nodes.get(edge["x_id"])
        target = nodes.get(edge["y_id"])
        if source is None or target is None:
            skipped_missing_target += 1
            continue

        kg_rows.append(
            {
                "relation": edge["relation"],
                "display_relation": edge["display_relation"],
                "x_index": str(edge_index),
                "x_id": edge["x_id"],
                "x_type": source["type"],
                "x_name": f"{source['name']} | {source['gloss']}" if source["gloss"] else source["name"],
                "x_source": source["source"],
                "y_index": str(edge_index),
                "y_id": edge["y_id"],
                "y_type": target["type"],
                "y_name": f"{target['name']} | {target['gloss']}" if target["gloss"] else target["name"],
                "y_source": target["source"],
            }
        )

    kg_fields = [
        "relation",
        "display_relation",
        "x_index",
        "x_id",
        "x_type",
        "x_name",
        "x_source",
        "y_index",
        "y_id",
        "y_type",
        "y_name",
        "y_source",
    ]
    node_fields = ["id", "offset", "pos", "ss_type", "type", "name", "lemmas", "lex_filenum", "gloss", "source"]
    rel_fields = ["relation", "count"]

    kg_count = write_csv(out_dir / "wordnet_kg.csv", kg_rows, kg_fields)
    node_count = write_csv(out_dir / "wordnet_nodes.csv", nodes.values(), node_fields)

    relation_counts = Counter(row["relation"] for row in kg_rows)
    relation_rows = [
        {"relation": relation, "count": str(count)}
        for relation, count in sorted(relation_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    write_csv(out_dir / "wordnet_relation_summary.csv", relation_rows, rel_fields)

    print(f"wrote {out_dir / 'wordnet_kg.csv'} ({kg_count} triples)")
    print(f"wrote {out_dir / 'wordnet_nodes.csv'} ({node_count} synsets)")
    print(f"wrote {out_dir / 'wordnet_relation_summary.csv'} ({len(relation_rows)} relations)")
    if skipped_missing_target:
        print(f"skipped missing-target pointers: {skipped_missing_target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/wordnet"))
    args = parser.parse_args()

    build_wordnet_kg(args.dict_dir, args.out_dir)


if __name__ == "__main__":
    main()
