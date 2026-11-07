# PNKG

Code for the WordNet and RENKI-KG experiments: two-stage knowledge-graph learning where entity coordinates are
first estimated from text side information (single- or multi-view kernel PCA of pretrained encoder embeddings)
and then held fixed while relation-specific heads are trained for link prediction. The KGE baselines
(TransE, DistMult, ComplEx, RotatE) are trained and evaluated under the same splits, negatives, objective and
model selection.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.14 and CUDA GPUs. Results are deterministic for a fixed seed on a given GPU model.

## Layout

| path | contents |
|------|----------|
| `pnkg/data.py` | KG loading, edge split, type-matched negative sampling, relation-balanced weights |
| `pnkg/embeddings.py` | encoder views and multi-view kernel PCA (`shared_coordinates`) |
| `pnkg/model.py` | PNKG: frozen entity coordinates and one MLP head per relation |
| `pnkg/training.py` | relation-balanced MSE training, best-validation checkpoint, evaluation |
| `pnkg/kge.py` | KGE baselines (PyTorch Geometric) |
| `pnkg/renki.py` | RENKI head network, relation types and per-relation AUC of Liu et al. (2024) |
| `run.py` | one WordNet run (PNKG or a KGE baseline) |
| `scripts/` | data preparation, RENKI-KG training, weight search, summaries and plots |
| `experiments/` | the full WordNet and RENKI-KG experiments |
| `final_simulations.ipynb` | simulations of multi-view kernel PCA recovery, view weighting and downstream risk |

## WordNet

Build the synset-level graph from the Princeton WordNet dict files
(`data.noun`, `data.verb`, `data.adj`, `data.adv`, available from https://wordnet.princeton.edu):

```bash
python scripts/build_wordnet_kg.py --dict-dir /path/to/WordNet/dict --out-dir data/wordnet
```

This gives 116,566 entities, 33 relations and 373,330 triples. Then

```bash
bash experiments/wordnet.sh
```

runs, on a 60/20/20 random edge split:

1. `search`: the view-weight grid (step 0.25, 35 points) over BERT, RoBERTa, ELECTRA and DeBERTa-v3, selected by
   mean validation AUROC over seeds 101 and 103 (the selection in the paper is `0.25;0.5;0.25;0`);
2. `pnkg`: PNKG with random coordinates, each single view, uniform weights and the selected weights, 10 seeds;
3. `kge`: TransE, DistMult, ComplEx and RotatE, 10 seeds;
4. `summary`: `results/wordnet/summary.csv` and `figures/wordnet.pdf`.

Stages can be run separately, e.g. `bash experiments/wordnet.sh pnkg kge summary`. A single run:

```bash
python run.py --kg-csv data/wordnet/wordnet_kg.csv \
  --model-ids "bert-base-uncased;roberta-base;google/electra-base-discriminator;microsoft/deberta-v3-base" \
  --model-weights "0.25;0.5;0.25;0" --seed 7 --out-dir results/wordnet/selected_seed7
python run.py --kg-csv data/wordnet/wordnet_kg.csv --model-type transe --seed 7 --out-dir results/wordnet/transe_seed7
```

Each run writes `metrics.json` (best epoch, validation and test AUROC, AP, accuracy and F1) to its `--out-dir`.

## RENKI-KG

RENKI-KG is the medical knowledge graph of Liu et al. (2024), *Representation-Enhanced Neural Knowledge
Integration* (RENKI); it is not redistributed here. Place its triples (`cxr_all.csv`, columns `X`, `Y`, `R`)
and entity names (`cxr_str.csv`, columns `X`, `str`) in `data/renki/`, then

```bash
bash experiments/renki.sh
```

This embeds every entity name with seven encoders (BERT, RoBERTa, NER-BERT, MentalBERT, SciBERT, BioBERT,
ClinicalBERT; last-layer `[CLS]`), builds the 128-dimensional multi-view coordinates and the per-view principal
components, and trains on a 70/10/20 edge split for 10 seeds:

- `mv7_d128`, `pca128_<view>`: the RENKI head on frozen multi-view or single-view coordinates;
- `random`: the RENKI head with randomly initialized, trainable embeddings;
- `transe`, `distmult`, `complex`, `rotate`: KGE baselines.

Per-seed results go to `results/renki/<config>.csv` and the plot to `figures/renki.pdf`.

## Simulations

`final_simulations.ipynb` runs the simulation study on synthetic data, so it needs no downloads. It uses the
linear-kernel, row-wise information-plus-noise model $E_s = V_s + m_s^{-1/2}\Xi_s$ with $N=60$ entities,
latent rank $d=4$ and $S=4$ views, and estimates the shared coordinates as the top-$d$ eigenvectors of
$\sum_s w_s E_s E_s^\top / N$.

1. **Subspace recovery vs. entity-code dimension.** Sweeps a common $m \in \{50,\dots,800\}$ under isotropic and
   anisotropic row noise and compares $\lVert\sin\Theta(\widehat Z, Z_\star)\rVert_{\mathrm{op}}$ with the
   $m^{-1/2}$ rate.
2. **Observable inverse-variance weighting.** With heterogeneous views $(m_1,\dots,m_4) = (800,400,200,100)$, it
   compares uniform weights, the plug-in inverse-variance weights of Appendix D (estimated from the observed
   $E_s$ only) and a ground-truth grid-search oracle (used for evaluation only).
3. **End-to-end supervised risk.** Trains relation heads on top of the pretrained coordinates and on the true
   coordinates and plots test MSE against the number of labeled triples $n$, the pretraining dimension $m$ and
   the head size $W$.

Open it in Jupyter, or run it end to end with

```bash
jupyter nbconvert --to notebook --execute final_simulations.ipynb
```

Figures for Experiments 1–2 go to `figs_entity_kernel/` and figures for Experiment 3 go
to `figures/`. Experiment 3 trains with PyTorch and runs 20 seeds in parallel (`N_JOBS = 20`); lower `N_JOBS`
on machines with fewer cores.
