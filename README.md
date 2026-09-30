# RelationalBioKG — Relational Link Prediction on Biomedical Knowledge Graphs for Drug Repurposing

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.0+-3C2179?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**RelationalBioKG is a framework for drug repurposing by heterogeneous knowledge-graph link
prediction: relational GNN encoders (R-GCN / CompGCN) + a DistMult decoder, trained on the
topology of a biomedical KG (lookup embeddings, no node features).** Given a biomedical KG and a
target relation (e.g. drug→target, drug→disease), it trains a link-prediction model and produces a
ranked shortlist of novel candidates, with a full protocol for tuning, ablation, and human review.

The method originates from **PathogenKG** (developed and validated in the *bacterial* domain: STRING
PPI + COG orthology + GO + DrugBank) and generalises the **same code and evaluation protocol** to
any human biomedical KG. The original training protocol is kept available (v1) next to a
consolidated one (v2, see [Training & evaluation protocol](#training--evaluation-protocol)).
Because the encoders are **featureless (topology-only)**, any KG whose nodes lack features can be
plugged in as-is.

### Reference use case in this repo: PheKnowLator

The application shipped and fully worked through here is the human
**[PheKnowLator](https://github.com/callahantiff/PheKnowLator) (PKT)** knowledge graph — a large
open human biomedical KG (chemicals, proteins, genes, GO, pathways, diseases, phenotypes, variants).
It is the concrete case study used to demonstrate the framework end-to-end; the same pipeline runs
on other biomedical KGs of comparable scale (DRKG, Hetionet, TxGNN-KG) by swapping the input TSV.

---

## Two repurposing tasks (on the PheKnowLator case study)

| Task | Target relation | Meaning | Edges | Drugs | Targets |
|---|---|---|---:|---:|---:|
| **A — DTI** | `Compound → Protein` (injected, DrugBank via UniProt) | pharmacodynamic drug–target | 10,305 | 2,487 | 2,188 proteins |
| **B — TREATS** | `Compound → Disease` (`is substance that treats`) | canonical drug–disease repurposing | 168,157 | 4,328 | 4,480 diseases |

Task A mirrors PathogenKG's `TARGET`; Task B is the classic human repurposing formulation
(Hetionet / DRKG / TxGNN lineage). Both can also be trained jointly (`--task DTI,TREATS`).
There is **no pharmacological drug–drug (DDI)** relation in PKT (chemical–chemical edges are ChEBI
structural only).

### Why Task A needed an external layer

PheKnowLator has **no pharmacological source**: its chemical→protein edges come from Reactome
(reaction participation), UniProt catalysts and CTD, so they describe substrates, cofactors and
products. Measured on the first build, the ten hub "compounds" were hydron, water, ATP, ADP,
phosphate and magnesium, carrying 31.4% of that relation, and only 12.5% of its edges involved a
compound with any therapeutic use. Training on it means predicting known biochemistry, not medicine;
filtering it by ChEBI role does not help (it retains 3.9% of the edges, still metabolites; details in
the paper).

The graphs therefore separate **three layers of evidence**, all kept in both tasks:

| Relation | Content | Source | Role |
|---|---|---|---|
| `DTI` | pharmacodynamic drug–target | DrugBank via UniProt | **Task A target**, Task B context |
| `DRUG_ADME` | metabolising enzymes, transporters, plasma carriers | DrugBank via UniProt | context |
| `CPI_BIOCHEM` | substrates, cofactors, products, catalysis | PheKnowLator | context |

The injected layer also matters for Task B: before it, only **4.9%** of the compounds with a
`TREATS` edge had any molecular target in the graph; after it, **34.3%** do, which is what closes the
drug → target → pathway → disease chain.

---

## The PheKnowLator (PKT) knowledge graph

Human biomedical KG in `dataset/PKT/` as `nodes.json` (~483 MB) + `edges.json` (~4.6 GB), zipped
(`nodes.zip`, `edges.zip`, 289 MB together) and streamed with `ijson`. It is the ontology backbone of
**KG-TransomicNet**, published on Hugging Face as
[`johndef64/KG-TransomicNet`](https://huggingface.co/datasets/johndef64/KG-TransomicNet) (folder
`PKT/`); see [Step 0](#step-0--download-the-source-graph) to get it.

- **780,753 nodes**, **11,132,839 edges**, 0 unresolved endpoints.
- Node types: rna 192,975 · **chemical 150,326** (ChEBI) · variant 144,966 · **protein 96,085** (PR) ·
  go 43,823 · **gene 27,328** (Entrez) · **disease 23,384** (MONDO) · pathway 16,382 · phenotype 17,121.
- `rdf:type` is 40.9% of edges (ontology hierarchy → dropped); predicates are stored as inverse pairs
  (only one direction is kept — the framework re-adds reverse edges internally).

Full analysis in [`analysis/out/`](analysis/out/) (`01_nodes_summary.md`, `02_edges_summary.md`,
`02_target_candidates.md`).

---

## Building the task subgraphs

Two task-specific graphs are extracted into the exact TSV format the framework expects
(`head <TAB> interaction <TAB> tail <TAB> source <TAB> type`, entity = `Type::id`), sharing an
identical molecular **CORE** (PPI + protein→GO + pathway + gene↔protein bridge):

| Subgraph | file | edges | nodes | PKT / injected | run with |
|---|---|---:|---:|---|---|
| Task A | `dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip` | 1,155,994 | 69,233 | 98.33% / 1.67% | `--task DTI` |
| Task B | `dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip` | 1,873,237 | 95,111 | 98.97% / 1.03% | `--task TREATS` |
| Unified | `dataset/PKT_subgraphs/pkt_unified.tsv.zip` | = Task B | = Task B | | `--task DTI,TREATS` |

### Step 0 — download the source graph

The two input files are the `PKT/` folder of the Hugging Face dataset
[`johndef64/KG-TransomicNet`](https://huggingface.co/datasets/johndef64/KG-TransomicNet): the ontology
backbone on its own, without the ArangoDB dump or the multi-omics layers, which this project does not
use.

**This happens automatically.** Every script in `analysis/` that reads them calls
`analysis/pkt_source.py`, which downloads a missing file from Hugging Face (public, no login) and
checks its SHA-256 against the release all results were computed from. Files already present are left
untouched. To fetch them up front, or to verify an existing copy:

```bash
python analysis/pkt_source.py
```

The same by hand:

```bash
mkdir -p dataset/PKT
for f in nodes.zip edges.zip; do
  curl -L -o dataset/PKT/$f \
    https://huggingface.co/datasets/johndef64/KG-TransomicNet/resolve/main/PKT/$f
done
sha256sum dataset/PKT/nodes.zip dataset/PKT/edges.zip    # must match the two lines below
# 129f4f1110d56e7695fce0d8e160f1b9c54c5664be8fff5d07f8204a4ccfad7b  nodes.zip  (64,673,549 bytes)
# 24642dec8b220eb9264113c29fc4784a0fea75e70d5bc5c594e1d5ce60fae2cc  edges.zip  (224,508,605 bytes)
```

The same with the Hugging Face CLI: `huggingface-cli download johndef64/KG-TransomicNet --repo-type
dataset --include "PKT/*.zip" --local-dir dataset`. These are the exact files every result in this
repository was computed from.

### Step 1 — the pharmacological layer (`analysis/10_build_dti_drugbank.py`)

Drug–target edges are absent from PheKnowLator and are injected from **DrugBank through the
cross-references published by UniProt**, so no closed resource has to be scraped: UniProt lists, for
every human reviewed protein, the DrugBank drugs it is linked to, and inverting that gives
drug → protein pairs. The DrugBank release is used only for drug names and regulatory status.

```bash
# drug names and status (Kaggle mirror of DrugBank 5.1.10)
curl -L -o dataset/DRUGBANK/drug-bank-5110.zip \
  https://www.kaggle.com/api/v1/datasets/download/devildev89/drug-bank-5110
python analysis/10_build_dti_drugbank.py     # -> dataset/PKT_subgraphs/dti_drugbank_edges.tsv
```

Mapping: proteins map for free, because Protein Ontology identifiers embed the UniProt accession
(`Protein::PR_P08684` = P08684); drugs map by normalised name to ChEBI compounds of the KG. Coverage
and losses are reported in [`analysis/out/10_dti_drugbank_report.md`](analysis/out/).

*Pharmacodynamics vs pharmacokinetics.* UniProt does not say in which role a protein is linked to a
drug — DrugBank lists targets, but also metabolising enzymes, transporters and plasma carriers. Roles
are assigned per edge, by strength of evidence: (1) if the drug's DrugBank record declares proteins
of only one kind, every edge of that drug takes that role; (2) otherwise the protein's own Gene
Ontology annotation decides (xenobiotic metabolism, efflux transport, conjugation → `DRUG_ADME`).
Validated against the roles DrugBank declares: predominant role agreement **87.7%**, Spearman 0.94
between our `DRUG_ADME` counts and DrugBank's enzyme/transporter/carrier counts, and 0.02 between
`DTI` and those counts — the two classes do not contaminate each other. Aromatase, a cytochrome P450
that *is* a drug target, stays on the target side.

### Step 2 — the task graphs (`analysis/06_build_subgraphs.py`)

```bash
python analysis/06_build_subgraphs.py                 # both tasks + unified
python analysis/06_build_subgraphs.py --no-pharma     # variant without the pharmacological layer
python analysis/07_build_ablation_subgraphs.py --task A   # context-ablation variants
python analysis/07_build_ablation_subgraphs.py --task B
```

Two correctness steps during extraction: relations are **renamed type-constrained** (fixing the
`molecularly interacts with` overloading → distinct `PPI` / `CPI_BIOCHEM` / `COMPOUND_GO`), and each
inverse pair is reduced to one direction (symmetric PPI deduped 617k → 308,704). Stats, including the
per-relation origin breakdown, in
[`analysis/out/06_subgraph_stats.md`](analysis/out/06_subgraph_stats.md).

To run the framework on a **different** biomedical KG, produce a TSV in the same
`head <TAB> interaction <TAB> tail` format and point `--tsv` at it — no other change is required.

### Step 3 — check that everything still runs

```bash
bash experiments/smoke_test.sh        # tiny run of every step; SMOKE_GPU=1 to use the GPU
```

---

## Repository structure

```
RelationalBioKG/
├── train_and_eval.py            # training & evaluation (--config, protocol v1/v2 flags, distmult baseline)
├── drug_eval.py                 # compound-centric repurposing eval (rebuilds split/graph/config from the run)
├── drug_eval_results.py         # summarise drug-eval outputs
├── tuning_hyperparameter.py     # Bayesian W&B HPO (PKT_HPO_PROTOCOL=v1|v2)
├── expert_review_script.py      # human 3-tier expert review driver (cohort → review sheet)
├── src/                         # encoders (hetero_rgcn/compgcn/rgat), kge_distmult baseline, utils, metrics, params
│
├── dataset/
│   ├── PKT/                     # reference KG: raw PheKnowLator (nodes.json + edges.json, zipped)
│   └── PKT_subgraphs/           # built task subgraphs (+ ablation/, node_labels.tsv)
│
├── analysis/                    # KG analysis & dataset builders (01–10_*.py) + out/
│   ├── 06_build_subgraphs.py    #   task graphs (three evidence layers)
│   ├── pkt_source.py            #   downloads + verifies dataset/PKT/*.zip from Hugging Face if missing
│   ├── 07_build_ablation_subgraphs.py  #   context-ablation variants (--task A|B)
│   ├── 09_extract_chebi_roles.py       #   ChEBI role inventory (why filtering does not work)
│   └── 10_build_dti_drugbank.py        #   drug--target layer from DrugBank via UniProt
├── experiments/                 # run scripts (E0–E4), config, baselines, summaries, smoke_test.sh
│   ├── mechanistic_chains.py    #   drug → predicted target → gene/pathway → disease routes
│   ├── dump_test_ranks.py       #   per-triple test ranks from a saved run (needs PYTHONHASHSEED=0)
│   └── stratified_analysis.py   #   who the model works for: degree, competitors, redundancy, regimes
└── docs/                        # release notes (docs/release_1.0.0.md)
```

---

## Environment setup

One command builds the conda env **`gnn`** exactly as verified (Python 3.10, PyTorch 2.7.0,
PyG 2.7.0, pinned `requirements.txt`), picking the CUDA build from the NVIDIA driver and checking
imports, GPU kernels and data files at the end:

```bash
bash create_env.sh                    # create or update env "gnn"
CUDA_TAG=cu126 bash create_env.sh     # force a build: cu128 | cu126 | cu118 | cpu
RECREATE=1 bash create_env.sh         # rebuild from scratch
conda activate gnn && wandb login     # W&B is needed for the HPO only
```

`requirements.txt` lists everything except PyTorch and the compiled PyG extensions, whose wheels
depend on the CUDA version and are installed by the script.

**Requirements:** Python 3.10, a CUDA GPU with adequate VRAM (the full-graph ranking needs a real
GPU — see the TDR note in `experiments/README.md`). The full experiments are meant for a GPU server;
`experiments/slurm/` has Slurm job files to adapt.

---

## Quick start

```bash
conda activate gnn

# 0. Inputs (one-time): the PKT graph is downloaded from Hugging Face automatically by the build
#    scripts (or now: python analysis/pkt_source.py); DrugBank names by hand, see Step 1

# 1. Build the task subgraphs (one-time): pharmacological layer, then the graphs
python analysis/10_build_dti_drugbank.py
python analysis/06_build_subgraphs.py
bash experiments/smoke_test.sh            # optional: checks every step still runs

# 2. Train (Task A / DTI, R-GCN, tuned config, protocol v2), as in the paper
PYTHONHASHSEED=0 python train_and_eval.py --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip \
  --task DTI --model rgcn --config PKT-DTI-best-v2b --runs 12 --epochs 1500 \
  --early_stopping --patience 50 --negative_sampling filtered --eval_filtered \
  --oversample_rate 1 --undersample_rate 1.0 --split_seed 42 --select_metric mixed \
  --train_negative_rate auto --disjoint_supervision 0.3 --warm_eval --dedup_eval
#    --model distmult = embedding-only baseline (no GNN); add --deterministic for bit-identical reruns

# 3. Compound-centric repurposing on a trained model (Task A ranks proteins);
#    split, graph and config are rebuilt automatically from the model's *_params.json
python drug_eval.py --model_folder models/<your_model_folder> \
  --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI --target_type Protein --compound all
```

Full, scripted pipeline (both tasks, HPO, ablations, repurposing) lives in
**[`experiments/`](experiments/)** — see [`experiments/README.md`](experiments/README.md).

| Exp | What | Script |
|---|---|---|
| **E0** | protocol comparison v1 vs v2 + popularity and DistMult baselines | `experiments/e0_protocol_compare.sh` |
| **E1** | main training: R-GCN vs CompGCN vs **DistMult baseline** | `experiments/e1_main_training.sh` |
| **E2** | Bayesian hyperparameter optimisation (W&B), baseline tuned too | `experiments/e2_hpo_tandem2.sh` |
| **E3** | ablations (component machinery + relational context) | `experiments/e3_ablation.sh` |
| **E4** | compound-centric repurposing + interpretability + expert review | `experiments/e4_repurposing.sh` |
| **E5** | per-triple ranks of every seed, then stratified analysis | `experiments/dump_test_ranks.py`, `experiments/stratified_multiseed.py` |

```bash
# E5 — who the model actually works for (no retraining; forward passes only)
PYTHONHASHSEED=0 python experiments/dump_test_ranks.py --model_folder models/<run> \
  --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI --run all
python experiments/stratified_multiseed.py --task DTI --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip \
  --model R-GCN=models/<rgcn run> --model DistMult=models/<distmult run> --reference DistMult \
  --out experiments/logs/v2/stratified_multiseed_DTI.md
```

`PYTHONHASHSEED=0` is not optional when a checkpoint is reloaded: entity ids come from the iteration
order of string sets, so a different hash seed silently maps the saved embeddings onto the wrong
entities and the ranks become noise. `dump_test_ranks.py` refuses to start without it, and prints the
MRR it reconstructs so it can be checked against the run's own logs.

### Training & evaluation protocol

Type-constrained **filtered** metrics — AUROC, AUPRC, MRR, Hits@1/3/10 and composite
**M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR** (candidates for ranking = nodes occurring in the target
relation; known positives masked). Focal loss (α=0.25, γ=3.0) + adversarial negative weighting
(α_adv=2.0), edge-level split stratified by target node.

Two sampling/selection protocols are available as flags of `train_and_eval.py` (defaults = v1,
reproduced bit-for-bit); the evidence behind v2 (protocol ladder E0) is reported in the paper:

| | **v1** (PathogenKG) | **v2** (consolidated, used for all results) |
|---|---|---|
| model selection / early stopping | validation loss | validation **M** (same criterion as the HPO) |
| data split | changes at every run | fixed (`--split_seed 42`), only init varies |
| positives / negatives | target oversampling ×5 (does not reweight the loss) | no oversampling, `--train_negative_rate k` |
| background graph | 50% random undersampling (isolates 21% of DTI nodes) | full graph |
| training target edges | inside the message-passing graph | disjoint supervision (`--disjoint_supervision 0.3`) |
| baseline | none | popularity (node degree) + **DistMult without message passing** |
| reproducibility | Python hash seed random per process | `PYTHONHASHSEED=0` |

**Key question answered by E1 under v2:** do the relational GNN encoders (R-GCN, CompGCN) improve
over an equally tuned embedding-only DistMult on the same data, split and protocol? In short: on
Task A they recognise plausible pairs better (AUROC 0.844 vs 0.786) but do not rank the true target
better (MRR 0.408 vs 0.407), except on drugs with context but no known target (about 15× the
baseline's MRR); on Task B the baseline wins on every metric (MRR 0.692 vs 0.348). Tables in
`experiments/logs/v2/`.

---

## Interpretability vs validation

A deliberate, honest distinction (see `experiments/README.md` for the full discussion):

- **Held-out filtered metrics** — non-circular quantitative validation (hide known edges, measure
  recovery).
- **`experiments/interpret_predictions.py`** — **interpretability, not validation.** It surfaces KG
  evidence for each novel link (shared PPI / pathway / GO for Task A; molecular meta-path + phenotype
  overlap for Task B). This evidence is *circular* w.r.t. the model (same graph it trained on): it
  explains *why*, it does not prove *true*.
- **`experiments/mechanistic_chains.py`** — the route from a predicted target to a disease
  (protein → gene / pathway / interaction partner → disease), searched in the Task B graph and ranked
  shortest and least hubby first. Each chain says whether the drug already treats that disease, which
  splits "mechanism for a known indication" from "repurposing candidate with a route". Same
  circularity caveat as above.
- **`expert_review_script.py`** — human 3-tier review (PathogenKG §5 style): brings knowledge
  *external* to the KG, which is what genuinely breaks the circularity. Produces a blinded review
  sheet (no rank, score or automatic tier) interleaved with random and mid-rank decoys, plus the key
  to score it with `aggregate`. On Task A, 22.2% of the top-20 predictions for nine drugs were rated
  plausible against 1.1% of the decoys ($p=4\cdot10^{-7}$), and the automatic KG triage agreed with
  the expert on only 8.3% of them — graph evidence explains a prediction, it does not validate it.

---

## Documentation

- [`experiments/README.md`](experiments/README.md) — experiment plan and run instructions.
- `experiments/logs/v2/` — result summaries behind the paper's tables: `e1_summary.{md,csv}` (E1),
  `stratified_multiseed_*.md` and `_per_seed.csv` (stratified analysis), `popularity_*.log`.
- `experiments/ablation/<TASK>_v<N>/` — for each ablation: `summary/`, `MANIFEST.md`,
  `PREREGISTRATION.json` (the frozen hypotheses) and, for DTI_v2, the circularity check.
- `experiments/prereg/` — the pre-registrations as committed before the confirmatory runs.
- [`docs/release_1.0.0.md`](docs/release_1.0.0.md) and [`CITATION.cff`](CITATION.cff) — release notes and how to cite.

---

## Credits & upstream

RelationalBioKG builds directly on the **PathogenKG** codebase and method
(De Filippis, Tommasino, Rinaldi — *PathogenKG: Cross-Species Drug Repurposing via Heterogeneous
Knowledge Graph Link Prediction*, ECML PKDD 2026). The reference knowledge graph used in this repo is
**[PheKnowLator](https://github.com/callahantiff/PheKnowLator)**.

## License

MIT — see [LICENSE](LICENSE).
