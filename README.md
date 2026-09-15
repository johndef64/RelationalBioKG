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

The link-prediction machinery is run on two target relations, selected after analysing the KG
(native coverage is sufficient — no external DrugBank injection needed):

| Task | Target relation | Meaning | Edges | Drugs | Targets |
|---|---|---|---:|---:|---:|
| **A — DTI** | `Compound → Protein` (`molecularly interacts with`) | mechanistic drug–target | 25,713 | 3,757 | 3,496 proteins |
| **B — TREATS** | `Compound → Disease` (`is substance that treats`) | canonical drug–disease repurposing | 168,157 | 4,328 | 4,480 diseases |

Task A mirrors PathogenKG's `TARGET`; Task B is the classic human repurposing formulation
(Hetionet / DRKG / TxGNN lineage). Both can also be trained jointly (`--task DTI,TREATS`).
There is **no pharmacological drug–drug (DDI)** relation in PKT (chemical–chemical edges are ChEBI
structural only).

---

## The PheKnowLator (PKT) knowledge graph

Human biomedical KG in `dataset/PKT/` as `nodes.json` (~483 MB) + `edges.json` (~4.6 GB), streamed
with `ijson`.

- **780,753 nodes**, **11,132,839 edges**, 0 unresolved endpoints.
- Node types: rna 192,975 · **chemical 150,326** (ChEBI) · variant 144,966 · **protein 96,085** (PR) ·
  go 43,823 · **gene 27,328** (Entrez) · **disease 23,384** (MONDO) · pathway 16,382 · phenotype 17,121.
- `rdf:type` is 40.9% of edges (ontology hierarchy → dropped); predicates are stored as inverse pairs
  (only one direction is kept — the framework re-adds reverse edges internally).

Full analysis in [`analysis/out/`](analysis/out/) (`01_nodes_summary.md`, `02_edges_summary.md`,
`04_relazioni_canoniche_e_scelta_target.md`).

---

## Task subgraphs

Two task-specific subgraphs are extracted from PKT into the exact TSV format the framework expects
(`head <TAB> interaction <TAB> tail`, entity = `Type::id`), sharing an identical molecular **CORE**
(PPI + protein→GO + pathway + gene↔protein bridge):

| Subgraph | file | edges | nodes | run with |
|---|---|---:|---:|---|
| Task A | `dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip` | 1,136,665 | 66,961 | `--task DTI` |
| Task B | `dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip` | 1,853,908 | 93,831 | `--task TREATS` |
| Unified | `dataset/PKT_subgraphs/pkt_unified.tsv.zip` | = Task B | = Task B | `--task DTI,TREATS` |

Two correctness steps during extraction: relations are **renamed type-constrained** (fixing the
`molecularly interacts with` overloading → distinct `PPI` / `DTI` / `COMPOUND_GO`), and each inverse
pair is reduced to one direction (symmetric PPI deduped 617k → 308,704). Built by
[`analysis/06_build_subgraphs.py`](analysis/06_build_subgraphs.py); stats in
[`analysis/out/06_subgraph_stats.md`](analysis/out/06_subgraph_stats.md).

To run the framework on a **different** biomedical KG, produce a TSV in the same
`head <TAB> interaction <TAB> tail` format and point `--tsv` at it — no other change is required.

---

## Repository structure

```
RelationalBioKG/
├── train_and_eval.py            # training & evaluation (--config, protocol v1/v2 flags, distmult baseline)
├── drug_eval.py                 # compound-centric repurposing eval (rebuilds split/graph/config from the run)
├── drug_eval_results.py         # summarise drug-eval outputs
├── tuning_hyperparameter.py     # Bayesian W&B HPO (PKT_HPO_PROTOCOL=v1|v2)
├── expert_review_script.py      # human 3-tier expert review driver (cohort → review sheet)
├── TODO_SERVER.md               # step-by-step run list for the next server session
├── src/                         # encoders (hetero_rgcn/compgcn/rgat), kge_distmult baseline, utils, metrics, params
│
├── dataset/
│   ├── PKT/                     # reference KG: raw PheKnowLator (nodes.json + edges.json, zipped)
│   └── PKT_subgraphs/           # built task subgraphs (+ ablation/, node_labels.tsv)
│
├── analysis/                    # KG analysis & subgraph builders (01–08_*.py) + out/
├── experiments/                 # server run scripts (E0–E4), config, baselines, summaries, README
└── docs/                        # project reports, consolidation plan v2, expert-validation requests
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
GPU — see the TDR note in `experiments/README.md`). Full experiment runs are meant for the **server**.

---

## Quick start

```bash
conda activate gnn

# 1. Build the task subgraphs from the reference PKT KG (one-time)
python analysis/06_build_subgraphs.py

# 2. Train (Task A / DTI, R-GCN, tuned config, consolidated protocol v2)
PYTHONHASHSEED=0 python train_and_eval.py --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip \
  --task DTI --model rgcn --config PKT-DTI-best --runs 5 --epochs 400 \
  --early_stopping --patience 50 --negative_sampling filtered --eval_filtered \
  --oversample_rate 1 --undersample_rate 1.0 --split_seed 42 --select_metric mixed \
  --train_negative_rate 5 --disjoint_supervision 0.3 --warm_eval
#    same command with --model distmult --learning_rate 0.03 = embedding-only baseline (no GNN)

# 3. Compound-centric repurposing on a trained model (Task A ranks proteins);
#    split, graph and config are rebuilt automatically from the model's *_params.json
python drug_eval.py --model_folder models/<your_model_folder> \
  --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI --target_type Protein --compound all
```

Full, scripted pipeline (both tasks, HPO, ablations, repurposing) lives in
**[`experiments/`](experiments/)** — see [`experiments/README.md`](experiments/README.md). The
run list for the next server session is in **[`TODO_SERVER.md`](TODO_SERVER.md)**.

| Exp | What | Script |
|---|---|---|
| **E0** | protocol comparison v1 vs v2 + popularity and DistMult baselines | `experiments/e0_protocol_compare.sh` |
| **E1** | main training: R-GCN vs CompGCN vs **DistMult baseline** | `experiments/e1_main_training.sh` |
| **E2** | Bayesian hyperparameter optimisation (W&B), baseline tuned too | `experiments/e2_hpo_tandem.sh` |
| **E3** | ablations (component machinery + relational context) | `experiments/e3_ablation.sh` |
| **E4** | compound-centric repurposing + interpretability + expert review | `experiments/e4_repurposing.sh` |

### Training & evaluation protocol

Type-constrained **filtered** metrics — AUROC, AUPRC, MRR, Hits@1/3/10 and composite
**M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR** (candidates for ranking = nodes occurring in the target
relation; known positives masked). Focal loss (α=0.25, γ=3.0) + adversarial negative weighting
(α_adv=2.0), edge-level split stratified by target node.

Two sampling/selection protocols are available as flags of `train_and_eval.py` (defaults = v1,
reproduced bit-for-bit); the full audit behind v2 is in
[`docs/piano_consolidamento_v2.md`](docs/piano_consolidamento_v2.md):

| | **v1** (PathogenKG, legacy) | **v2** (consolidated) |
|---|---|---|
| model selection / early stopping | validation loss | validation **M** (same criterion as the HPO) |
| data split | changes at every run | fixed (`--split_seed 42`), only init varies |
| positives / negatives | target oversampling ×5 (does not reweight the loss) | no oversampling, `--train_negative_rate k` |
| background graph | 50% random undersampling (isolates 21% of DTI nodes) | full graph |
| training target edges | inside the message-passing graph | disjoint supervision (`--disjoint_supervision 0.3`) |
| baseline | none | popularity (node degree) + **DistMult without message passing** |
| reproducibility | Python hash seed random per process | `PYTHONHASHSEED=0` |

**Key question answered by E1 under v2:** do the relational GNN encoders (R-GCN, CompGCN) improve
over an equally tuned embedding-only DistMult on the same data, split and protocol?

---

## Interpretability vs validation

A deliberate, honest distinction (see `experiments/README.md` for the full discussion):

- **Held-out filtered metrics** — non-circular quantitative validation (hide known edges, measure
  recovery).
- **`experiments/interpret_predictions.py`** — **interpretability, not validation.** It surfaces KG
  evidence for each novel link (shared PPI / pathway / GO for Task A; molecular meta-path + phenotype
  overlap for Task B). This evidence is *circular* w.r.t. the model (same graph it trained on): it
  explains *why*, it does not prove *true*.
- **`expert_review_script.py`** — human 3-tier review (PathogenKG §5 style): brings knowledge
  *external* to the KG, which is what genuinely breaks the circularity. Produces a review sheet with
  empty `expert_tier / expert_plausible / expert_notes` columns; clinician request docs are in
  [`docs/`](docs/).
- A **time-split** (train on an older PheKnowLator release, test on later-added edges) would be the
  strongest automatic external validation — proposed, not yet built.

---

## Documentation

- [`TODO_SERVER.md`](TODO_SERVER.md) — what to run on the server, in order (Italian).
- [`docs/piano_consolidamento_v2.md`](docs/piano_consolidamento_v2.md) — audit of the training/evaluation protocol, v2 changes and their verification.
- [`docs/report_progetto_RelationalPKT.md`](docs/report_progetto_RelationalPKT.md) — full project report.
- [`docs/report_HPO_risultati_finali.md`](docs/report_HPO_risultati_finali.md) — first HPO (protocol v1; superseded by the v2 HPO).
- [`docs/richiesta_candidati_validazione_medico.md`](docs/richiesta_candidati_validazione_medico.md) — candidate request for a clinician/biologist.
- [`docs/richiesta_candidati_validazione_oncologo.md`](docs/richiesta_candidati_validazione_oncologo.md) — oncology-tailored version (with TCGA / multi-omics cross-check).
- [`experiments/README.md`](experiments/README.md) — experiment plan and run instructions.

---

## Credits & upstream

RelationalBioKG builds directly on the **PathogenKG** codebase and method
(De Filippis, Tommasino, Rinaldi — *PathogenKG: Cross-Species Drug Repurposing via Heterogeneous
Knowledge Graph Link Prediction*, ECML PKDD 2026). The reference knowledge graph used in this repo is
**[PheKnowLator](https://github.com/callahantiff/PheKnowLator)**.

## License

MIT — see [LICENSE](LICENSE).
