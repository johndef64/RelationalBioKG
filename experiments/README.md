# Experiments

Scripts that reproduce every result of the paper on the two task graphs built by
`analysis/06_build_subgraphs.py`. They need the conda env `gnn` and a GPU with enough memory for
full-graph ranking (see [GPU notes](#gpu-notes)).

## Tasks

| id | target relation | edges | dataset | `--task` | target node type |
|---|---|---:|---|---|---|
| **A** | drug→protein, pharmacodynamic target | 10,305 | `dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip` | `DTI` | `Protein` |
| **B** | drug→disease (repurposing) | 168,157 | `dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip` | `TREATS` | `Disease` |

Both graphs carry three layers of compound–protein evidence:
- `DTI`: pharmacodynamic targets from DrugBank via UniProt, the Task A target;
- `DRUG_ADME`: metabolising enzymes, transporters, plasma carriers;
- `CPI_BIOCHEM`: PheKnowLator's own biochemistry (substrates, cofactors, catalysis).

The last two are always context.

## Experiments

| id | what | script |
|---|---|---|
| **E0** | protocol ladder: v1 → v2 one correction at a time, plus popularity and DistMult baselines | `e0_protocol_compare.sh`, `protocol_compare_summary.py` |
| **E2** | Bayesian hyperparameter search of validation M (W&B), same budget for every model | `e2_hpo_tandem2.sh` (calls `e2_hpo_sweep.sh`) |
| **E1** | main comparison: R-GCN, CompGCN, DistMult × 2 tasks × 12 seeds | `e1_run_v2b.sh` (wraps `e1_main_training.sh`), `e1_summary.py` |
| **E3** | component and relational-context ablations, pre-registered | `e3_ablation.sh`, `ablation_summary.py` |
| **E4** | compound-centric ranking, blinded expert review, mechanistic chains | `e4_repurposing.sh`, `../expert_review_script.py`, `mechanistic_chains.py` |
| **E5** | per-triple ranks of every seed, stratified analysis | `dump_test_ranks.py`, `stratified_multiseed.py` |

## Protocol v1 vs v2

The protocol is selected by flags of `train_and_eval.py`. Their defaults reproduce the original
PathogenKG protocol (v1). **Every result of the paper uses v2.**

| | v1 (PathogenKG) | v2 (consolidated) |
|---|---|---|
| checkpoint / early stopping | validation loss | validation M (same as the HPO) |
| split | changes at every run | fixed (`--split_seed 42`) |
| positives / negatives | oversample ×5, 1 negative | no oversampling, `--train_negative_rate k` |
| background graph | 50% random undersampling | full graph |
| training target edges | all inside the message-passing graph | `--disjoint_supervision 0.3` |
| extra reporting | — | `--warm_eval` (no cold start), `--dedup_eval` (no near-duplicate compounds) |

`experiments/config.sh` defines `FLAGS_V1` and `FLAGS_V2`, and selects between them with
`PROTOCOL` (pass `PROTOCOL=v2`). It also exports `PYTHONHASHSEED=0`. `--deterministic` replaces the
GPU scatter-additions with fixed-order reductions, so runs with the same seed are bit-identical; the
ablations use it by default.

## Run order

```bash
# data (one-time)
python analysis/10_build_dti_drugbank.py    # pharmacological layer (needs dataset/DRUGBANK/)
python analysis/06_build_subgraphs.py
python analysis/07_build_ablation_subgraphs.py --task A
python analysis/07_build_ablation_subgraphs.py --task B
python analysis/08_build_node_labels.py     # readable labels for the expert review sheet
bash experiments/smoke_test.sh              # checks that every step runs (a few minutes)

# E0 — protocol ladder
bash experiments/e0_protocol_compare.sh
python experiments/protocol_compare_summary.py        # -> experiments/protocol_compare_summary.md

# E2 — HPO: W&B projects RelationalPKT-<TASK>-v2b-<model>, best configs -> PKT-<TASK>-best-v2b
bash experiments/e2_hpo_tandem2.sh          # 30 trials/model on Task A, 15 on Task B

# E1 — 3 models x 2 tasks x 12 seeds, 1,500-epoch budget
bash experiments/e1_run_v2b.sh              # or: bash experiments/e1_run_v2b.sh A
python experiments/e1_summary.py            # -> experiments/logs/v2/e1_summary.{md,csv}

# E3 — ablations, one versioned folder per run: experiments/ablation/<TASK>_v<N>/
PROTOCOL=v2 ABL_TASK=A ABL_RUNS=10 bash experiments/e3_ablation.sh all
PROTOCOL=v2 ABL_TASK=B ABL_RUNS=5  bash experiments/e3_ablation.sh all
#   ABL_DRY=1 prints the plan; the summary is written to <version>/summary/

# E4 — ranking + interpretability on an E1 model folder
bash experiments/e4_repurposing.sh A models/<task A folder>
bash experiments/e4_repurposing.sh B models/<task B folder>

# E5 — ranks of every seed, then the strata (forward passes only)
PYTHONHASHSEED=0 python experiments/dump_test_ranks.py --model_folder models/<folder> \
  --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI --run all
python experiments/stratified_multiseed.py --task DTI --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip \
  --model R-GCN=models/<rgcn folder> --model CompGCN=models/<compgcn folder> \
  --model DistMult=models/<distmult folder> --reference DistMult \
  --out experiments/logs/v2/stratified_multiseed_DTI.md
```

All knobs (`RUNS`, `EPOCHS`, `MODELS`, …) live in `experiments/config.sh` and can be overridden
inline. The `.sh` scripts activate `gnn` themselves; the `analysis/*.py` scripts and the two E5
scripts are run with `gnn` active. `slurm/` has job files for a Slurm cluster, to adapt to your
environment.

## Context ablation variants (E3)

Built by `analysis/07_build_ablation_subgraphs.py`. Every variant keeps the target relation intact,
so the models differ only in the biology they can see.

| Task A variant | drops | asks |
|---|---|---|
| `core_ppi` | everything but PPI | how far the bare interactome gets |
| `no_ppi` | protein–protein interactions | does the interactome carry the signal? |
| `no_go` | the three protein–GO relations | does functional annotation carry it? |
| `no_pathway` | pathway membership | do pathways carry it? |
| `no_drugctx` | compound–GO / compound–pathway | how much comes from the drug side? |
| `no_biochem` | `CPI_BIOCHEM` | does PheKnowLator's biochemistry help predict pharmacological targets? |

| Task B variant | drops | asks |
|---|---|---|
| `no_pharma` | `DTI` | what does the pharmacological layer add to indication prediction? |
| `no_biochem` | `CPI_BIOCHEM` | same question for the biochemical layer |
| `no_disease_ctx` | disease–phenotype, gene–disease | how much comes from the disease side? |
| `no_ppi` / `no_go` / `no_pathway` | as above | |

The hypotheses of each task are in `experiments/prereg/`, committed before the confirmatory runs,
and are copied into every version folder; `ablation_summary.py` applies them (Holm among the
primary variants, TOST equivalence with a ±0.02 MRR margin). See `ablation/README.md`.

## Evaluation

Type-constrained **filtered** ranking: the candidates are the nodes that occur in the target
relation, and known positives are masked. Metrics are AUROC, AUPRC, MRR, Hits@1/3/10 and the
composite **M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR**. Training uses a focal loss (α=0.25, γ=3.0) with
adversarial negative weighting (α_adv=2.0).

## Interpretability and validation (E4)

- **`interpret_predictions.py` — interpretability, not validation.**
  - For every novel top-k prediction it surfaces supporting structure from the *same* graph the model
    trained on: a PPI neighbour, shared pathway or shared GO term with a known target (Task A); a
    molecular path drug→target→gene→disease and phenotype overlap (Task B).
  - It also assigns an automatic triage tier.
  - This evidence is circular: it explains why a link was predicted, not whether it is true. The one
    non-circular signal it reports is held-out recovery.
- **`mechanistic_chains.py`** searches the route from a predicted target to a disease in the Task B
  graph, along four typed meta-paths, and flags whether the drug already treats that disease. It
  belongs to interpretability as well.
- **`../expert_review_script.py` — blinded expert review, the non-circular step.**
  - It writes a `_BLIND.csv` sheet for the reviewer and a `_KEY.csv` held back until the review is
    finished.
  - The sheet shows drug and protein only, in random order, with no rank, score or evidence.
  - The model's predictions are interleaved with decoys drawn from the middle of the ranking and at
    random from the candidate pool.

```bash
bash experiments/expert_review.sh                        # cohort and settings: top of expert_review_script.py
python expert_review_script.py aggregate <filled_BLIND.csv> <KEY.csv>
python experiments/mechanistic_chains.py --review <filled_BLIND.csv> --key <KEY.csv>
```

| column | what the expert writes |
|---|---|
| `expert_tier` | `High` documented or strong mechanistic rationale · `Moderate` indirect but reasonable · `Low` no support |
| `expert_plausible` | `y` / `n` |
| `expert_notes` | mechanism, reference, reasoning (`ADME` for pharmacokinetic pairs) |

## E5 — per-triple ranks and stratified analysis

- **`dump_test_ranks.py`** rebuilds split, graph and architecture from a run's `*_params.json` and
  ranks with the same function E1 used (`src/evaluation_metrics_filtered._compute_ranks`).
  - It prints the reconstructed MRR, which must equal the run's test MRR.
  - Per triple it stores both ranks, the endpoint degrees, the cold-start flag and the candidates that
    outrank the true tail.
  - `--run all` dumps every seed of a folder; `e1_models_index.py` lists which checkpoint is which
    seed.
- **`stratified_multiseed.py`** reports MRR per stratum as mean ± sd over seeds, with Welch's test
  against a reference model. The strata are:
  - supervision regime;
  - target and drug degree;
  - redundancy of the held-out edge;
  - what outranks the true target, against a random-candidate baseline.
- **`stratified_analysis.py`** does the same for a single run.

**`PYTHONHASHSEED=0` is mandatory when a checkpoint is reloaded.** Entity ids follow the iteration
order of string sets, so a different hash seed maps the saved embeddings onto the wrong entities.
`dump_test_ranks.py` refuses to start without it.

## GPU notes

Full-graph ranking is memory-bound. `PKT_EVAL_BATCH_ROWS` sets the ranking batch: use a large value
(e.g. `1000000`) on a large GPU and a small one (e.g. `50000`) on small or display-attached GPUs. On
Windows, a display-attached GPU can also hit the 2-second TDR watchdog during ranking; for Task B on
such a GPU add `--encode_on_cpu` to `dump_test_ranks.py`.

A quick pipeline check without the ranking step:
```bash
python train_and_eval.py --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI \
  --config BIOKG-64 --model compgcn --runs 1 --epochs 1
```
