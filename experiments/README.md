# PKT experiments — PathogenKG method on the human PheKnowLator KG

Reproduces the PathogenKG experimental pipeline on the two PKT subgraphs built by
`analysis/06_build_subgraphs.py`. Same code, same protocol; only the KG (and the target
relation) change. Run on the **server** (env `gnn`, a real GPU — see the TDR note below).

## Tasks
| id | target relation | dataset | `--task` | target node type |
|---|---|---|---|---|
| **A** | drug→protein (DTI) | `dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip` | `DTI` | `Protein` |
| **B** | drug→disease (repurposing) | `dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip` | `TREATS` | `Disease` |

## Mapping PathogenKG README → these experiments
| PathogenKG (paper/README) | repo script | here |
|---|---|---|
| §3.1 Training & Eval (Table 4) | `train_and_eval.py` | **E1** `e1_main_training.sh` |
| §3.2 pipeline validation on DRKG | `tuning_dataset_drkg.py` | already validated upstream — reused as-is |
| §3.2/3.3 Bayesian HPO (metric M) | `tuning_hyperparameter.py` | **E2** `e2_hpo_sweep.sh` |
| §3.4 Compound-centric repurposing (Table 5) | `drug_eval.py`, `drug_eval_results.py` | **E4** `e4_repurposing.sh` |
| interpretability of novel links (KG evidence) | (new) | **E4** `interpret_predictions.py` |
| §5 tiered biological plausibility (expert) | `drug_eval_script.py` (manual cohort) | **E4** `expert_review_script.py` |
| ablations (loss/sampling/model/context) | flags of `train_and_eval.py` | **E3** `e3_ablation.sh` |
| §4 KG stats (Table 2) | `kg_stats_visualization.py` | optional, on the subgraph TSVs |

## Adaptations made to the original code (backward compatible)
- `train_and_eval.py`: added `--config` (pick `BIOKG-128` etc. from `src/models_params.json`;
  the module default was hardcoded to the bacterial `pathogen31-cmp-gene`). Fixed a `--dry_run`
  `NameError`.
- `drug_eval.py`: added `--target_type` (was hardcoded to `ExtGene`; PKT needs `Protein`/`Disease`).
- `tuning_hyperparameter.py`: dataset/task/W&B entity now read from env vars
  (`PKT_TSV`, `PKT_TASK`, `WANDB_ENTITY`, `WANDB_PROJECT`, `PKT_HPO_*`).

## Protocol v1 vs v2 (read first)
The pipeline has two training protocols, selected by flags of `train_and_eval.py` whose defaults
reproduce the legacy one (verified bit-for-bit). Full rationale and evidence:
[`docs/piano_consolidamento_v2.md`](../docs/piano_consolidamento_v2.md).

| | v1 (legacy PathogenKG) | v2 (consolidated) |
|---|---|---|
| checkpoint / early stopping | validation loss | validation M (same as HPO) |
| split | changes at every run | fixed (`--split_seed 42`) |
| positives / negatives | oversample ×5, 1 negative | no oversampling, `--train_negative_rate k` |
| background graph | 50% random undersampling | full graph |
| training target edges | all inside the message-passing graph | `--disjoint_supervision 0.3` |
| extra reporting | — | `--warm_eval` (no cold-start), best epoch, time |

`experiments/config.sh` defines `FLAGS_V1`, `FLAGS_V2`, `PROTOCOL` (default `v1` until E0 validates
v2) and exports `PYTHONHASHSEED=0` (without it identical commands gave different results).

## Run order
```bash
# one-time: build the subgraphs (if not already present)
python analysis/06_build_subgraphs.py

# E0 — protocol comparison v1 vs v2 (+ popularity and DistMult baselines). RUN THIS FIRST.
TASKS=DTI CMP_MODELS=rgcn bash experiments/e0_protocol_compare.sh pair   # minimum
bash experiments/e0_protocol_compare.sh                                  # full ladder
python experiments/protocol_compare_summary.py        # -> experiments/protocol_compare_summary.md

# E2 — HPO under protocol v2 (default): W&B projects RelationalPKT-<TASK>-v2-<model>,
#      models rgcn + compgcn + distmult baseline; best configs -> PKT-<TASK>-best-v2
bash experiments/e2_hpo_tandem.sh
#      (legacy sweep: PKT_HPO_PROTOCOL=v1 bash experiments/e2_hpo_sweep.sh A)

# E1 — main training & model comparison with the v2 protocol and v2 configs
PROTOCOL=v2 bash experiments/e1_main_training.sh

# E3 — ablations (component machinery + relational context). NOTE: e3_ablation.sh still encodes the
#      v1 component list; it is to be adapted to v2 after E0 (see docs/piano_consolidamento_v2.md §4)
bash experiments/e3_ablation.sh          # or: component | context

# E4 — repurposing + interpretability (point at a model folder from E1)
# folder name = <task>_<dataset>_<timestamp>, e.g.:
bash experiments/e4_repurposing.sh A models/dti_pkt_taskA_dti_<timestamp>
bash experiments/e4_repurposing.sh B models/treats_pkt_taskB_treats_<timestamp>

# (one-time) build readable node labels used by the expert review sheet
python analysis/08_build_node_labels.py
```
All knobs (RUNS, EPOCHS, HP_CONFIG, MODELS, …) live in `experiments/config.sh` and can be
overridden inline, e.g. `RUNS=3 EPOCHS=100 bash experiments/e1_main_training.sh` for a quick pass.

## Evaluation protocol
Edge-level stratified split, multi-seed, focal loss (α=0.25, γ=3.0) + adversarial negative
weighting (α_adv=2.0), type-constrained **filtered** evaluation (candidates = nodes that occur in
the target relation, known positives masked). Sampling/selection: v1 or v2, see above. Metrics: AUROC, AUPRC, MRR, Hits@1/3/10 and composite
**M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR**.

## Interpretability vs validation (E4)
Two distinct things, do not conflate them:

- **`experiments/interpret_predictions.py` — INTERPRETABILITY (not validation).** For every
  novel top-k prediction it surfaces supporting structure from the *same* KG the model
  trained on (Task A: PPI neighbour / shared pathway / shared GO with a known target;
  Task B: molecular path drug→target→gene→disease + phenotype overlap) and an AUTO triage
  tier. This evidence is **circular** w.r.t. the model — it explains *why* the model
  predicted a link, it does not prove it true. The only non-circular signal it reports is
  **held-out recovery** (`is_test_target`: positives removed from training and re-found).

- **`expert_review_script.py` — human 3-tier review (PathogenKG §5 style).** Runs the
  cohort through drug_eval, attaches the interpretability evidence + readable node labels,
  and writes a **review sheet** with empty `expert_tier / expert_plausible / expert_notes`
  columns. The expert brings knowledge *external* to the KG → this breaks the circularity.
  Works without a curated list (ranks all compounds) or with a pinned cohort. Capture flow:
  fill the sheet in a spreadsheet, then `python expert_review_script.py aggregate <filled.csv>`.

For an *automatic* external (non-circular) validation, use a **time-split** (train on an older
PheKnowLator release, test against edges added later) or an unintegrated external DB — that is
the strongest evidence of genuine discovery; the interpretability script is not that.

## E4 in practice — step by step

**Prereq (one-time):** build the readable node-label lookup used to make the review sheet
human-legible (turns `Protein::PR_Q6JQN1` → *"acyl-CoA dehydrogenase … (human)"*):
```bash
python analysis/08_build_node_labels.py       # -> dataset/PKT_subgraphs/node_labels.tsv
```

**Step 1 — rank + interpret (automatic).** `e4_repurposing.sh` runs `drug_eval.py` (ranks
every candidate target per compound with the mature model), then `interpret_predictions.py`
(KG evidence + held-out recovery on the novel links):
```bash
bash experiments/e4_repurposing.sh A models/dti_pkt_taskA_dti_<timestamp>       # DTI  → Proteins
bash experiments/e4_repurposing.sh B models/treats_pkt_taskB_treats_<timestamp> # TREATS → Diseases
```
Outputs land in `models/<folder>/drug_eval_results/`:
`*_rankings_*.json` (ranked predictions), `*_summary_*.csv` (per-compound metrics),
`*_interpreted.csv` (novel links + `kg_evidence` + `auto_tier` + `held_out_recovered`).

**Step 2 — expert review sheet (PathogenKG §5 style).** Edit the config block at the top of
`expert_review_script.py` (repo root):
```python
TASK         = "A"                                   # "A" = DTI (proteins), "B" = TREATS (diseases)
MODEL_FOLDER = os.path.join("models", "dti_pkt_taskA_dti_<timestamp>")
TOPK         = 50
CANDIDATES   = []           # [] = review ALL compounds; or pin a cohort:
                            #   ["Compound::CHEBI_28918", "Compound::CHEBI_45783", ...]
```
then run it via the wrapper (activates `gnn` for you) — it ranks the cohort (all compounds if
`CANDIDATES` is empty), attaches the KG evidence and readable labels, and writes the review sheet:
```bash
bash experiments/expert_review.sh
# -> models/<folder>/drug_eval_results/expert_review_task<A|B>_<ts>.csv
```

**Step 3 — capture the human evaluation.** The sheet has, per candidate link, the model
columns (`drug_label, prediction_label, confidence, kg_evidence, auto_tier, held_out_recovered`)
plus three EMPTY columns the expert fills in Excel/Sheets:

| column | what the expert writes |
|---|---|
| `expert_tier` | `1` known/plausible mechanism · `2` possible · `3` implausible |
| `expert_plausible` | `y` / `n` |
| `expert_notes` | mechanism, reference, reasoning |

Save the filled file, then aggregate it (expert-tier distribution, auto-vs-expert agreement,
% plausible):
```bash
bash experiments/expert_review.sh aggregate models/<folder>/drug_eval_results/expert_review_taskA_<ts>_filled.csv
```

> All experiment steps are bash `.sh` that auto-activate the `gnn` env via `config.sh`
> (`e1`–`e4`, `expert_review.sh`, and `interpret.sh` for standalone interpretability) — so they
> run without activating conda first. The `analysis/*.py` data-prep scripts are the exception:
> run them once with `gnn` active. (Activation is best-effort; if `conda` isn't on PATH the
> script falls back to the current Python.)
The expert's judgement is knowledge *external* to the KG — that is what makes this step a real
(non-circular) validation, with the `kg_evidence`/`auto_tier` columns acting only as triage to
focus the expert on the strongest candidates first.

## Smoke test (already run locally)
`train_and_eval.py` on `pkt_taskA_dti.tsv.zip --task DTI` loads (1,136,665 triples), selects
25,713 DTI targets, splits, trains and runs filtered evaluation end-to-end. The only local
failure is the final full-graph ranking hitting the **Windows TDR GPU watchdog (2 s)** on the
4 GB Quadro M2200 — a local hardware/OS limit, not a code issue. It does **not** occur on the
server GPU (Linux, adequate VRAM).
```bash
# quick local pipeline check (small config), no ranking step:
python train_and_eval.py --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI \
  --config BIOKG-64 --model compgcn --runs 1 --epochs 1
```
