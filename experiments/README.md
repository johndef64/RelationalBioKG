# PKT experiments — PathogenKG method on the human PheKnowLator KG

Reproduces the PathogenKG experimental pipeline on the two PKT subgraphs built by
`analysis/06_build_subgraphs.py`. Same code, same protocol; only the KG (and the target
relation) change. Run on the **server** (env `gnn`, a real GPU — see the TDR note below).

## Tasks
| id | target relation | edges | dataset | `--task` | target node type |
|---|---|---:|---|---|---|
| **A** | drug→protein, pharmacodynamic target | 10,305 | `dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip` | `DTI` | `Protein` |
| **B** | drug→disease (repurposing) | 168,157 | `dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip` | `TREATS` | `Disease` |

Both graphs carry three layers of compound--protein evidence: `DTI` (pharmacodynamic targets,
injected from DrugBank through UniProt cross-references — the Task A target), `DRUG_ADME`
(metabolising enzymes, transporters, plasma carriers) and `CPI_BIOCHEM` (PheKnowLator's own
biochemistry: substrates, cofactors, catalysis). The last two are always context. Why the target had
to be injected: [`TICKET_01_DTI_drug_scope.md`](../TICKET_01_DTI_drug_scope.md).

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
| — (no counterpart upstream) | (new) | **E5** `dump_test_ranks.py` + `stratified_analysis.py` |

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

`experiments/config.sh` defines `FLAGS_V1`, `FLAGS_V2`, `PROTOCOL` (default `v1`, so pass
`PROTOCOL=v2` — E0 has since validated v2 and every result from E1 onwards uses it) and exports
`PYTHONHASHSEED=0` (without it identical commands gave different results).

E0 measured what the consolidation is worth, one correction at a time on Task A: M from 0.476 to
0.648, of which 78% comes from selecting the checkpoint on validation M instead of the loss. The
three architectures span 0.027 on the same task, so the protocol matters roughly sixfold more than
the choice of encoder ([`docs/report_E1.md`](../docs/report_E1.md) for the comparison it enabled).

## Run order
```bash
# one-time: build the subgraphs (if not already present)
python analysis/10_build_dti_drugbank.py    # pharmacological layer (needs dataset/DRUGBANK/ + UniProt)
python analysis/06_build_subgraphs.py
python analysis/07_build_ablation_subgraphs.py --task A
python analysis/07_build_ablation_subgraphs.py --task B

# after ANY dataset change: check that every step still runs (a few minutes, CPU)
bash experiments/smoke_test.sh

# E0 — protocol comparison v1 vs v2 (+ popularity and DistMult baselines). RUN THIS FIRST.
TASKS=DTI CMP_MODELS=rgcn bash experiments/e0_protocol_compare.sh pair   # minimum
bash experiments/e0_protocol_compare.sh                                  # full ladder
python experiments/protocol_compare_summary.py        # -> experiments/protocol_compare_summary.md

# E2 — HPO under protocol v2 (default): W&B projects RelationalPKT-<TASK><SUFFIX>-<model>,
#      models rgcn + compgcn + distmult baseline; best configs -> PKT-<TASK>-best<SUFFIX>
bash experiments/e2_hpo_tandem2.sh    # suffix -v2b, 30 trials/model on Task A, 15 on Task B
#      Task B costs ~10x a Task A trial, so the two no longer share one budget. Use tandem2.
#      (one task only:  bash experiments/e2_hpo_tandem2.sh A)
#      (same budget for both, older behaviour: PKT_HPO_RUNS=30 bash experiments/e2_hpo_tandem.sh)
#      (legacy sweep: PKT_HPO_PROTOCOL=v1 bash experiments/e2_hpo_sweep.sh A)

# E1 — main training & model comparison with the v2 protocol and v2 configs
PROTOCOL=v2 bash experiments/e1_main_training.sh

# E3 — ablations (component machinery + relational context), 13 variants x 5 seeds per task.
#      Resumes by itself: a variant whose log already has 5 completed runs is skipped.
PROTOCOL=v2 ABL_TASK=A bash experiments/e3_ablation.sh all      # ABL_TASK=B for the other task
python experiments/ablation_summary.py --logdir experiments/logs/v2/e3_DTI --out experiments/logs/v2/e3_DTI

# E4 — repurposing + interpretability (point at a model folder from E1)
# folder name = <task>_<dataset>_<timestamp>, e.g.:
bash experiments/e4_repurposing.sh A models/dti_pkt_taskA_dti_<timestamp>
bash experiments/e4_repurposing.sh B models/treats_pkt_taskB_treats_<timestamp>

# (one-time) build readable node labels used by the expert review sheet
python analysis/08_build_node_labels.py

# E5 — per-triple ranks of a finished run, then the stratified analysis (CPU + one forward pass)
PYTHONHASHSEED=0 python experiments/dump_test_ranks.py \
  --model_folder models/dti_pkt_taskA_dti_<timestamp> \
  --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI
python experiments/stratified_analysis.py --ranks models/<run>/test_ranks_DTI.csv \
  --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI \
  --compare "DistMult=models/<other run>/test_ranks_DTI.csv"
```
All knobs (RUNS, EPOCHS, HP_CONFIG, MODELS, …) live in `experiments/config.sh` and can be
overridden inline, e.g. `RUNS=3 EPOCHS=100 bash experiments/e1_main_training.sh` for a quick pass.

## Context ablation variants (E3)

Built by `analysis/07_build_ablation_subgraphs.py`; every variant keeps the target relation intact,
so the models differ only in the biology they can see.

| Task A variant | drops | asks |
|---|---|---|
| `core_ppi` | everything but PPI | how far the bare interactome gets |
| `no_ppi` | protein--protein interactions | does the interactome carry the signal? |
| `no_go` | the three protein--GO relations | does functional annotation carry it? |
| `no_pathway` | pathway membership | do pathways carry it? |
| `no_drugctx` | compound--GO / compound--pathway | how much comes from the drug side? |
| `no_biochem` | `CPI_BIOCHEM` | does PheKnowLator's biochemistry help predict pharmacological targets? |

| Task B variant | drops | asks |
|---|---|---|
| `no_pharma` | `DTI` | what does the injected pharmacological layer add to indication prediction? |
| `no_biochem` | `CPI_BIOCHEM` | same question for the biochemical layer |
| `no_disease_ctx` | disease--phenotype, gene--disease | how much comes from the disease side? |
| `no_ppi` / `no_go` / `no_pathway` | as above | |

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

**Step 3 — capture the human evaluation.** The script writes **two** files: a `_BLIND.csv` for the
reviewer and a `_KEY.csv` that must not be opened before the review is finished. The blind sheet
carries only `item_id, drug_label, prediction_label` plus three EMPTY columns, and its rows are
shuffled across drugs: no rank, no score, no graph evidence, no automatic tier, and the model's
predictions are interleaved with decoys (random candidates and mid-rank ones) in a proportion known
only to the key. Without both provisions a plausibility rate cannot be interpreted.

| column | what the expert writes |
|---|---|
| `expert_tier` | `High` documented or strong mechanistic rationale · `Moderate` indirect but reasonable · `Low` no support |
| `expert_plausible` | `y` / `n` |
| `expert_notes` | mechanism, reference, reasoning (`ADME` for pharmacokinetic pairs) |

Save the filled file, then aggregate it against the key — plausibility per stratum, the difference
between predictions and decoys, held-out positives recovered, and auto-vs-expert agreement:
```bash
python expert_review_script.py aggregate <filled_BLIND.csv> <matching_KEY.csv>
```

**Step 4 — the mechanistic chain behind each prediction.** A pair is not yet a hypothesis: what makes
it one is the route from the predicted target to a disease. `mechanistic_chains.py` searches it in the
**Task B** graph (Task A has no diseases in it) along four typed meta-paths — protein→gene→disease,
protein→gene←disease-by-dysfunction, protein→pathway←gene→disease, protein→PPI→gene→disease — keeping
the shortest, and among equals the ones through the most specific intermediates:
```bash
python experiments/mechanistic_chains.py \
  --review models/<folder>/drug_eval_results/<sheet>_BLIND_compiled.csv \
  --key    models/<folder>/drug_eval_results/<sheet>_KEY.csv
# or, without a review, for any list of pairs (CSV with columns drug,protein as Type::id):
python experiments/mechanistic_chains.py --pairs my_pairs.csv
```
Each chain is flagged by whether the drug already treats the disease it lands on: those explain an
existing indication, the others are repurposing candidates with a stated route. The chains come from
the graph the model trained on, so they belong to interpretability, not validation.

Instructions written for the reviewer are in
[`docs/drug_eval/istruzioni_revisione_taskA.md`](../docs/drug_eval/istruzioni_revisione_taskA.md), the
cohort and its rationale in [`docs/coorte_validazione_taskA.md`](../docs/coorte_validazione_taskA.md),
and the result of the first round in [`docs/report_E4.md`](../docs/report_E4.md): 22.2% of the top-20
predictions plausible against 1.1% of the decoys, all eight recovered held-out targets confirmed, and
an auto-vs-expert agreement of 8.3% — the graph-based triage explains predictions but does not rank
them by truth.

> All experiment steps are bash `.sh` that auto-activate the `gnn` env via `config.sh`
> (`e1`–`e4`, `expert_review.sh`, and `interpret.sh` for standalone interpretability) — so they
> run without activating conda first. The `analysis/*.py` data-prep scripts are the exception:
> run them once with `gnn` active. (Activation is best-effort; if `conda` isn't on PATH the
> script falls back to the current Python.)
The expert's judgement is knowledge *external* to the KG — that is what makes this step a real
(non-circular) validation, with the `kg_evidence`/`auto_tier` columns acting only as triage to
focus the expert on the strongest candidates first.

## E5 — per-triple ranks and stratified analysis

E1 keeps aggregate metrics only, so nothing in the logs answers "who does the model work for".
`dump_test_ranks.py` recomputes the rank of every test triple from a finished run and
`stratified_analysis.py` reads that dump.

- **`dump_test_ranks.py`** rebuilds split, message-passing graph and architecture from the run's own
  `*_params.json` (the same code path as `drug_eval.py`) and ranks with
  `src/evaluation_metrics_filtered._compute_ranks`, i.e. the function E1 itself used: filtered,
  type-constrained, both directions. It prints the MRR it reconstructs, which must equal the run's
  test MRR — treat any mismatch as a bug, not as a new result. Per triple it stores the two ranks,
  the degrees of both endpoints (target-relation and context), the cold-start flag and the top-k
  candidates that outrank the true tail.
- **`stratified_analysis.py`** produces: MRR by degree; what beats the true target (how often the
  higher-ranked candidates are relatives of it — same family by label, specific GO function or
  pathway with at most 50 members, direct interaction — each against a random-candidate baseline);
  redundancy of the held-out edge; cold start; and the three supervision regimes, with
  `--compare "label=other_ranks.csv"` for a paired comparison between models on the same split.

**`PYTHONHASHSEED=0` is mandatory here.** Entity ids follow the iteration order of string sets, so a
different hash seed maps the saved embeddings onto the wrong entities: the run completes and reports
an MRR near zero. `dump_test_ranks.py` refuses to start without it; `config.sh` exports it for the
`.sh` scripts, but these two are run directly with `python`.

Results for Task A are in [`docs/report_stratificata.md`](../docs/report_stratificata.md). Task B
needs the `treats_*` checkpoints, which live on the server: only `metrics`/`params` were copied back.

## Smoke test (already run locally)
`train_and_eval.py` on `pkt_taskA_dti.tsv.zip --task DTI` loads (1,155,994 triples), selects
10,305 DTI targets, splits, trains and runs filtered evaluation end-to-end. The only local
failure is the final full-graph ranking hitting the **Windows TDR GPU watchdog (2 s)** on the
4 GB Quadro M2200 — a local hardware/OS limit, not a code issue. It does **not** occur on the
server GPU (Linux, adequate VRAM).
```bash
# quick local pipeline check (small config), no ranking step:
python train_and_eval.py --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI \
  --config BIOKG-64 --model compgcn --runs 1 --epochs 1
```
