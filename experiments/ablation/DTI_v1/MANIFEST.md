# E3 ablation — Task A (DTI), version v1

| | |
|---|---|
| status | **complete**: 13/13 variants, 5/5 seeds each |
| executed | 2026-09-23 10:24 → 2026-09-24 01:25, one uninterrupted session, variants back to back |
| machine | a GPU machine **other than the server** used for HPO and E1; the GPU model was not logged (runs before 2026-09-24 did not print it) |
| code | repository after commit `f94d416` (2026-09-22), before the deterministic mode existed |
| deterministic | **no** — see "Known issue" below |
| model / config | R-GCN, `PKT-DTI-best-v2b`, protocol v2 (`FLAGS_V2`), 1500-epoch budget, patience 50, 10 training negatives |
| split | `split_seed=42`, 6867 / 1058 / 2380 target triples, identical in every variant |
| seeds | 42, 43, 44, 45, 46 (the same five in every variant) |
| GPU time | 14.5 h |

## Contents

| folder | what |
|---|---|
| `logs/` | the 13 training logs, `e3_<variant>_<timestamp>.log` |
| `models/<variant>/` | params, metrics and the five checkpoints `rgcn_run0..4.pt` of each variant (not in git: large). `../original_folder_names.json` maps each folder to the name it had in `models/` |
| `summary/` | `ablation_summary.md` and `.csv`, from `python experiments/ablation_summary.py --logdir experiments/ablation/DTI_v1/logs --out experiments/ablation/DTI_v1/summary` |

## Variants

| family | variant | what changes (everything else as `*_full`) | graph |
|---|---|---|---|
| comp | `comp_full` | reference | task graph, 1,155,994 edges |
| comp | `comp_no_focal` | focal loss off (α=1, γ=0) | same |
| comp | `comp_no_adv` | adversarial weighting of negatives off (α_adv=0) | same |
| comp | `comp_neg1` | 1 training negative per positive instead of 10 | same |
| comp | `comp_no_disjoint` | supervision edges back in the message-passing graph | same |
| comp | `comp_undersample05` | 50% random undersampling of the context graph | same |
| ctx | `ctx_full` | reference | `ablation/pkt_ablA_full` (1,155,994) |
| ctx | `ctx_core_ppi` | only the interactome kept as context | 319,009 |
| ctx | `ctx_no_ppi` | without protein–protein interactions | 847,290 |
| ctx | `ctx_no_go` | without protein–GO annotation | 874,616 |
| ctx | `ctx_no_pathway` | without pathway membership | 1,008,955 |
| ctx | `ctx_no_drugctx` | without compound–GO and compound–pathway | 771,781 |
| ctx | `ctx_no_biochem` | without the inherited biochemistry (`CPI_BIOCHEM`) | 1,130,281 |

The target relation is untouched in every variant: 10,305 `DTI` edges.

## Audit (2026-09-24)

- every variant differs from its reference in exactly one flag or one dataset, verified from the
  `[i] Protocol` line of each log and from each folder's `rgcn_params.json`;
- no run reached the epoch ceiling (latest selected checkpoint: epoch 1385 of 1500);
- no out-of-memory, no traceback;
- consistency with E1: `comp_full` MRR 0.401 against 0.407 for E1 seeds 0–4.

## Known issue: non-deterministic GPU sums

`comp_full` and `ctx_full` are the same experiment: `pkt_ablA_full` holds the same rows **in the same
order** as the task graph (same content hash), so the split and the recipe coincide. (The comment in
`e3_ablation.sh` claiming a different row order, hence a different split, was wrong.) Their validation
curves agree to the fourth decimal and then drift apart, because R-GCN sums messages and decoder
gradients with atomic GPU additions, whose order is not fixed. On seeds 0–3 the test MRR still agrees
within 0.001; on seed 4 the drift moves the selected checkpoint (epoch 375 against 590) and the MRR
goes from 0.356 to 0.386. The same drift appears against E1, run on the server.

The effect is noise of the same order as several of the ablation effects, and it weakens the paired
tests. From v2 onwards the ablation runs with `--deterministic` (see `src/deterministic_ops.py`), under
which two executions are bit-identical.

## Headline results

Component family, test MRR against `comp_full` 0.401: 1 negative instead of 10 −0.163 (Holm
p = 0.001); focal, adversarial, disjoint supervision and undersampling each between −0.028 and −0.044,
none significant after Holm. Undersampling and returning supervision edges to the graph lower AUROC
significantly (−0.032 and −0.024).

Context family, against `ctx_full` 0.407: removing any single block changes the MRR by at most
−0.028 (compound context, Holm p = 0.054); the interactome and the inherited biochemistry change it by
nothing (−0.0004, +0.002). Keeping only the interactome costs −0.102 (Holm p = 0.032). No single block
is necessary, but they cannot all be removed: the context is redundant.
