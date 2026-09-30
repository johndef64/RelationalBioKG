# experiments/ablation/ — the E3 ablations, one versioned folder each

Every ablation run writes to its own folder, created by `experiments/e3_ablation.sh`. Two variants are
only ever compared inside one version, so they share machine, code and settings.

```
<TASK>_v<N>/
  MANIFEST.md          one row per launch: date, host, GPU, code commit
  PREREGISTRATION.json the hypotheses, frozen when the version was created
  summary/             ablation_summary.md / .csv
  COMPLETE             present once every variant has all its seeds
  logs/, models/       training logs and checkpoints (not in the repository)
```

Relaunching the same command resumes the latest version of a task if it is not `COMPLETE`, and starts
the next one if it is. `ABL_VERSION=v<N>` forces a version, `ABL_NEW=1` a new one.

## Versions

| version | task | runs | deterministic | GPU | result |
|---|---|---|---|---|---|
| [`DTI_v1`](DTI_v1/MANIFEST.md) | A (DTI) | 13 variants × 5 seeds | no | not logged | pilot; the hypotheses of `DTI_v2` were derived from it |
| [`DTI_v2`](DTI_v2/MANIFEST.md) | A (DTI) | 13 variants × 10 seeds | yes | RTX 5000 Ada | confirmatory: 5/5 primary effects confirmed, 3 equivalences; circularity check in `stratified_no_drugctx.md` |
| [`TREATS_v1`](TREATS_v1/MANIFEST.md) | B (TREATS) | 13 variants × 5 seeds | yes | RTX 5000 Ada | 2/3 primary component effects confirmed; focal loss and disjoint supervision raise MRR (focal: opposite to the pre-registered direction, exploratory); no context block has a detectable effect |

## Pre-registration

The source files are `experiments/prereg/PREREGISTRATION_DTI.json` and `…_TREATS.json`. Their commit
date proves when they were written. Before a version runs, they fix:
- which variants are primary (Holm is applied among them only);
- the direction of their tests;
- which "does not matter" claims are tested for equivalence (TOST, margin 0.02 MRR).

`e3_ablation.sh` copies the task's file into the version folder when the version is created.
`ablation_summary.py` follows it and writes a table of effects with 95% intervals and a verdict per
variant. `DTI_v1` is the pilot and has no pre-registration: applying the hypotheses derived from it
back to it would be circular.

## Reading a version

```bash
python experiments/ablation_summary.py --logdir experiments/ablation/DTI_v2/logs \
                                       --out    experiments/ablation/DTI_v2/summary \
                                       --prereg experiments/ablation/DTI_v2/PREREGISTRATION.json
```

Deltas are against the reference of the same family (`comp_full`, `ctx_full`), with paired tests over
seeds.
