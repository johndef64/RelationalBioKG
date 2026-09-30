# experiments/ablation/ — every E3 run, one versioned folder each

Everything an ablation run produces lives in its own folder here: logs, checkpoints, summary and a
manifest of where and how it ran. Nothing goes to `models/` (reserved for E1) or `experiments/logs/`.
The folders are created by `experiments/e3_ablation.sh`; see its header for the details.

```
<TASK>_v<N>/
  MANIFEST.md    one row per launch: date, host, GPU, code commit, what was run
  settings.env   the settings the version was started with (a resume with others is refused)
  logs/          e3_<variant>_<timestamp>.log
  models/<variant>/   params, metrics, checkpoints rgcn_run<i>.pt   (not in git)
  summary/       ablation_summary.md / .csv
  COMPLETE       present once every variant has all its seeds
```

Relaunching the same command resumes the latest version of that task if it is not `COMPLETE`, and
starts the next one if it is. `ABL_VERSION=v<N>` forces a version, `ABL_NEW=1` forces a new one.
Two variants are only ever compared inside one version, so they share machine, code and settings.

## Versions

| version | task | status | deterministic | where it ran | notes |
|---|---|---|---|---|---|
| [`DTI_v1`](DTI_v1/MANIFEST.md) | A (DTI) | complete, 13 × 5 seeds | no | a machine other than the server, 23–24/09/2026 | first complete Task A ablation; GPU sums not reproducible, see the manifest |
| [`DTI_v2`](DTI_v2/MANIFEST.md) | A (DTI) | complete, 13 × 10 seeds | yes | iknos-gpu2 (RTX 5000 Ada), 24–25/09/2026, `--deterministic` | confirmatory run: 5/5 primary effects confirmed, 3 equivalences; circularity check in `stratified_no_drugctx.md`. Checkpoints not in git |
| [`TREATS_v1`](TREATS_v1/MANIFEST.md) | B (TREATS) | complete, 13 × 5 seeds | yes | iknos-gpu2 (RTX 5000 Ada), 24–25/09/2026, `--deterministic` | 2/3 primary component effects confirmed; focal loss and disjoint supervision raise MRR (focal: opposite to the pre-registered direction, exploratory); no context block detectable |

Not here, on purpose: the partial Task A run on the server (10/13 variants, interrupted on
21/09/2026) and the partial Task B run (3/13). They used other hardware and the non-deterministic
code, and must not be merged with any version above. If they are copied from the server, keep them
apart (for instance `archives/e3_server_partial/`).

In git, each version keeps only `summary/`, `MANIFEST.md`, `PREREGISTRATION.json`, `COMPLETE` and
the circularity tables; logs and checkpoints are archived separately.

## Pre-registration

`experiments/prereg/PREREGISTRATION_DTI.json` and `…_TREATS.json` (tracked by git since before the
confirmatory runs, so the commit date proves when they were written) fix, **before** a version is run, which
variants are primary (Holm is applied among them only), the direction of their tests, and which
"does not matter" claims are tested for equivalence (TOST, margin 0.02 MRR). `e3_ablation.sh` copies
the task's file into the version folder the moment the version is created, keeping its timestamp, and
`ablation_summary.py` follows it and writes a table of effects with 95% intervals and a verdict per
variant. Without the file, the summary behaves as before: every variant primary, two-sided, Holm over
the family.

`DTI_v1` is the pilot and has no pre-registration: the hypotheses for `DTI_v2` were derived from it.
Applying them back to `DTI_v1` would be circular; its summary stays the plain one.

## Reading a version

```bash
python experiments/ablation_summary.py --logdir experiments/ablation/DTI_v1/logs \
                                       --out    experiments/ablation/DTI_v1/summary
```
Deltas are against the reference of the same family (`comp_full`, `ctx_full`), with paired tests over
seeds and Holm correction.
