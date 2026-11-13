# experiments/block1/ — experiments after release v1.0.0

Experiments of **block 1** (architecture, graph, evaluation, with featureless nodes as in the thesis).
Plan, evaluation and order: `future_works/04_TODO_blocco1.md` (local, not in the repository).

## Rules

1. **Additive only.** Nothing released in v1.0.0 (Zenodo, DOI 10.5281/zenodo.23065358) is modified:
   not `train_and_eval.py`, `src/`, `drug_eval.py`, the existing `experiments/` scripts, configs or
   result summaries. The scripts here may *import* v1.0.0 code, read-only. New model or training code,
   when needed, lives in this folder.
2. **Results stay here.** Every experiment writes into one versioned folder
   `results/<EXP>_v<N>/` with `summary/` (tables, per-seed CSV, provenance JSON), `MANIFEST.md` and
   `COMPLETE`. Never into `models/`, `experiments/logs/` or `experiments/ablation/`. A COMPLETE version
   is never overwritten: a new run is a new version.
3. **Tracked in git, not released.** Code and `results/*/summary/` are committed and pushed. Job
   output (`slurm/*.out`), job zips (`*.zip`), the site environment (`slurm/site_env.sh`) and caches
   (`cache/`) stay local. Pushing to `main` does not create a Zenodo version; Zenodo archives only
   when a GitHub *release* is published, so no release is made from this work until a v2.
4. **Cluster only.** Every run is an `sbatch` job from `slurm/`, submitted from the repository root;
   each job first checks its requirements and ends with one zip of everything it wrote.

## One-time setup on the cluster

```bash
cp experiments/block1/slurm/site_env.example.sh experiments/block1/slurm/site_env.sh   # then edit it
```

## Experiments

| ID | What | Code | Job | Needs | Status |
|---|---|---|---|---|---|
| X0a | MRR by distance from the supervision: PPI hops (Task A), Mondo steps (Task B, step 0 of X7) | `x0a_distance_curve.py` | `slurm/x0a_distance_curve.sbatch` | E1 rank dumps + params, task graphs, `dataset/PKT/edges.zip` | v1 complete |
| X0b | Smoothing of relatives in the embeddings: relative contrast in the decoder space, Dirichlet energy, effective rank | `x0b_embedding_smoothing.py` | `slurm/x0b_embedding_smoothing.sbatch` | E1 checkpoints + params + rank dumps, task graphs; Mondo cache of X0a (optional) | v2 complete (v1 superseded) |
| X0c | Classical link-prediction heuristics (popularity, CN, AA, RA, Katz-3, PPR, typed relatives, collaborative filtering) under the E1 protocol | `x0c_heuristics.py` | `slurm/x0c_heuristics.sbatch` | E1 rank dumps + params, task graphs | v2 complete (v1 superseded) |

| B0 | Prerequisite of phase B: `train_b1.py` trains bit for bit what `train_and_eval.py` trains (E1 R-GCN, seed 42, E1 flags, `--deterministic`; arms v1 / b1 / b1reg) | `test_equivalence.py` | `slurm/b0_equivalence.sbatch` | E1 R-GCN params + metrics, task graphs, GPU | code ready |

Phase A (diagnostics, no training) = X0a, X0b, X0c: independent of each other, run in parallel.
The synthesis of phase A (`results/PHASE_A.md`) and the results are kept locally until they are
reviewed. Per-triple ranks written by X0c (`results/*/ranks/`) are always git-ignored, as for the E1 ranks.

## Training in block 1: `train_b1.py`

Every training of phase B goes through `train_b1.py`. It imports `train_and_eval.py` and calls its
`main()`, so the loop, the loss, the negatives, the evaluation and the selection are the v1.0.0 code,
not a copy. It adds two things from outside, without editing any v1 file:

- `--override KEY=VALUE` changes a hyperparameter of the `--config` block. The block is copied into
  the run folder (`b1_models_params.json`) and the copy is what the v1 code reads;
- `--model <name>` can name a model registered in `b1_models.py`, which starts from the block of a
  v1 model (`base`) plus its own defaults.

It requires `PYTHONHASHSEED=0` and writes only under `results/` (checkpoints git-ignored). The v1
tools that rebuild a model from `src/models_params.json` (`drug_eval.py`, `dump_test_ranks.py`) do
not know the overrides or the new models.

B0 checks the claim before any phase-B run. E1 itself cannot be the bit-level reference: it was
trained before `--deterministic` existed. The reference is therefore `train_and_eval.py` run now
with `--deterministic`, and the distance from E1 seed 42 is reported for information only.

```bash
sbatch --export=ALL,B0_MODE=check experiments/block1/slurm/b0_equivalence.sbatch   # requirements
sbatch experiments/block1/slurm/b0_equivalence.sbatch                               # run (v1)
```

### X0a

```bash
sbatch --export=ALL,X0A_MODE=check experiments/block1/slurm/x0a_distance_curve.sbatch   # requirements
sbatch experiments/block1/slurm/x0a_distance_curve.sbatch                               # run (v1)
```

The E1 rank dumps live in `models/` (git-ignored). Where they are missing but the E1 checkpoints are
there, a GPU job rebuilds them with `experiments/dump_test_ranks.py` (v1.0.0, unchanged), and X0a
can be chained on it:

```bash
J=$(sbatch --parsable experiments/block1/slurm/x0a_dump_ranks.sbatch)
sbatch --dependency=afterok:$J experiments/block1/slurm/x0a_distance_curve.sbatch
```

The check verifies the packages, the six E1 folders (params + 12 rank dumps each), that the 36 dumps
are aligned and reproduce the E1 MRR, that the split rebuilt by `drug_eval.get_dataset_for_drug_eval`
gives exactly the dumped test pairs and training degrees, and (Task B) that `edges.zip` is the
released file. Results: `results/X0a_v1/summary/x0a_{DTI,TREATS}.md`.
