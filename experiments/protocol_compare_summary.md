# E0 — protocol comparison: v1 (legacy) vs v2 (consolidated)

Mean ± sample std over runs (test set). M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR. *warm* = excluding cold-start test triples. Δ and Welch p on M are vs `v1_fixsplit` (same fixed test set as all later variants); `v1` changes split at every run, so its test sets differ. Configs are those tuned under v1 for all variants (conservative for v2). With ~3 runs, p-values are indicative only.


## DTI — rgcn

| variant | n | AUROC | AUPRC | MRR | Hits@10 | **M** | warm MRR | ΔM vs fixsplit | p(M) | best ep. | time/run (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy) | 3 | 0.725 ± 0.019 | 0.751 ± 0.023 | 0.076 ± 0.009 | 0.127 ± 0.019 | **0.476 ± 0.016** | 0.084 ± 0.010 | +0.001 | 0.951 | 35 | 24 |
| + fixed split | 3 | 0.722 ± 0.024 | 0.745 ± 0.024 | 0.081 ± 0.014 | 0.131 ± 0.023 | **0.475 ± 0.019** | 0.091 ± 0.016 | — | — | 37 | 25 |
| + select on val M | 3 | 0.800 ± 0.014 | 0.838 ± 0.010 | 0.287 ± 0.007 | 0.352 ± 0.011 | **0.610 ± 0.009** | 0.321 ± 0.008 | +0.135 | 0.002 | 298 | 143 |
| + no oversampling, 5 train negatives | 3 | 0.799 ± 0.014 | 0.836 ± 0.010 | 0.284 ± 0.008 | 0.348 ± 0.009 | **0.608 ± 0.009** | 0.318 ± 0.008 | +0.133 | 0.002 | 293 | 120 |
| + full context graph | 3 | 0.816 ± 0.011 | 0.850 ± 0.009 | 0.296 ± 0.009 | 0.361 ± 0.013 | **0.622 ± 0.009** | 0.327 ± 0.010 | +0.147 | 0.001 | 290 | 126 |
| + disjoint supervision = **v2** | 3 | 0.848 ± 0.005 | 0.879 ± 0.003 | 0.317 ± 0.004 | 0.427 ± 0.005 | **0.648 ± 0.001** | 0.349 ± 0.004 | +0.173 | 0.004 | 290 | 92 |

## DTI — baselines (fixed split)

| baseline | n | AUROC | AUPRC | MRR | Hits@10 | **M** | warm MRR | note |
|---|---|---|---|---|---|---|---|---|
| popularity (node degree) | 1 | 0.665 ± 0.000 | 0.683 ± 0.000 | 0.041 ± 0.000 | 0.088 ± 0.000 | **0.423 ± 0.000** | 0.044 ± 0.000 | MRR with random tie-breaking: 0.041 |

### How to read it

- `v1 → + fixed split`: evaluation change only (same training). A large gap means the v1 mean±sd was dominated by split variance.
- `+ select on val M`: effect of choosing the checkpoint with the HPO criterion instead of the validation loss (look at *best ep.*).
- `+ no oversampling`: for R-GCN it should be ~neutral (the loss is a mean over positives and R-GCN averages per relation); for CompGCN it also removes the duplicated target edges from the graph.
- `+ full context graph`: effect of keeping the edges (and 1-to-1 bridges) that random undersampling deleted.
- `+ disjoint supervision`: removes the train/test mismatch of scoring edges that are in the graph.
- A GNN row must beat both baselines; otherwise message passing is not adding information.
