# E0 — protocol comparison: v1 (legacy) vs v2 (consolidated)

Mean ± sample std over runs (test set). M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR. *warm* = excluding cold-start test triples. Δ and Welch p on M are vs `v1_fixsplit` (same fixed test set as all later variants); `v1` changes split at every run, so its test sets differ. Configs are those tuned under v1 for all variants (conservative for v2). With ~3 runs, p-values are indicative only.


## DTI — rgcn

| variant | n | AUROC | AUPRC | MRR | Hits@10 | **M** | warm MRR | ΔM vs fixsplit | p(M) | best ep. | time/run (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| + disjoint supervision = **v2** | 3 | 0.779 ± 0.044 | 0.781 ± 0.048 | 0.060 ± 0.042 | 0.097 ± 0.058 | **0.492 ± 0.044** | 0.065 ± 0.046 | — | — | 297 | 38 |

## DTI — baselines (fixed split)

| baseline | n | AUROC | AUPRC | MRR | Hits@10 | **M** | warm MRR | note |
|---|---|---|---|---|---|---|---|---|
| DistMult (no GNN) `v2_lr0.01` | 3 | 0.496 ± 0.016 | 0.499 ± 0.016 | 0.004 ± 0.001 | 0.004 ± 0.000 | **0.300 ± 0.010** | 0.003 ± 0.000 | embeddings only, no message passing (v2 protocol) |
| DistMult (no GNN) `v2_lr0.03` | 3 | 0.546 ± 0.103 | 0.565 ± 0.127 | 0.026 ± 0.040 | 0.043 ± 0.069 | **0.346 ± 0.087** | 0.028 ± 0.044 | embeddings only, no message passing (v2 protocol) |
| DistMult (no GNN) `v2_lr0.1` | 3 | 0.783 ± 0.011 | 0.811 ± 0.014 | 0.124 ± 0.044 | 0.212 ± 0.045 | **0.531 ± 0.025** | 0.137 ± 0.049 | embeddings only, no message passing (v2 protocol) |
| popularity (node degree) | 1 | 0.665 ± 0.000 | 0.683 ± 0.000 | 0.041 ± 0.000 | 0.088 ± 0.000 | **0.423 ± 0.000** | 0.044 ± 0.000 | MRR with random tie-breaking: 0.041 |

### How to read it

- `v1 → + fixed split`: evaluation change only (same training). A large gap means the v1 mean±sd was dominated by split variance.
- `+ select on val M`: effect of choosing the checkpoint with the HPO criterion instead of the validation loss (look at *best ep.*).
- `+ no oversampling`: for R-GCN it should be ~neutral (the loss is a mean over positives and R-GCN averages per relation); for CompGCN it also removes the duplicated target edges from the graph.
- `+ full context graph`: effect of keeping the edges (and 1-to-1 bridges) that random undersampling deleted.
- `+ disjoint supervision`: removes the train/test mismatch of scoring edges that are in the graph.
- A GNN row must beat both baselines; otherwise message passing is not adding information.
