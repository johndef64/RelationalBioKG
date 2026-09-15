# E0 — protocol comparison: v1 (legacy) vs v2 (consolidated)

Mean ± sample std over runs (test set). M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR. *warm* = excluding cold-start test triples. Δ and Welch p on M are vs `v1_fixsplit` (same fixed test set as all later variants); `v1` changes split at every run, so its test sets differ. Configs are those tuned under v1 for all variants (conservative for v2). With ~3 runs, p-values are indicative only.


## DTI — rgcn

| variant | n | AUROC | AUPRC | MRR | Hits@10 | **M** | warm MRR | ΔM vs fixsplit | p(M) | best ep. | time/run (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (legacy) | 3 | 0.811 ± 0.015 | 0.813 ± 0.018 | 0.135 ± 0.019 | 0.191 ± 0.022 | **0.542 ± 0.018** | 0.147 ± 0.020 | +0.008 | 0.642 | 38 | 54 |
| + fixed split | 3 | 0.803 ± 0.024 | 0.805 ± 0.028 | 0.127 ± 0.015 | 0.182 ± 0.022 | **0.534 ± 0.022** | 0.138 ± 0.016 | — | — | 37 | 53 |
| + disjoint supervision = **v2** | 3 | 0.936 ± 0.001 | 0.948 ± 0.001 | 0.436 ± 0.013 | 0.558 ± 0.016 | **0.741 ± 0.005** | 0.474 ± 0.013 | +0.207 | 0.002 | 292 | 183 |

## DTI — baselines (fixed split)

| baseline | n | AUROC | AUPRC | MRR | Hits@10 | **M** | warm MRR | note |
|---|---|---|---|---|---|---|---|---|
| DistMult (no GNN) `v2_lr0.01` | 3 | 0.901 ± 0.001 | 0.932 ± 0.001 | 0.587 ± 0.003 | 0.662 ± 0.002 | **0.788 ± 0.001** | 0.638 ± 0.004 | embeddings only, no message passing (v2 protocol) |
| DistMult (no GNN) `v2_lr0.03` | 3 | 0.907 ± 0.001 | 0.936 ± 0.001 | 0.598 ± 0.005 | 0.683 ± 0.004 | **0.795 ± 0.002** | 0.650 ± 0.006 | embeddings only, no message passing (v2 protocol) |
| DistMult (no GNN) `v2_lr0.1` | 3 | 0.899 ± 0.004 | 0.929 ± 0.002 | 0.551 ± 0.002 | 0.651 ± 0.001 | **0.772 ± 0.001** | 0.600 ± 0.002 | embeddings only, no message passing (v2 protocol) |
| popularity (node degree) | 1 | 0.735 ± 0.000 | 0.710 ± 0.000 | 0.117 ± 0.000 | 0.188 ± 0.000 | **0.478 ± 0.000** | 0.126 ± 0.000 | MRR with random tie-breaking: 0.118 |

### How to read it

- `v1 → + fixed split`: evaluation change only (same training). A large gap means the v1 mean±sd was dominated by split variance.
- `+ select on val M`: effect of choosing the checkpoint with the HPO criterion instead of the validation loss (look at *best ep.*).
- `+ no oversampling`: for R-GCN it should be ~neutral (the loss is a mean over positives and R-GCN averages per relation); for CompGCN it also removes the duplicated target edges from the graph.
- `+ full context graph`: effect of keeping the edges (and 1-to-1 bridges) that random undersampling deleted.
- `+ disjoint supervision`: removes the train/test mismatch of scoring edges that are in the graph.
- A GNN row must beat both baselines; otherwise message passing is not adding information.
