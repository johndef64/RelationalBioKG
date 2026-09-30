# E1 — relational GNN encoders vs embedding-only DistMult baseline

Test set, mean ± sample std over runs (same split, same protocol for all models). M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR. Best mean per row in **bold**. p = Welch two-sided t-test of each GNN vs DistMult. *warm MRR* excludes cold-start test triples (an endpoint with no edge in the training graph).


## Task DTI

| Metric | DistMult (no GNN) (n=12) | R-GCN (n=12) | CompGCN (n=12) | p R-GCN vs DistMult | p CompGCN vs DistMult |
|---|---|---|---|---|---|
| AUROC | 0.786 ± 0.005 | **0.844 ± 0.004** | 0.833 ± 0.004 | <0.001 | <0.001 |
| AUPRC | 0.842 ± 0.003 | **0.879 ± 0.003** | 0.875 ± 0.002 | <0.001 | <0.001 |
| MRR | 0.407 ± 0.009 | **0.408 ± 0.016** | 0.352 ± 0.013 | 0.83 | <0.001 |
| Hits@1 | **0.375 ± 0.010** | 0.373 ± 0.019 | 0.294 ± 0.016 | 0.76 | <0.001 |
| Hits@3 | 0.420 ± 0.009 | **0.420 ± 0.015** | 0.378 ± 0.011 | 0.94 | <0.001 |
| Hits@10 | 0.465 ± 0.008 | **0.472 ± 0.010** | 0.460 ± 0.009 | 0.059 | 0.19 |
| M | 0.657 ± 0.004 | **0.684 ± 0.007** | 0.657 ± 0.005 | <0.001 | 0.74 |
| warm_MRR | 0.449 ± 0.010 | **0.450 ± 0.017** | 0.388 ± 0.014 | 0.86 | <0.001 |
| dedup_MRR | 0.405 ± 0.009 | **0.408 ± 0.016** | 0.351 ± 0.013 | 0.61 | <0.001 |
| dedup_stereo_MRR | 0.405 ± 0.009 | **0.408 ± 0.016** | 0.351 ± 0.013 | 0.61 | <0.001 |

- DistMult (no GNN): best run 3 → AUROC 0.788, AUPRC 0.843, MRR 0.424, Hits@10 0.480, M 0.664; mean best epoch 581  (`e1_DTI_distmult_20260918_153059.log`)
- R-GCN: best run 3 → AUROC 0.849, AUPRC 0.882, MRR 0.439, Hits@10 0.490, M 0.698; mean best epoch 654  (`e1_DTI_rgcn_20260918_130753.log`)
- CompGCN: best run 11 → AUROC 0.831, AUPRC 0.876, MRR 0.369, Hits@10 0.476, M 0.664; mean best epoch 560  (`e1_DTI_compgcn_20260918_135627.log`)

## Task TREATS

| Metric | DistMult (no GNN) (n=12) | R-GCN (n=12) | CompGCN (n=12) | p R-GCN vs DistMult | p CompGCN vs DistMult |
|---|---|---|---|---|---|
| AUROC | **0.925 ± 0.001** | 0.885 ± 0.005 | 0.911 ± 0.001 | <0.001 | <0.001 |
| AUPRC | **0.947 ± 0.001** | 0.913 ± 0.002 | 0.924 ± 0.001 | <0.001 | <0.001 |
| MRR | **0.692 ± 0.002** | 0.348 ± 0.039 | 0.229 ± 0.010 | <0.001 | <0.001 |
| Hits@1 | **0.676 ± 0.003** | 0.294 ± 0.038 | 0.178 ± 0.010 | <0.001 | <0.001 |
| Hits@3 | **0.696 ± 0.002** | 0.367 ± 0.042 | 0.237 ± 0.011 | <0.001 | <0.001 |
| Hits@10 | **0.720 ± 0.002** | 0.450 ± 0.040 | 0.327 ± 0.011 | <0.001 | <0.001 |
| M | **0.841 ± 0.001** | 0.682 ± 0.014 | 0.644 ± 0.005 | <0.001 | <0.001 |
| warm_MRR | **0.698 ± 0.002** | 0.352 ± 0.039 | 0.232 ± 0.010 | <0.001 | <0.001 |
| dedup_MRR | **0.673 ± 0.003** | 0.336 ± 0.037 | 0.223 ± 0.010 | <0.001 | <0.001 |
| dedup_stereo_MRR | **0.648 ± 0.003** | 0.319 ± 0.033 | 0.214 ± 0.009 | <0.001 | <0.001 |

- DistMult (no GNN): best run 0 → AUROC 0.925, AUPRC 0.948, MRR 0.696, Hits@10 0.724, M 0.843; mean best epoch 1498  (`e1_TREATS_distmult_20260920_212826.log`)
- R-GCN: best run 5 → AUROC 0.876, AUPRC 0.909, MRR 0.411, Hits@10 0.507, M 0.703; mean best epoch 755  (`e1_TREATS_rgcn_20260918_151619.log`)
- CompGCN: best run 0 → AUROC 0.913, AUPRC 0.927, MRR 0.248, Hits@10 0.349, M 0.653; mean best epoch 1496  (`e1_TREATS_compgcn_20260920_123200.log`)
