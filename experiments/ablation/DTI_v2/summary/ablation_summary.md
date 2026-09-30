# E3 ablation summary

Mean ± sample-std (n−1) over runs. Significance = Holm-adjusted **paired** t-test vs the family's `*_full` reference: `**` p<0.01, `*` p<0.05 (two-sided). Δ and both tests (t-test + Wilcoxon, raw + Holm) are in `ablation_summary.csv`.

## Component ablation

| tag | n | AUROC | AUPRC | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|---|
| comp_full  _(ref)_ | 10 | 0.844 ± 0.004 | 0.879 ± 0.003 | 0.409 ± 0.017 | 0.374 ± 0.021 | 0.420 ± 0.015 | 0.471 ± 0.012 |
| comp_neg1 | 10 | 0.847 ± 0.008 | 0.876 ± 0.005 | 0.257 ± 0.026** | 0.191 ± 0.028** | 0.280 ± 0.026** | 0.382 ± 0.022** |
| comp_no_adv | 10 | 0.845 ± 0.005 | 0.879 ± 0.003 | 0.378 ± 0.012** | 0.335 ± 0.015** | 0.394 ± 0.011** | 0.455 ± 0.009** |
| comp_no_disjoint | 10 | 0.819 ± 0.008 | 0.856 ± 0.005 | 0.362 ± 0.013 | 0.336 ± 0.014 | 0.369 ± 0.013 | 0.412 ± 0.012 |
| comp_no_focal | 10 | 0.848 ± 0.008 | 0.874 ± 0.008 | 0.368 ± 0.017** | 0.313 ± 0.023** | 0.394 ± 0.013** | 0.467 ± 0.007 |
| comp_undersample05 | 10 | 0.816 ± 0.005 | 0.856 ± 0.005 | 0.363 ± 0.021 | 0.331 ± 0.024 | 0.375 ± 0.018 | 0.420 ± 0.016 |

## Relational-context ablation

| tag | n | AUROC | AUPRC | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|---|
| ctx_core_ppi | 10 | 0.789 ± 0.003** | 0.831 ± 0.003** | 0.292 ± 0.028** | 0.257 ± 0.033** | 0.303 ± 0.025** | 0.353 ± 0.021** |
| ctx_full  _(ref)_ | 10 | 0.844 ± 0.004 | 0.879 ± 0.003 | 0.409 ± 0.017 | 0.374 ± 0.021 | 0.420 ± 0.015 | 0.471 ± 0.012 |
| ctx_no_biochem | 10 | 0.844 ± 0.006 | 0.879 ± 0.004 | 0.403 ± 0.024 | 0.367 ± 0.027 | 0.415 ± 0.022 | 0.469 ± 0.017 |
| ctx_no_drugctx | 10 | 0.843 ± 0.005 | 0.877 ± 0.003 | 0.383 ± 0.030** | 0.344 ± 0.036** | 0.397 ± 0.027** | 0.453 ± 0.019** |
| ctx_no_go | 10 | 0.839 ± 0.005 | 0.876 ± 0.005 | 0.405 ± 0.015 | 0.374 ± 0.016 | 0.415 ± 0.014 | 0.460 ± 0.012 |
| ctx_no_pathway | 10 | 0.840 ± 0.009 | 0.875 ± 0.007 | 0.396 ± 0.013 | 0.361 ± 0.016 | 0.408 ± 0.012 | 0.458 ± 0.011 |
| ctx_no_ppi | 10 | 0.848 ± 0.008 | 0.881 ± 0.006 | 0.409 ± 0.019 | 0.374 ± 0.025 | 0.423 ± 0.017 | 0.474 ± 0.011 |

## Effects against the reference

Tests follow the pre-registration `PREREGISTRATION.json` (last modified 2026-09-24 18:37:13): primary variants are tested **comp: less, ctx: less** with Holm among primaries only; secondary variants get a two-sided raw p and no star; equivalence is TOST against ±margin.

### MRR (equivalence margin ±0.02)

| tag | role | Δ | 95% CI | p (Holm if primary) | p equivalence | verdict |
|---|---|---|---|---|---|---|
| comp_neg1 | primary | -0.152 | [-0.175, -0.129] | 1.77e-07 |  | effect confirmed |
| comp_no_adv | primary | -0.031 | [-0.043, -0.019] | 0.000232 |  | effect confirmed |
| comp_no_disjoint | secondary | -0.046 | [-0.062, -0.030] | 9.68e-05 |  | secondary (descriptive) |
| comp_no_focal | primary | -0.041 | [-0.059, -0.023] | 0.000317 |  | effect confirmed |
| comp_undersample05 | secondary | -0.045 | [-0.065, -0.025] | 0.000633 |  | secondary (descriptive) |
| ctx_core_ppi | primary | -0.117 | [-0.145, -0.089] | 5.18e-06 |  | effect confirmed |
| ctx_no_biochem | secondary | -0.006 | [-0.022, +0.009] | 0.395 | 0.0361 | equivalent to reference |
| ctx_no_drugctx | primary | -0.025 | [-0.041, -0.009] | 0.00289 |  | effect confirmed |
| ctx_no_go | secondary | -0.004 | [-0.021, +0.013] | 0.637 | 0.0299 | equivalent to reference |
| ctx_no_pathway | secondary | -0.013 | [-0.025, -0.001] | 0.0404 | 0.106 | secondary (descriptive) |
| ctx_no_ppi | secondary | +0.001 | [-0.011, +0.012] | 0.896 | 0.00245 | equivalent to reference |

### AUROC (equivalence margin ±0.01)

| tag | role | Δ | 95% CI | p (Holm if primary) | p equivalence | verdict |
|---|---|---|---|---|---|---|
| comp_neg1 | primary | +0.002 | [-0.004, +0.008] | 1 |  | not shown |
| comp_no_adv | primary | +0.000 | [-0.001, +0.002] | 1 |  | not shown |
| comp_no_disjoint | secondary | -0.026 | [-0.031, -0.021] | 6.77e-07 |  | secondary (descriptive) |
| comp_no_focal | primary | +0.004 | [-0.000, +0.008] | 1 |  | not shown |
| comp_undersample05 | secondary | -0.028 | [-0.032, -0.025] | 3.87e-08 |  | secondary (descriptive) |
| ctx_core_ppi | primary | -0.056 | [-0.058, -0.053] | 5.45e-12 |  | effect confirmed |
| ctx_no_biochem | secondary | +0.000 | [-0.005, +0.005] | 1 | 0.000991 | equivalent to reference |
| ctx_no_drugctx | primary | -0.002 | [-0.007, +0.003] | 0.219 |  | not shown |
| ctx_no_go | secondary | -0.005 | [-0.011, +0.001] | 0.0769 | 0.0436 | equivalent to reference |
| ctx_no_pathway | secondary | -0.004 | [-0.011, +0.002] | 0.184 | 0.0443 | equivalent to reference |
| ctx_no_ppi | secondary | +0.003 | [-0.003, +0.010] | 0.244 | 0.0194 | equivalent to reference |

