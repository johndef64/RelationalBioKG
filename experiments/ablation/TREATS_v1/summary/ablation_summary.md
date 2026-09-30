# E3 ablation summary

Mean ± sample-std (n−1) over runs. Significance = Holm-adjusted **paired** t-test vs the family's `*_full` reference: `**` p<0.01, `*` p<0.05 (two-sided). Δ and both tests (t-test + Wilcoxon, raw + Holm) are in `ablation_summary.csv`.

## Component ablation

| tag | n | AUROC | AUPRC | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|---|
| comp_full  _(ref)_ | 5 | 0.884 ± 0.006 | 0.912 ± 0.003 | 0.348 ± 0.034 | 0.295 ± 0.033 | 0.367 ± 0.037 | 0.450 ± 0.037 |
| comp_neg1 | 5 | 0.898 ± 0.002 | 0.916 ± 0.003 | 0.225 ± 0.018** | 0.169 ± 0.016** | 0.237 ± 0.019** | 0.334 ± 0.024** |
| comp_no_adv | 5 | 0.901 ± 0.002 | 0.919 ± 0.001 | 0.265 ± 0.029* | 0.205 ± 0.026* | 0.280 ± 0.032* | 0.378 ± 0.036* |
| comp_no_disjoint | 5 | 0.899 ± 0.002 | 0.919 ± 0.001 | 0.421 ± 0.018 | 0.377 ± 0.020 | 0.438 ± 0.018 | 0.497 ± 0.014 |
| comp_no_focal | 5 | 0.892 ± 0.008 | 0.915 ± 0.006 | 0.431 ± 0.024 | 0.378 ± 0.027 | 0.454 ± 0.025 | 0.529 ± 0.018 |
| comp_undersample05 | 5 | 0.891 ± 0.003 | 0.916 ± 0.001 | 0.333 ± 0.024 | 0.277 ± 0.024 | 0.353 ± 0.026 | 0.438 ± 0.024 |

## Relational-context ablation

| tag | n | AUROC | AUPRC | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|---|
| ctx_full  _(ref)_ | 5 | 0.884 ± 0.006 | 0.912 ± 0.003 | 0.348 ± 0.034 | 0.295 ± 0.033 | 0.367 ± 0.037 | 0.450 ± 0.037 |
| ctx_no_biochem | 5 | 0.887 ± 0.007 | 0.915 ± 0.003 | 0.349 ± 0.047 | 0.294 ± 0.044 | 0.370 ± 0.052 | 0.451 ± 0.051 |
| ctx_no_disease_ctx | 5 | 0.890 ± 0.003 | 0.915 ± 0.002 | 0.325 ± 0.027 | 0.271 ± 0.025 | 0.341 ± 0.030 | 0.426 ± 0.031 |
| ctx_no_go | 5 | 0.888 ± 0.006 | 0.915 ± 0.002 | 0.347 ± 0.039 | 0.293 ± 0.039 | 0.365 ± 0.044 | 0.447 ± 0.040 |
| ctx_no_pathway | 5 | 0.886 ± 0.004 | 0.914 ± 0.002 | 0.359 ± 0.021 | 0.302 ± 0.020 | 0.381 ± 0.023 | 0.465 ± 0.024 |
| ctx_no_pharma | 5 | 0.884 ± 0.004 | 0.913 ± 0.001 | 0.356 ± 0.028 | 0.301 ± 0.026 | 0.376 ± 0.031 | 0.457 ± 0.030 |
| ctx_no_ppi | 5 | 0.884 ± 0.007 | 0.913 ± 0.004 | 0.363 ± 0.037 | 0.308 ± 0.038 | 0.385 ± 0.039 | 0.466 ± 0.036 |

## Effects against the reference

Tests follow the pre-registration `PREREGISTRATION.json` (last modified 2026-09-24 18:37:31): primary variants are tested **comp: less, ctx: two-sided** with Holm among primaries only; secondary variants get a two-sided raw p and no star; equivalence is TOST against ±margin.

### MRR (equivalence margin ±0.02)

| tag | role | Δ | 95% CI | p (Holm if primary) | p equivalence | verdict |
|---|---|---|---|---|---|---|
| comp_neg1 | primary | -0.123 | [-0.173, -0.073] | 0.00364 |  | effect confirmed |
| comp_no_adv | primary | -0.084 | [-0.138, -0.029] | 0.0134 |  | effect confirmed |
| comp_no_disjoint | secondary | +0.072 | [+0.036, +0.109] | 0.00537 |  | secondary (descriptive) |
| comp_no_focal | primary | +0.082 | [+0.022, +0.143] | 0.99 |  | not shown |
| comp_undersample05 | secondary | -0.015 | [-0.065, +0.035] | 0.444 |  | secondary (descriptive) |
| ctx_no_biochem | secondary | +0.001 | [-0.047, +0.049] | 0.965 | 0.165 | secondary (descriptive) |
| ctx_no_disease_ctx | primary | -0.024 | [-0.065, +0.018] | 0.381 |  | not shown |
| ctx_no_go | secondary | -0.002 | [-0.073, +0.070] | 0.948 | 0.259 | secondary (descriptive) |
| ctx_no_pathway | secondary | +0.011 | [-0.048, +0.069] | 0.64 | 0.339 | secondary (descriptive) |
| ctx_no_pharma | primary | +0.007 | [-0.047, +0.061] | 0.723 |  | not shown |
| ctx_no_ppi | secondary | +0.015 | [-0.008, +0.038] | 0.145 | 0.29 | secondary (descriptive) |

### AUROC (equivalence margin ±0.01)

| tag | role | Δ | 95% CI | p (Holm if primary) | p equivalence | verdict |
|---|---|---|---|---|---|---|
| comp_neg1 | primary | +0.014 | [+0.008, +0.020] | 1 |  | not shown |
| comp_no_adv | primary | +0.017 | [+0.007, +0.026] | 1 |  | not shown |
| comp_no_disjoint | secondary | +0.015 | [+0.007, +0.023] | 0.00649 |  | secondary (descriptive) |
| comp_no_focal | primary | +0.008 | [-0.007, +0.023] | 1 |  | not shown |
| comp_undersample05 | secondary | +0.007 | [-0.003, +0.017] | 0.128 |  | secondary (descriptive) |
| ctx_no_biochem | secondary | +0.003 | [-0.001, +0.007] | 0.128 | 0.00396 | equivalent to reference |
| ctx_no_disease_ctx | primary | +0.006 | [-0.004, +0.016] | 0.325 |  | not shown |
| ctx_no_go | secondary | +0.004 | [-0.007, +0.016] | 0.373 | 0.119 | secondary (descriptive) |
| ctx_no_pathway | secondary | +0.002 | [-0.010, +0.014] | 0.665 | 0.0678 | secondary (descriptive) |
| ctx_no_pharma | primary | +0.000 | [-0.009, +0.009] | 0.954 |  | not shown |
| ctx_no_ppi | secondary | +0.000 | [-0.006, +0.006] | 0.934 | 0.00621 | equivalent to reference |

