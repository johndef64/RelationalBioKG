#!/usr/bin/env bash
# =============================================================================
# !!! DEPRECATED (2026-09-14) — DO NOT RUN !!!
# HeterogeneousCompGCN never receives num_bases (train_and_eval.py get_model), so every
# DIAG-TREATS-nb* config below trains the SAME model: this script cannot test anything.
# The CompGCN-TREATS collapse is examined by E0 instead (experiments/e0_protocol_compare.sh:
# removing the x5 duplicated TREATS edges from the graph and the disjoint supervision variants).
# See docs/piano_consolidamento_v2.md (P8).
# =============================================================================
echo "[diag] DEPRECATED: CompGCN ignores num_bases, this diagnostic is a no-op. See docs/piano_consolidamento_v2.md (P8)."; exit 1
# =============================================================================
# DIAGNOSTIC (standalone, optional) — CompGCN collapse on TREATS: capacity or mismatch?
# =============================================================================
#
# WHY THIS EXISTS (evaluation)
# ----------------------------
# In the HPO, ALL 15 CompGCN-TREATS trials collapsed to test_MRR ~0.038 (range
# 0.033-0.045), while AUPRC stayed fine (~0.81) and RGCN-TREATS reached MRR ~0.246
# (AUPRC ~0.93). Three measured facts:
#   1. Systematic across the whole search space -> NOT undertuning; structural.
#   2. AUPRC ok, MRR ~0 -> CompGCN SEPARATES positives but cannot RANK the true
#      disease near the top (it's a ranking failure, not a classification one).
#   3. CompGCN works on DTI (MRR mean 0.318) -> the problem is specific to TREATS.
#
# INTERPRETATION under test here:
#   CompGCN shares ONE composition operator (sub/corr) across all relations. TREATS
#   is promiscuous (4328 drugs x 4480 diseases) and semantically distant from the
#   molecular CORE (PPI/GO/pathway), so the shared operator can't specialize it and
#   the drug-disease signal gets washed out. RGCN's relation-specific weight matrices
#   give the per-relation capacity TREATS needs. `num_bases` is CompGCN's relation-
#   basis capacity knob -> the closest lever to test the "capacity bottleneck" claim.
#
# DECISION (see docs/report_HPO_risultati_finali.md §2bis):
#   The recommended course is to LEAVE CompGCN-TREATS as-is and report it as the
#   baseline / interpretable negative result (it justifies RGCN as primary). This
#   script is NOT meant to "rescue" the baseline — even a success would stay below
#   RGCN (0.246) and would break comparability with the DTI CompGCN / PathogenKG.
#   It is a *diagnostic* to strengthen the interpretation for the paper, if wanted.
#
# READING THE OUTPUT:
#   - test_MRR rises with num_bases  -> per-relation CAPACITY was the bottleneck.
#   - test_MRR stays flat ~0.04      -> it's a STRUCTURAL mismatch (shared operator),
#                                       not mere capacity -> interpretation confirmed.
#   Either way it's reported as diagnostic evidence, not as a new baseline.
#
# USAGE:
#   bash experiments/diag_compgcn_treats.sh
#   NB_LIST="20 30 50" DIAG_RUNS=2 DIAG_EPOCHS=200 bash experiments/diag_compgcn_treats.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/config.sh   # activates gnn env; exports TSV_B, TASK_B, PATIENCE, LOG_DIR

DIAG_RUNS="${DIAG_RUNS:-2}"        # few seeds: this is a trend check, not a final result
DIAG_EPOCHS="${DIAG_EPOCHS:-200}"
NB_LIST="${NB_LIST:-20 30 50}"     # 20 = HPO best (top of the search space); 30/50 = beyond it
PARAMS="src/models_params.json"

# --- inject temp DIAG-* configs; restore the params file on ANY exit -----------
BACKUP="$(mktemp)"
cp "$PARAMS" "$BACKUP"
restore () { cp "$BACKUP" "$PARAMS"; rm -f "$BACKUP"; echo "[diag] restored $PARAMS (temp configs removed)"; }
trap restore EXIT

python - "$PARAMS" "$NB_LIST" <<'PY'
import json, sys
path, nbs = sys.argv[1], [int(x) for x in sys.argv[2].split()]
d = json.load(open(path))
if "PKT-TREATS-best" not in d or "compgcn" not in d["PKT-TREATS-best"]:
    sys.exit("[diag] ERROR: PKT-TREATS-best/compgcn not found in " + path)
base = dict(d["PKT-TREATS-best"]["compgcn"])       # start from the tuned TREATS CompGCN
for nb in nbs:
    cfg = dict(base); cfg["num_bases"] = nb
    d[f"DIAG-TREATS-nb{nb}"] = {"compgcn": cfg}
json.dump(d, open(path, "w"), indent=4)
print("[diag] injected:", ", ".join(f"DIAG-TREATS-nb{nb}" for nb in nbs))
PY

# --- run one training per num_bases -------------------------------------------
for nb in $NB_LIST; do
  cfg="DIAG-TREATS-nb${nb}"
  log="${LOG_DIR}/diag_compgcn_treats_nb${nb}_$(date +%Y%m%d_%H%M%S).log"
  echo "[diag] === CompGCN / TREATS / num_bases=${nb} -> $log ==="
  run_logged "$log" python train_and_eval.py \
    --tsv "$TSV_B" --task "$TASK_B" --model compgcn --config "$cfg" \
    --runs "$DIAG_RUNS" --epochs "$DIAG_EPOCHS" \
    --early_stopping --patience "$PATIENCE" --negative_sampling filtered --eval_filtered \
    --oversample_rate 5 --undersample_rate 0.5 --alpha 0.25 --gamma 3.0 --alpha_adv 2.0
done

echo "[diag] done. Compare 'Test MRR' across num_bases values in ${LOG_DIR}/diag_compgcn_treats_*.log"
echo "[diag] Reference: RGCN-TREATS test_MRR ~0.246 ; CompGCN-TREATS HPO best ~0.035 (num_bases=20)."
echo "[diag]   MRR climbs with num_bases -> capacity bottleneck confirmed."
echo "[diag]   MRR flat ~0.04            -> structural mismatch (shared composition operator)."
