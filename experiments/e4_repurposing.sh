#!/usr/bin/env bash
# E4 — Compound-centric drug repurposing + reasoned biological validation
#      (PathogenKG §3.4 / Table 5 + §5 tiered plausibility, adapted to PKT).
#
# 1) drug_eval.py ranks EVERY candidate target for each compound with the mature model:
#      Task A -> ranks all Proteins   (novel drug-target hypotheses)
#      Task B -> ranks all Diseases   (novel drug-disease repurposing hypotheses)
# 2) interpret_predictions.py = INTERPRETABILITY (not validation): grounds each NOVEL top-k
#    link in KG evidence + reports held-out recovery. NB the KG evidence is circular w.r.t.
#    the model; only held-out recovery is non-circular. For expert 3-tier review use the
#    driver expert_review_script.py (writes a sheet with empty expert columns).
#
# Usage:  bash experiments/e4_repurposing.sh A models/dti_pkt_taska_dti_<ts>
#         bash experiments/e4_repurposing.sh B models/treats_pkt_taskb_treats_<ts>
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/config.sh

WHICH="${1:?usage: $0 <A|B> <model_folder> [topk]}"
MODEL_FOLDER="${2:?path to trained model folder (from E1)}"
TOPK="${3:-20}"

if [ "$WHICH" = "A" ]; then TSV="$TSV_A"; TASK="$TASK_A"; TGT="$TGT_A"
else                        TSV="$TSV_B"; TASK="$TASK_B"; TGT="$TGT_B"; fi

log="${LOG_DIR}/e4_${WHICH}_$(date +%Y%m%d_%H%M%S).log"
# CANDIDATE_POOL: relation (default, as in the paper: only nodes occurring in the target relation, the
# pool of training negatives and of the evaluation) or all (every node of the target type).
CANDIDATE_POOL="${CANDIDATE_POOL:-relation}"
echo "[E4] task=$WHICH model=$MODEL_FOLDER target_type=$TGT pool=$CANDIDATE_POOL -> $log"

# 1) rank all compounds against all targets of the right type
run_logged "$log" python drug_eval.py --model_folder "$MODEL_FOLDER" --tsv "$TSV" --task "$TASK" \
  --target_type "$TGT" --compound all --topk "$TOPK" --candidate_pool "$CANDIDATE_POOL"

# 2) interpretability: KG-evidence + held-out recovery on the novel links
RANKINGS=$(ls -t "${MODEL_FOLDER}"/drug_eval_results/*rankings*.json | head -1)
run_logged_append "$log" python experiments/interpret_predictions.py --rankings "$RANKINGS" --tsv "$TSV" \
  --task_type "$WHICH" --topk "$TOPK"

echo "[E4] done. See ${MODEL_FOLDER}/drug_eval_results/ (rankings, *_interpreted.csv)."
echo "[E4] For expert 3-tier review, set MODEL_FOLDER/TASK in expert_review_script.py and run it."
