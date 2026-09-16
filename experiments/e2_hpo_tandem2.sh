#!/usr/bin/env bash
# E2 (tandem 2) — both hyperparameter sweeps with an ASYMMETRIC budget, then the extraction.
#
# Same idea as e2_hpo_tandem.sh (Task A then Task B, sequential: one GPU can't hold two sweeps),
# but the two tasks no longer share one trial budget. They don't cost the same and they don't carry
# the same prior knowledge:
#
#   Task A (DTI)     10.305 target edges / 1,16 M edges   -> an R-GCN trial is a couple of minutes.
#                    The target relation was rebuilt from scratch (DrugBank via UniProt) and the
#                    learning-rate grid now reaches 1e-1, a region no earlier trial ever visited,
#                    so this sweep EXPLORES. 30 trials per model.
#   Task B (TREATS)  168.157 target edges / 1,87 M edges  -> roughly ten times the cost per trial,
#                    CompGCN on the full graph is the OOM-prone case. 15 trials per model, even
#                    though its history is thinner (the old sweep stopped at 22 trials, R-GCN only).
#
# Budget per model, not per task: three models (rgcn, compgcn, distmult) are swept for each task.
# Ceiling 500 epochs, patience 10 evaluations (v2 defaults of e2_hpo_sweep.sh).
#
# Usage:
#   bash experiments/e2_hpo_tandem2.sh              # A (30) then B (15), then extraction
#   bash experiments/e2_hpo_tandem2.sh A            # only Task A, then its extraction
#   PKT_HPO_RUNS_A=40 PKT_HPO_RUNS_B=20 bash experiments/e2_hpo_tandem2.sh
#   HPO_SUFFIX=-v2c bash experiments/e2_hpo_tandem2.sh          # a different sweep generation
#
# Writes: W&B projects RelationalPKT-<TASK><SUFFIX>-<model>, configs PKT-<TASK>-best<SUFFIX>
#         in src/models_params.json, reports in experiments/hpo_best/.
# Prereq: wandb login
set -uo pipefail
cd "$(dirname "$0")/.."

WHICH="${1:-AB}"
export HPO_SUFFIX="${HPO_SUFFIX:--v2b}"
PKT_HPO_RUNS_A="${PKT_HPO_RUNS_A:-30}"
PKT_HPO_RUNS_B="${PKT_HPO_RUNS_B:-15}"

# e2_hpo_sweep.sh defaults to 100 runs/model when it is launched on its own: always set these.
export PKT_HPO_PROTOCOL="${PKT_HPO_PROTOCOL:-v2}"
if [ "$PKT_HPO_PROTOCOL" = "v2" ]; then
  export PKT_HPO_EPOCHS="${PKT_HPO_EPOCHS:-500}"
  export PKT_HPO_PATIENCE="${PKT_HPO_PATIENCE:-10}"
  N_MODELS=$(echo "${PKT_HPO_MODELS:-rgcn compgcn distmult}" | wc -w)
else
  export PKT_HPO_EPOCHS="${PKT_HPO_EPOCHS:-200}"
  export PKT_HPO_PATIENCE="${PKT_HPO_PATIENCE:-6}"
  N_MODELS=$(echo "${PKT_HPO_MODELS:-rgcn compgcn}" | wc -w)
fi

case "$WHICH" in
  A|B|AB) ;;
  *) echo "usage: $0 [A|B|AB]"; exit 1 ;;
esac

echo "[E2-tandem2] protocol=$PKT_HPO_PROTOCOL suffix=$HPO_SUFFIX models=$N_MODELS epochs=$PKT_HPO_EPOCHS patience=$PKT_HPO_PATIENCE"
[ "$WHICH" != "B" ] && echo "[E2-tandem2] Task A: $PKT_HPO_RUNS_A trials/model -> $((PKT_HPO_RUNS_A*N_MODELS)) runs"
[ "$WHICH" != "A" ] && echo "[E2-tandem2] Task B: $PKT_HPO_RUNS_B trials/model -> $((PKT_HPO_RUNS_B*N_MODELS)) runs"

sweep () {   # $1 = A|B   $2 = runs per model
  echo "[E2-tandem2] === Task $1 -> projects RelationalPKT-*${HPO_SUFFIX}-<model> ==="
  PKT_HPO_RUNS="$2" bash experiments/e2_hpo_sweep.sh "$1"
}

# NB: --suffix=-v2b (with '='): argparse would read a bare '-v2b' as an option, not as a value
extract () {   # $1 = DTI|TREATS
  python experiments/get_best_hpo_config.py --task "$1" --suffix="$HPO_SUFFIX" --write \
    || echo "[E2-tandem2] extraction failed for $1 — rerun: python experiments/get_best_hpo_config.py --task $1 --suffix=$HPO_SUFFIX --write"
}

[ "$WHICH" != "B" ] && sweep A "$PKT_HPO_RUNS_A"
[ "$WHICH" != "A" ] && sweep B "$PKT_HPO_RUNS_B"

echo "[E2-tandem2] extracting best configs from W&B ..."
[ "$WHICH" != "B" ] && extract DTI
[ "$WHICH" != "A" ] && extract TREATS

echo "[E2-tandem2] done. Check the configs landed:"
echo "  python -c \"import json;d=json.load(open('src/models_params.json'));print({k:list(v) for k,v in d.items() if k.endswith('${HPO_SUFFIX}')})\""
echo "[E2-tandem2] Then E1:  CFG_A=PKT-DTI-best${HPO_SUFFIX} CFG_B=PKT-TREATS-best${HPO_SUFFIX} PROTOCOL=v2 EPOCHS=1500 bash experiments/e1_main_training.sh"
